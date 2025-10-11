#!/usr/bin/env python3
import sys, json, subprocess, queue, threading, time, os, requests
from vosk import Model, KaldiRecognizer
import webrtcvad, collections, struct

# ==== CONFIG: adjust paths as needed ====
# Ears
VOSK_MODEL_DIR = os.environ.get("VOSK_MODEL_DIR", "/home/freezypaws/models/vosk/en-us")
# Mouth
PIPER_BIN = "/home/freezypaws/models/piper/piper"
PIPER_MODEL = "/home/freezypaws/models/piper/voices/en_US/amy_low/en_US-amy-low.onnx"
PIPER_CONFIG = "/home/freezypaws/models/piper/voices/en_US/amy_low/en_US-amy-low.onnx.json"
# Brain
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLAMA_MODEL_NAME = os.environ.get("LLAMA_MODEL_NAME", "local-llama")
LLAMA_SYS_PROMPT = os.environ.get("LLAMA_SYS_PROMPT",
    "You are Navi, a small friendly robot assistant. Be concise, helpful, and kind."
)

# If voice is multi-speaker, set SPEAKER_ID to an int, else None
SPEAKER_ID = None
APLAY_DEV = "default" #ie: "plughw:1,0"

# ASR audio format must match arecord command
SAMPLE_RATE = int(os.environ.get("ASR_RATE", "16000"))
BUF_SIZE = 4000  # bytes; matches arecord -F 20000 frame chunks fine enough
SPEAK_BACKOFF_SEC = 0.35  # small extra mute after TTS ends
MIN_UTTERANCE_CHARS = int(os.environ.get("MIN_UTTERANCE_CHARS", "8"))
MAX_UTTERANCE_CHARS = int(os.environ.get("MAX_UTTERANCE_CHARS", "512"))
# VAD webrtcvad stuff
VAD = webrtcvad.Vad(0)  # 0-3 (3=aggressive)
# assuming 16 kHz mono S16_LE
FRAME_MS = 30
BYTES_PER_FRAME = int(SAMPLE_RATE * 2 * FRAME_MS / 1000)  # 960 bytes at 16khz
TAIL_FRAMES = 25  # ~750 ms of trailing silence to finalize an utterance
MAX_FRAMES   = int(6_000 / FRAME_MS)
START_FRAMES = 2 # need only ~60 ms of speech to trigger

SPEAKING = False # Changes to true when Navi is talking

# ==== Personalities ====
PERSONALITIES = {
    "calm": {
        "prefix": ["Okay.", "Got it.", "Mm-hmm."],
        "style": {"length_scale": 1.05, "noise_scale": 0.3, "noise_w": 0.35, "sentence_silence": 0.25},
        "rephrase": lambda txt: f"I heard: {txt}."
    },
    "cheerful": {
        "prefix": ["Sweet!", "You got it!", "Nice!"],
        "style": {"length_scale": 0.98, "noise_scale": 0.5, "noise_w": 0.45, "sentence_silence": 0.18},
        "rephrase": lambda txt: f"You said “{txt}”, right?"
    },
    "playful": {
        "prefix": ["Hehe.", "Alrighty.", "Oki-doki."],
        "style": {"length_scale": 1.02, "noise_scale": 0.6, "noise_w": 0.5, "sentence_silence": 0.22},
        "rephrase": lambda txt: f"Echoing back: “{txt}”…"
    },
}
ACTIVE_PERSONALITY = "calm"   # change live with a voice command below

# ==== Audio ingest from stdin (arecord) ====
audio_q = queue.Queue()

def Think(user_text):
    """Send user_text to llama.cpp server and return assistant reply (string)."""
    payload = {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": LLAMA_SYS_PROMPT},
            {"role": "user", "content": user_text.strip()}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False
    }
    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # OpenAI-style response
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sorry, I had trouble talking to the language model: {e}"


def Reader():
    while True:
        data = sys.stdin.buffer.read(BYTES_PER_FRAME)
        if not data:
           break
        audio_q.put(data)

def Say(text, style):
    global SPEAKING
    if not text: #make sure we're saying something
        return

    SPEAKING = True
    try:
        # Build Piper command with style knobs
        piper_cmd = [
            PIPER_BIN,
            "--model", PIPER_MODEL,
            "--config", PIPER_CONFIG,
            "--length_scale", str(style["length_scale"]),
            "--noise_scale", str(style["noise_scale"]),
            "--noise_w", str(style["noise_w"]),
            "--sentence_silence", str(style["sentence_silence"]),
            "--threads", "2",
            "--output_file", "-"
            # stream wav to stdout
        ]
        if SPEAKER_ID is not None:
            piper_cmd += ["--speaker", str(SPEAKER_ID)]

        aplay_cmd = ["aplay", "-q", "-D", APLAY_DEV, "-"]

        # Pipe text -> piper -> aplay
        p1 = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(aplay_cmd, stdin=p1.stdout)
        p1.stdin.write((text.strip() + "\n").encode("utf-8"))
        p1.stdin.close()
        p2.wait()
        p1.wait()
        time.sleep(SPEAK_BACKOFF_SEC)  # let the speaker “ring down”
    finally:
        SPEAKING = False

def Log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)

def main():
    if not os.path.isdir(VOSK_MODEL_DIR):
        Log(f"ERROR: Vosk model path missing: {VOSK_MODEL_DIR}")
        sys.exit(1)
    Log(f"VOSK_MODEL_DIR={VOSK_MODEL_DIR}")
    Log(f"has rescore={os.path.isdir(os.path.join(VOSK_MODEL_DIR,'rescore'))}, "
    f"rnnlm={os.path.isdir(os.path.join(VOSK_MODEL_DIR,'rnnlm'))}, "
    f"ivector={os.path.isdir(os.path.join(VOSK_MODEL_DIR,'ivector'))}")

    Log("Navi is listening…")

    model = Model(VOSK_MODEL_DIR)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    rec.SetWords(False)
    rec.SetGrammar(r'''[
    "hey navi",
    "hey navi can you tell me my schedule",
    "can you tell me my schedule",
    "what's on my calendar",
    "open calendar",
    "what time is it",
    "what's the weather",
    "volume up",
    "volume down",
    "stop"
    ]''')

    # keep ~1s of pre-speech audio for natural start cut-in,
    # and track when we're currently in a voiced segment.
    ring = collections.deque(maxlen=int(600 / FRAME_MS))  # ~1 second of 20ms frames
    voiced = False
    tail_silence_frames = 0
    speech_frames = 0

    t = threading.Thread(target=Reader, daemon=True)
    t.start()

    try:
        partial = ""
        last_speech_time = time.time()
        while True:
            data = audio_q.get()
            #if Navi is speaking, drop frames (prevents feedback)
            if SPEAKING:
                ring.clear()
                voiced = False
                tail_silence_frames = 0
                speech_frames = 0
                continue
            # run VAD on each 20ms frame
            is_speech = VAD.is_speech(data, SAMPLE_RATE)

            if not voiced:
                if is_speech:
                    speech_frames += 1
                    # entering speech: push pre-roll context then current frame
                    if speech_frames >= START_FRAMES:
                        for c in ring:
                            rec.AcceptWaveform(c)
                        ring.clear()
                        rec.AcceptWaveform(data)
                        voiced = True
                        tail_silence_frames = 0
                else:
                    speech_frames = 0
                    ring.append(data)

            else:
                # already in an utterance
                if is_speech:
                    rec.AcceptWaveform(data)
                    tail_silence_frames = 0
                else:
                    # we were in speech; count trailing silence
                    tail_silence_frames += 1
                    if tail_silence_frames >= TAIL_FRAMES or speech_frames >= MAX_FRAMES:
                        # finalize utterance
                        result = json.loads(rec.Result())
                        text = (result.get("text") or "").strip().lower()
                        if text:
                            Log(f"Mic heard: {text}")
                            # make sure it's a good length of input
                            if len(text) >= MIN_UTTERANCE_CHARS:
                                if len(text) > MAX_UTTERANCE_CHARS:
                                    text = text[:MAX_UTTERANCE_CHARS]

                                if text.startswith("hey navi"):
                                    user = text[8:].strip()  # after wake word

                                    # Personality switching: "navi personality cheerful"
                                    global ACTIVE_PERSONALITY
                                    if user.startswith("navi personality "):
                                        choice = user.replace("navi personality", "").strip()
                                        if choice in PERSONALITIES:
                                            ACTIVE_PERSONALITY = choice
                                            Say(f"Personality set to {choice}.", PERSONALALITIES[choice]["style"])
                                        else:
                                            Say(f"I don't have a {choice} personality yet.", PERSONALITIES[ACTIVE_PERSONALITY]["style"])
                                    else:
                                        Log(f"You: {text}")
                                        persona = PERSONALITIES[ACTIVE_PERSONALITY]
                                        prefix = persona["prefix"][int(time.time()) % len(persona["prefix"])]
                                        reply = f"{prefix} {Think(user)}"
                                        Log(f"Navi: {reply}")
                                        Say(reply, persona["style"])
                        # reset for next utterance
                        rec.Reset()
                        ring.clear()
                        voiced = False
                        tail_silence_frames = 0
    except KeyboardInterrupt:
        pass
    
    # Flush any final result on EOF
    try:
        final = json.loads(rec.FinalResult()).get("text","").strip()
        if len(final) >= MIN_UTTERANCE_CHARS:
            log(f"You (final): {final}")
            reply = Think(final)
            log(f"Navi: {reply}")
            speak(reply)
    except Exception:
        pass

if __name__ == "__main__":
    main()

