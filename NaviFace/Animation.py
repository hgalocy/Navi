import pyglet
import moderngl
import numpy as np
import time
import math
import subprocess
import threading
import sys
import select
import os
import subprocess
from collections import deque


if "DISPLAY" not in os.environ:
    try:
        display = subprocess.check_output(
            "who | grep '(:' | head -n1 | awk '{print $NF}'",
            shell=True
        ).decode().strip("()\n")
        os.environ["DISPLAY"] = display
    except:
        os.environ["DISPLAY"] = ":0"
        
kPiperPath = "/home/freezypaws/piper/piper"
kModelPath = "/home/freezypaws/piper/voices/en_US/amy_low/en_US-amy-low.onnx"

# is_talking_ = False
audio_level_ = 0.0
current_mouth_ = 0.0
talk_hold_time_ = 0.18
prev_rms_ = 0.0
speech_impulse_ = 0.0
state_lock = threading.Lock()

# --- Audio format constants (Piper raw output) ---
SAMPLE_RATE = 22050
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # S16_LE
BYTES_PER_FRAME = CHANNELS * BYTES_PER_SAMPLE

# Read 256 *frames* per chunk (not 256 bytes).
CHUNK_FRAMES = 256
CHUNK_BYTES = CHUNK_FRAMES * BYTES_PER_FRAME
CHUNK_DT = CHUNK_FRAMES / SAMPLE_RATE  # ~11.6 ms

# --- Playback-timed envelope tracking ---
MAX_ENV_SECONDS = 120.0
MAX_ENV_SAMPLES = int(MAX_ENV_SECONDS / CHUNK_DT) + 10

env_ring_ = deque(maxlen=MAX_ENV_SAMPLES)  # holds tuples: (audio_t0, env)
talk_lock_ = threading.Lock()

play_start_mono_ = None           # monotonic time where playback clock was "armed"
gen_done_ = True                  # has Piper finished generating all chunks?
audio_written_s_ = 0.0            # audio timeline end (seconds)
audio_end_s_ = 0.0                # end of utterance in audio timeline
talk_active_ = False              # derived "speaking" state for animation

# Controls how far "ahead" we let writes get (keeps buffering bounded)
WRITE_AHEAD_LIMIT_S = 0.08        # 80ms

# Approximate offset between play clock and what you hear.
# Start with buffer-time + a little safety, then tune by eye.
PLAYBACK_LATENCY_S = 0.06

# Envelope follower parameters (attack/release)
ENV_ATTACK = 0.65                 # 0..1, higher = faster rise
ENV_RELEASE = 0.8                # 0..1, higher = faster fall


window = pyglet.window.Window(fullscreen=True)
width, height = window.get_framebuffer_size()
ctx = moderngl.create_context()

prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330
        uniform vec2 resolution;
        uniform vec2 gaze;
        uniform float headTilt;
        uniform float blink;
        uniform float mouthOpen;
        uniform float zoom; 
        uniform float time;
        
        out vec4 fragColor;

        float sdEllipse(vec2 p, vec2 r) {
            // Signed-ish distance for ellipse (good enough for soft masks)
            float k = length(p / r);
            return k - 1.0;
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;
            uv *= zoom;

            // subtle movement of whole face
            uv -= gaze * 0.015;

            // ---- Head Tilt Rotation ----
            float s = sin(headTilt);
            float c = cos(headTilt);
            uv = mat2(c,-s,s,c) * uv;

            float eye_offset = 0.33;
            float eye_y = 0.27;
            float eye_width = 0.23;
            float eye_height = 0.18;

            float dynamic_height = max(eye_height * (1.0 - blink), 0.01);

            // -------------------------
            // ---- Gaze to Move Eye Centers ----
            // -------------------------
            vec2 left_center  = vec2(-eye_offset, eye_y);
            vec2 right_center = vec2( eye_offset, eye_y);

            vec2 left_uv  = uv - left_center;
            vec2 right_uv = uv - right_center;

            vec2 pupil_offset = gaze * 0.15;
            pupil_offset = clamp(pupil_offset, vec2(-0.04,-0.02), vec2(0.04,0.02));
            vec2 left_pupil_uv  = uv - (left_center  + pupil_offset);
            vec2 right_pupil_uv = uv - (right_center + pupil_offset);

            float left_eye  = length(vec2(left_uv.x / eye_width,  left_uv.y / dynamic_height));
            float right_eye = length(vec2(right_uv.x / eye_width, right_uv.y / dynamic_height));

            float eye_mask = max(
                smoothstep(1.0, 0.88, left_eye),
                smoothstep(1.0, 0.88, right_eye)
            );

            // -------------------------
            // ---- Brighter Eye Core ----
            // -------------------------
            float left_pupil  = length(vec2(left_pupil_uv.x / (eye_width*0.7), left_pupil_uv.y / (eye_height*0.7)));
            float right_pupil = length(vec2(right_pupil_uv.x / (eye_width*0.7), right_pupil_uv.y / (eye_height*0.7)));

            float core_glow = max(
                smoothstep(0.65, 0.2, left_pupil),
                smoothstep(0.65, 0.2, right_pupil)
            );

            vec3 base_cyan = vec3(0.4, 1.0, 1.0);

            // -------------------------
            // ---- Rainbow cheeky arc (top tint) ----
            // -------------------------
            float arc = clamp(1.0 - uv.y * 2.5, 0.0, 1.0);
            vec3 rainbow_tint = vec3(0.8, 0.9, 1.0);

            vec3 eye_color = mix(base_cyan, rainbow_tint, arc * 0.4);

            vec3 pupil_color = vec3(0.8, 1.0, 1.0);
            eye_color += pupil_color * core_glow * 0.9;

            float highlight = max(
                smoothstep(0.25, 0.0, left_pupil),
                smoothstep(0.25, 0.0, right_pupil)
            );

            eye_color += vec3(1.0, 1.0, 1.0) * highlight * 0.35;

            vec3 final_color = eye_color * eye_mask;
            // -------------------------
            // Blush
            // -------------------------
            vec2 blush_r = vec2(0.18, 0.07);     // horizontal stretch oval
            float blush_y = eye_y - 0.25;

            // Move blush outward (cheeks)
            vec2 blush_left_center  = vec2(-eye_offset - 0.18, blush_y);
            vec2 blush_right_center = vec2( eye_offset + 0.18, blush_y);

            float bL = sdEllipse(uv - blush_left_center, blush_r);
            float bR = sdEllipse(uv - blush_right_center, blush_r);

            float blushMask = max(
                smoothstep(0.20, -0.10, bL),
                smoothstep(0.20, -0.10, bR)
            );

            // Keep blush subtle and soft
            vec3 blushCol = vec3(1.0, 0.25, 0.60);
            final_color += blushCol * blushMask * 0.22;

            // -------------------------
            // Mouth
            //--------------------------

            // mouth width
            vec2 mouth_center = vec2(0.0, -0.12);
            mouth_center.y += sin(time * 2.0) * 0.002;
            vec2 p = uv - mouth_center;

            // subtle organic mouth wobble when talking
            p.y += sin(p.x * 10.0) * mouthOpen * 0.01;

            // width limit
            float halfWidth = 0.16;
            float widthMask = 1.0 - smoothstep(halfWidth - 0.015, halfWidth, abs(p.x));

            // smile curve (top boundary)
            float top_curve = p.x * p.x * mix(1.2, 1.5, mouthOpen);
            top_curve -= mouthOpen * 0.015;

            // closed smile line
            float closedSmile =
                smoothstep(0.022, 0.0, abs(p.y - top_curve))
                * widthMask
                * (1.0 - smoothstep(0.02, 0.07, mouthOpen));
                
            // mouth opening depth
            float depth = max(mouthOpen * 0.13, 0.0001);

            // ellipse parameters
            float a = 0.16;      // width
            float b = depth;     // height

            // ellipse centered under the smile
            float ellipse =
                (p.x*p.x)/(a*a) +
                ((p.y - (top_curve - depth))*(p.y - (top_curve - depth)))/(b*b);

            // inside ellipse
            float ellipseEdge = smoothstep(1.10, 0.98, ellipse);
            float insideEllipse = smoothstep(1.12, 0.90, ellipse);

            // only draw bottom half
            insideEllipse *= step(p.y, top_curve);

            // only when talking
            insideEllipse *= smoothstep(0.02, 0.06, mouthOpen);

            // distance from the smile curve (used for glow)
            float distFromCurve = abs(p.y - top_curve);

            // main glow like the idle smile
            float topGlow = smoothstep(0.14, 0.0, distFromCurve);
            topGlow *= mix(1.0, 1.4, smoothstep(0.02, 0.07, mouthOpen));

            float bodyFill = insideEllipse * 0.25;
            float edgeGlow = ellipseEdge * 0.75;

            float glow =
                topGlow * closedSmile +     // fuzzy smile edge
                edgeGlow * insideEllipse +  // fuzzy ellipse edge
                bodyFill * insideEllipse;   // soft interior

            // clamp so it doesn't over-brighten
            glow = clamp(glow, 0.0, 1.0);

            vec3 mouthColor = vec3(0.4, 1.0, 1.0);

            final_color += mouthColor * glow * 0.9;
                fragColor = vec4(final_color, 1.0);
            }
                """,
)

start_time = time.time()
prog["resolution"].value = (width, height)
prog["zoom"].value = 1.35
prog["gaze"].value = (0.0, 0.0)
prog["headTilt"].value = 0.0
prog["blink"].value = 0.0
prog["mouthOpen"].value = 0.0
prog["time"].value = 0.0

quad = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype="f4")

vbo = ctx.buffer(quad.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

@window.event
def on_resize(w, h):
    prog["resolution"].value = (w, h)

#Make Navi look like she's looking around
next_gaze_time_ = 0.0
target_gaze_ = [0.0, 0.0]
current_gaze_ = [0.0 , 0.0]
def micro_jitter(t, speed=10.0, amount=0.004, offset=0.0):
    return math.sin(t * speed + offset) * amount

def update_gaze(t):
    global next_gaze_time_, target_gaze_, current_gaze_
    if t > next_gaze_time_:
        target_gaze_ = [
            np.random.uniform(-0.35, 0.35),
            np.random.uniform(-0.15, 0.15)
        ]
        next_gaze_time_ = t + np.random.uniform(1.5, 4.0)
    # smooth eye motion
    speed = 0.08
    current_gaze_[0] += (target_gaze_[0]-current_gaze_[0]) * speed
    current_gaze_[1] += (target_gaze_[1]-current_gaze_[1]) * speed

    jitter_x = micro_jitter(t, 11.3, 0.002)
    jitter_y = micro_jitter(t, 9.7, 0.002, 1.7)
    return [current_gaze_[0] + jitter_x, current_gaze_[1] + jitter_y]

#Make Navi blink every so often
next_blink_time_ = np.random.uniform(3, 7)
blink_start_ = -1.0
kBlinkDuration = 0.16
def update_blink(t):
    global next_blink_time_, blink_start_
    # start blink
    if blink_start_ < 0 and t > next_blink_time_:
        blink_start_ = t
    # if blinking
    if blink_start_ >= 0:
        progress = (t - blink_start_) / kBlinkDuration
        if progress >= 1.0:
            blink_start_ = -1.0
            next_blink_time_ = t + np.random.uniform(3,7)
            return 0.0
        return math.sin(progress * math.pi) ** 0.6
    return 0.0

#Head bobbing swaying slightly
target_tilt_ = 0
current_tilt_ = 0
next_head_move_ = 0
def update_head(t):
    global target_tilt_, current_tilt_, next_head_move_
    if t > next_head_move_:
        target_tilt_ = np.random.uniform(-0.07, 0.07)
        next_head_move_ = t + np.random.uniform(3, 6)
    current_tilt_ += (target_tilt_ - current_tilt_) * 0.02
    return current_tilt_ + micro_jitter(t, 6.5, 0.002)

def _reset_talk_state_locked():
    global env_ring_, play_start_mono_, gen_done_, audio_written_s_, audio_end_s_, talk_active_
    env_ring_.clear()
    play_start_mono_ = None
    gen_done_ = True
    audio_written_s_ = 0.0
    audio_end_s_ = 0.0
    talk_active_ = False

def _get_playback_time_s():
    """Estimate playback time in seconds, aligned to the utterance start."""
    with talk_lock_:
        if play_start_mono_ is None:
            return 0.0
        t = time.monotonic() - play_start_mono_ - PLAYBACK_LATENCY_S
    return max(0.0, t)

def _sample_env_at_playback_time(play_t):
    """
    Consume old samples and return the most recent envelope value at play_t.
    Assumes env_ring_ is time-ordered by audio_t0.
    """
    with talk_lock_:
        # If no samples have been written yet:
        if not env_ring_:
            return 0.0

        # Drop stale entries so env_ring_[0] is the latest <= play_t
        while len(env_ring_) >= 2 and env_ring_[1][0] <= play_t:
            env_ring_.popleft()

        return env_ring_[0][1]


#Navi is talking
def navi_speak(text):
    global talk_active_, audio_level_, prev_rms_, speech_impulse_
    global play_start_mono_, audio_written_s_, gen_done_, audio_end_s_

    with talk_lock_:
        gen_done_ = False
        audio_written_s_ = 0.0
        audio_end_s_ = 0.0
        play_start_mono_ = None
        env_ring_.clear()

    def run():
        global talk_active_, audio_level_, prev_rms_, speech_impulse_
        global play_start_mono_, audio_written_s_, gen_done_, audio_end_s_
        cmd = [
            kPiperPath,
            "--model", kModelPath,
            "--output_raw",
            "--threads", "2"
        ]
        piper = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", "plughw:0,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "--buffer-time=20000", "--period-time=5000"],
            stdin=subprocess.PIPE
        )
        try:
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()

            started = False
            local_prev_level = 0.0
            env = 0.0  # local envelope follower state

            while True:
                chunk = piper.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break

                if not started:
                    started = True
                    with talk_lock_:
                        play_start_mono_ = time.monotonic()
                    print("Navi started speaking")

                # Convert to float samples
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                if samples.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0)

                # Attack/Release envelope follower
                if rms > env:
                    env = env + (rms - env) * ENV_ATTACK
                else:
                    env = env + (rms - env) * ENV_RELEASE

                # Map to something usable (gain + clamp); tune this
                env_norm = min(env * 3.0, 1.0)

                # Timestamp on the audio timeline (not wall time)
                with talk_lock_:
                    audio_t0 = audio_written_s_
                    env_ring_.append((audio_t0, env_norm))
                    audio_written_s_ += (samples.size / SAMPLE_RATE)  # exact dt if last chunk short

                # --- Pace writes so we don't buffer too far ahead ---
                # This keeps wall clock and audio timeline aligned.
                while True:
                    with talk_lock_:
                        if play_start_mono_ is None:
                            break
                        elapsed = time.monotonic() - play_start_mono_
                        ahead = audio_written_s_ - elapsed
                    if ahead <= WRITE_AHEAD_LIMIT_S:
                        break
                    time.sleep(min(ahead - WRITE_AHEAD_LIMIT_S, 0.01))

                # Write to aplay after pacing
                aplay.stdin.write(chunk)

                # compute amplitude
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                if samples.size == 0:
                    continue
                rms = np.sqrt(np.mean(samples * samples)) / 32768.0
                level = min(rms * 3.2, 1.0)
                with state_lock:
                    if level > audio_level_:
                        audio_level_ = audio_level_ * 0.3 + level * 0.7
                    else:
                        audio_level_ = audio_level_ * 0.75 + level * 0.25
                rise = max(audio_level_ - local_prev_level, 0.0)
                fall = max(local_prev_level - audio_level_, 0.0)
                local_prev_level = audio_level_

                with state_lock:
                    speech_impulse_ = speech_impulse_ * 0.72 + rise * 3.8 - fall * 0.9
                    speech_impulse_ = max(0.0, min(1.0, speech_impulse_))
                prev_rms_ = rms
        
        finally:
            try:
                aplay.stdin.close()
                print("Navi finished generating (audio still may be playing)")
            except Exception:
                pass
            piper.wait()
            aplay.wait()

            audio_level_ = 0.0
            with talk_lock_:
                gen_done_ = True
                audio_end_s_ = audio_written_s_

            speech_impulse_ = 0.0
            prev_rms_ = 0.0
            print("Navi finished speaking")
        
    threading.Thread(target=run, daemon=True).start()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        navi_speak("Hello Hannah. I am Navi.")
    elif symbol == pyglet.window.key.ESCAPE:
        window.close()

@window.event
def on_draw():
    global talk_active_, current_mouth_, speech_impulse_

    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline().strip()
        if cmd:
            navi_speak(cmd)

    ctx.clear(0.0, 0.0, 0.0)
    t = time.time() - start_time
    prog["time"].value = t

    play_t = _get_playback_time_s()
    env_play = _sample_env_at_playback_time(play_t)
    with talk_lock_:
        still_playing = (not gen_done_) or (play_t < audio_end_s_)
        talk_active_ = still_playing
    if talk_active_:
        target_mouth = env_play
        prog["gaze"].value = (0.0, 0.0)
        prog["headTilt"].value = update_head(t) + target_mouth * 0.01
    else:
        prog["gaze"].value = update_gaze(t)
        prog["headTilt"].value = update_head(t)
        target_mouth = 0.0


    # smooth transition
    current_mouth_ += (target_mouth - current_mouth_) * 0.25
    prog["mouthOpen"].value = current_mouth_

    # ---- Subtle breathing motion ----
    prog["zoom"].value = 1.35 + math.sin(t * 0.25) * 0.01

    # ---- Blinking ----
    prog["blink"].value = update_blink(t)

    vao.render(moderngl.TRIANGLE_STRIP)

pyglet.app.run()