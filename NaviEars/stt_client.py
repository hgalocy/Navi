import subprocess
import requests
import time
import os

PC_IP = os.getenv("PC_IP", "192.168.1.126")   # <-- change this
PORT = int(os.getenv("PC_PORT", "8000"))

# Update this to match your arecord device:
# Use `arecord -l` to confirm.
ARECORD_DEVICE = os.getenv("ARECORD_DEVICE", "plughw:2,0")

SR = int(os.getenv("SR", "16000"))
SECONDS = int(os.getenv("SECONDS", "5"))
WAV_PATH = "/tmp/navi_utterance.wav"


def record_wav():
    # Records mono 16kHz signed 16-bit little endian WAV
    cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-f", "S16_LE",
        "-r", str(SR),
        "-c", "1",
        "-d", str(SECONDS),
        WAV_PATH
    ]
    print(f"[Xavier] Recording {SECONDS}s from {ARECORD_DEVICE} to {WAV_PATH} ...")
    subprocess.check_call(cmd)


def send_to_pc():
    url = f"http://{PC_IP}:{PORT}/transcribe"
    with open(WAV_PATH, "rb") as f:
        files = {"file": ("navi.wav", f, "audio/wav")}
        print(f"[Xavier] Sending audio to {url} ...")
        r = requests.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    # sanity check server
    health_url = f"http://{PC_IP}:{PORT}/health"
    try:
        h = requests.get(health_url, timeout=5).json()
        print("[Xavier] PC health:", h)
    except Exception as e:
        print("[Xavier] Could not reach PC server:", e)
        return

    while True:
        record_wav()
        result = send_to_pc()
        print("[NAVI HEARD] >", result.get("text", ""))
        print("----")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
