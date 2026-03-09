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

# ---- Calculations made for mouth movement on speaking ----
audio_level_ = 0.0
current_mouth_ = 0.0
speech_impulse_ = 0.0
state_lock = threading.Lock()

# --- Audio format constants (Piper raw output) ---
kSampleRate = 22050
kChannels = 1
kBytesPerSample = 2  # S16_LE
kBytesPerFrame = kChannels * kBytesPerSample
# Read 256 *frames* per chunk (not 256 bytes).
kChunkFrames = 256
kChunkBytes = kChunkFrames * kBytesPerFrame
kChunkDt = kChunkFrames / kSampleRate  # ~11.6 ms

# --- Playback-timed envelope tracking ---
kMaxEnvSeconds = 120.0
kMaxEnvSamples = int(kMaxEnvSeconds / kChunkDt) + 10

env_ring_ = deque(maxlen=kMaxEnvSamples)  # holds tuples: (audio_t0, env)
talk_lock_ = threading.Lock()

play_start_mono_ = None           # monotonic time where playback clock was "armed"
gen_done_ = True                  # has Piper finished generating all chunks?
audio_written_s_ = 0.0            # audio timeline end (seconds)
audio_end_s_ = 0.0                # end of utterance in audio timeline
talk_active_ = False              # derived "speaking" state for animation

# Controls how far "ahead" we let writes get (keeps buffering bounded)
kWriteAheadLimit = 0.08        # 80ms

# Approximate offset between play clock and what you hear.
# Start with buffer-time + a little safety, then tune by eye.
kPlaybackLatency = 0.06

# Envelope follower parameters (attack/release)
kEnvAttack = 0.65                 # 0..1, higher = faster rise
kEnvRelease = 0.8                # 0..1, higher = faster fall


window_ = pyglet.window.Window(fullscreen=True)
width_, height_ = window_.get_framebuffer_size()
ctx_ = moderngl.create_context()

prog_ = ctx_.program(
    vertex_shader="""
        #version 330
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330
        uniform vec2 resolution_;
        uniform vec2 gaze_;
        uniform float head_tilt_;
        uniform float blink_;
        uniform float mouth_open_;
        uniform float zoom_; 
        uniform float time_;
        
        out vec4 frag_color_;

        // Draw an ellipse
        float SdEllipipse(vec2 p, vec2 r) {
            float k = length(p / r);
            return k - 1.0;
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * resolution_) / resolution_.y;
            uv *= zoom_;

            // -------------------------
            // Whole Head Movement
            // -------------------------
            // subtle movement of whole face
            uv -= gaze_ * 0.015;
            // head rotation
            float s = sin(head_tilt_);
            float c = cos(head_tilt_);
            uv = mat2(c,-s,s,c) * uv;

            // -------------------------
            // Eyes
            // -------------------------
            vec3 base_cyan = vec3(0.4, 1.0, 1.0);
            float eye_offset = 0.33;
            float eye_y = 0.27;
            float eye_width = 0.23;
            float eye_height = 0.18;
            float dynamic_height = max(eye_height * (1.0 - blink_), 0.01);
            // move eye centers (gaze)
            vec2 left_center  = vec2(-eye_offset, eye_y);
            vec2 right_center = vec2( eye_offset, eye_y);
            vec2 left_uv  = uv - left_center;
            vec2 right_uv = uv - right_center;
            vec2 pupil_offset = gaze_ * 0.15;
            pupil_offset = clamp(pupil_offset, vec2(-0.04,-0.02), vec2(0.04,0.02));
            vec2 left_pupil_uv  = uv - (left_center  + pupil_offset);
            vec2 right_pupil_uv = uv - (right_center + pupil_offset);
            float left_eye  = length(vec2(left_uv.x / eye_width,  left_uv.y / dynamic_height));
            float right_eye = length(vec2(right_uv.x / eye_width, right_uv.y / dynamic_height));
            float eye_mask = max(
                smoothstep(1.0, 0.88, left_eye),
                smoothstep(1.0, 0.88, right_eye)
            );
            // brighter eye core
            float left_pupil  = length(vec2(left_pupil_uv.x / (eye_width*0.7), left_pupil_uv.y / (eye_height*0.7)));
            float right_pupil = length(vec2(right_pupil_uv.x / (eye_width*0.7), right_pupil_uv.y / (eye_height*0.7)));
            float core_glow = max(
                smoothstep(0.65, 0.2, left_pupil),
                smoothstep(0.65, 0.2, right_pupil)
            );
            // rainbow cheeky arc (top tint)
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
            vec2 blush_r = vec2(0.18, 0.07); // horizontal stretch oval
            float blush_y = eye_y - 0.25;
            // move blush outward to cheeks
            vec2 blush_left_center  = vec2(-eye_offset - 0.18, blush_y);
            vec2 blush_right_center = vec2( eye_offset + 0.18, blush_y);
            float bl = SdEllipipse(uv - blush_left_center, blush_r);
            float br = SdEllipipse(uv - blush_right_center, blush_r);
            float blush_mask = max(
                smoothstep(0.20, -0.10, bl),
                smoothstep(0.20, -0.10, br)
            );
            // keep blush subtle and soft
            vec3 blush_col = vec3(1.0, 0.25, 0.60);
            final_color += blush_col * blush_mask * 0.22;

            // -------------------------
            // Mouth
            //--------------------------
            vec2 mouth_center = vec2(0.0, -0.12);
            mouth_center.y += sin(time_ * 2.0) * 0.002;
            vec2 p = uv - mouth_center;
            // subtle organic mouth wobble when talking
            p.y += sin(p.x * 10.0) * mouth_open_ * 0.01;
            // width limit
            float half_width = 0.16;
            float width_mask = 1.0 - smoothstep(half_width - 0.015, half_width, abs(p.x));
            // smile curve (top boundary)
            float top_curve = p.x * p.x * mix(1.2, 1.5, mouth_open_);
            top_curve -= mouth_open_ * 0.015;
            // closed smile line
            float closed_smile =
                smoothstep(0.022, 0.0, abs(p.y - top_curve))
                * width_mask
                * (1.0 - smoothstep(0.02, 0.07, mouth_open_));
            // mouth opening depth
            float depth = max(mouth_open_ * 0.13, 0.0001);
            // ellipse parameters
            float a = 0.16;      // width
            float b = depth;     // height
            // ellipse centered under the smile
            float ellipse =
                (p.x*p.x)/(a*a) +
                ((p.y - (top_curve - depth))*(p.y - (top_curve - depth)))/(b*b);
            // inside ellipse
            float ellipse_edge = smoothstep(1.10, 0.98, ellipse);
            float inside_ellipse = smoothstep(1.12, 0.90, ellipse);
            // only draw bottom half
            inside_ellipse *= step(p.y, top_curve);
            // only when talking
            inside_ellipse *= smoothstep(0.02, 0.06, mouth_open_);
            // distance from the smile curve (used for glow)
            float dist_from_curve = abs(p.y - top_curve);
            // main glow like the idle smile
            float top_glow = smoothstep(0.14, 0.0, dist_from_curve);
            top_glow *= mix(1.0, 1.4, smoothstep(0.02, 0.07, mouth_open_));
            float body_fill = inside_ellipse * 0.25;
            float edge_glow = ellipse_edge * 0.75;
            float glow =
                top_glow * closed_smile +     // fuzzy smile edge
                edge_glow * inside_ellipse +  // fuzzy ellipse edge
                body_fill * inside_ellipse;   // soft interior
            // clamp so it doesn't over-brighten
            glow = clamp(glow, 0.0, 1.0);
            vec3 mouth_color = vec3(0.4, 1.0, 1.0);
            final_color += mouth_color * glow * 0.9;
            
            frag_color_ = vec4(final_color, 1.0);
        }""",
)

start_time_ = time.time()
prog_["resolution_"].value = (width_, height_)
prog_["zoom_"].value = 1.35
prog_["gaze_"].value = (0.0, 0.0)
prog_["head_tilt_"].value = 0.0
prog_["blink_"].value = 0.0
prog_["mouth_open_"].value = 0.0
prog_["time_"].value = 0.0

quad_ = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype="f4")

vbo_ = ctx_.buffer(quad_.tobytes())
vao_ = ctx_.simple_vertex_array(prog_, vbo_, "in_pos")

@window_.event
def on_resize(w, h):
    prog_["resolution_"].value = (w, h)

#Make Navi look like she's looking around
next_gaze_time_ = 0.0
target_gaze_ = [0.0, 0.0]
current_gaze_ = [0.0 , 0.0]
def MicroJitter(t, speed=10.0, amount=0.004, offset=0.0):
    return math.sin(t * speed + offset) * amount

def UpdateGaze(t):
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

    jitter_x = MicroJitter(t, 11.3, 0.002)
    jitter_y = MicroJitter(t, 9.7, 0.002, 1.7)
    return [current_gaze_[0] + jitter_x, current_gaze_[1] + jitter_y]

#Make Navi blink every so often
next_blink_time_ = np.random.uniform(3, 7)
blink_start_ = -1.0
kBlinkDuration = 0.16
def UpdateBlink(t):
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
def UpdateHead(t):
    global target_tilt_, current_tilt_, next_head_move_
    if t > next_head_move_:
        target_tilt_ = np.random.uniform(-0.07, 0.07)
        next_head_move_ = t + np.random.uniform(3, 6)
    current_tilt_ += (target_tilt_ - current_tilt_) * 0.02
    return current_tilt_ + MicroJitter(t, 6.5, 0.002)

def GetPlaybackTimeS():
    """Estimate playback time in seconds, aligned to the utterance start."""
    with talk_lock_:
        if play_start_mono_ is None:
            return 0.0
        t = time.monotonic() - play_start_mono_ - kPlaybackLatency
    return max(0.0, t)

def SampleEnvAtPlaybackTime(play_t):
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
def NaviSpeak(text):
    global talk_active_, audio_level_, speech_impulse_
    global play_start_mono_, audio_written_s_, gen_done_, audio_end_s_

    with talk_lock_:
        gen_done_ = False
        audio_written_s_ = 0.0
        audio_end_s_ = 0.0
        play_start_mono_ = None
        env_ring_.clear()

    def run():
        global talk_active_, audio_level_, speech_impulse_
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
                chunk = piper.stdout.read(kChunkBytes)
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
                    env = env + (rms - env) * kEnvAttack
                else:
                    env = env + (rms - env) * kEnvRelease

                # Map to something usable (gain + clamp); tune this
                env_norm = min(env * 3.0, 1.0)

                # Timestamp on the audio timeline (not wall time)
                with talk_lock_:
                    audio_t0 = audio_written_s_
                    env_ring_.append((audio_t0, env_norm))
                    audio_written_s_ += (samples.size / kSampleRate)  # exact dt if last chunk short

                # --- Pace writes so we don't buffer too far ahead ---
                # This keeps wall clock and audio timeline aligned.
                while True:
                    with talk_lock_:
                        if play_start_mono_ is None:
                            break
                        elapsed = time.monotonic() - play_start_mono_
                        ahead = audio_written_s_ - elapsed
                    if ahead <= kWriteAheadLimit:
                        break
                    time.sleep(min(ahead - kWriteAheadLimit, 0.01))

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
            print("Navi finished speaking")
        
    threading.Thread(target=run, daemon=True).start()

@window_.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        NaviSpeak("Hello Hannah. I am Navi.")
    elif symbol == pyglet.window.key.ESCAPE:
        window_.close()

@window_.event
def on_draw():
    global talk_active_, current_mouth_, ctx_

    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline().strip()
        if cmd:
            NaviSpeak(cmd)

    ctx_.clear(0.0, 0.0, 0.0)
    t = time.time() - start_time_
    prog_["time_"].value = t

    play_t = GetPlaybackTimeS()
    env_play = SampleEnvAtPlaybackTime(play_t)
    with talk_lock_:
        still_playing = (not gen_done_) or (play_t < audio_end_s_)
        talk_active_ = still_playing
    if talk_active_:
        target_mouth = env_play
        prog_["gaze_"].value = (0.0, 0.0)
        prog_["head_tilt_"].value = UpdateHead(t) + target_mouth * 0.01
    else:
        prog_["gaze_"].value = UpdateGaze(t)
        prog_["head_tilt_"].value = UpdateHead(t)
        target_mouth = 0.0


    # smooth transition
    current_mouth_ += (target_mouth - current_mouth_) * 0.25
    prog_["mouth_open_"].value = current_mouth_

    # ---- Subtle breathing motion ----
    prog_["zoom_"].value = 1.35 + math.sin(t * 0.25) * 0.01

    # ---- Blinking ----
    prog_["blink_"].value = UpdateBlink(t)

    vao_.render(moderngl.TRIANGLE_STRIP)

pyglet.app.run()