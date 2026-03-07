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

is_talking_ = False
audio_level_ = 0.0
current_mouth_ = 0.0
last_talk_time_ = 0
talk_hold_time_ = 0.8
prev_rms_ = 0.0
mouth_phase_ = 0.0

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
float depth = mouthOpen * 0.13;

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

// choose idle vs open
// soft glow gradient like the eyes
float mouthMask = mix(closedSmile, insideEllipse, smoothstep(0.01, 0.05, mouthOpen));

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

# prog["time"].value = 0.0
start_time = time.time()

prog["resolution"].value = (width, height)
prog["zoom"].value = 1.35
prog["gaze"].value = (0.0, 0.0)
prog["headTilt"].value = 0.0
prog["blink"].value = 0.0
prog["mouthOpen"].value = 0.0
start = time.time()
prog["time"].value = time.time() - start

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
next_gaze_time = 0
target_gaze = [0,0]
current_gaze = [0,0]
def update_gaze(t):
    global next_gaze_time, target_gaze, current_gaze

    if t > next_gaze_time:
        target_gaze = [
            np.random.uniform(-0.35,0.35),
            np.random.uniform(-0.15,0.15)
        ]
        next_gaze_time = t + np.random.uniform(1.5,4)
    # smooth eye motion
    speed = 0.08
    current_gaze[0] += (target_gaze[0]-current_gaze[0]) * speed
    current_gaze[1] += (target_gaze[1]-current_gaze[1]) * speed

    # Pixar micro jitter
    jitter_x = micro_jitter(t, 11.3, 0.002)
    jitter_y = micro_jitter(t, 9.7, 0.002, 1.7)

    return [
        current_gaze[0] + jitter_x,
        current_gaze[1] + jitter_y
    ]

#Make Navi blink every so often
next_blink_time = np.random.uniform(3,7)
blink_start = -1
blink_duration = 0.16
def update_blink(t):
    global next_blink_time, blink_start

    # start blink
    if blink_start < 0 and t > next_blink_time:
        blink_start = t
    # if blinking
    if blink_start >= 0:
        progress = (t - blink_start) / blink_duration
        if progress >= 1.0:
            blink_start = -1
            next_blink_time = t + np.random.uniform(3,7)
            return 0.0
        return math.sin(progress * math.pi) ** 0.6
    return 0.0

#Head bobbing swaying slightly
target_tilt = 0
current_tilt = 0
next_head_move = 0
def update_head(t):
    global target_tilt, current_tilt, next_head_move

    if t > next_head_move:
        target_tilt = np.random.uniform(-0.07,0.07)
        next_head_move = t + np.random.uniform(3,6)
    current_tilt += (target_tilt - current_tilt) * 0.02
    # Pixar head micro-corrections
    jitter = micro_jitter(t, 6.5, 0.002)
    return current_tilt + jitter

# Smooth Pixar-style micro jitter
def micro_jitter(t, speed=10.0, amount=0.004, offset=0.0):
    return math.sin(t * speed + offset) * amount

#Navi is talking
def navi_speak(text):
    global is_talking_, audio_level_, last_talk_time_, prev_rms_

    def run():
        global is_talking_, audio_level_, last_talk_time_, prev_rms_

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
            ["aplay", "-D", "plughw:0,0", "-r", "22050", "-f", "S16_LE", "-t", "raw"],
            stdin=subprocess.PIPE
        )

        piper.stdin.write(text.encode())
        piper.stdin.close()

        started = False
        while True:
            chunk = piper.stdout.read(1024)
            if not chunk:
                break
            if not started:
                is_talking_ = True
                started = True
            aplay.stdin.write(chunk)
            # compute amplitude
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

            if samples.size > 0:

                rms = np.sqrt(np.mean(samples * samples)) / 32768.0

                # boost articulation
                level = min(rms * 3.0, 1.0)

                # fast attack, slow release envelope
                if level > audio_level_:
                    audio_level_ = audio_level_ * 0.4 + level * 0.6
                else:
                    audio_level_ = audio_level_ * 0.85 + level * 0.15

        aplay.stdin.close()
        piper.wait()
        aplay.wait()

        audio_level_ = 0.0
        is_talking_ = False
        last_talk_time_ = time.time()

    threading.Thread(target=run).start()
    
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        navi_speak("Hello Hannah. I am Navi.")

@window.event
def on_draw():
    global is_talking_, current_mouth_, mouth_phase_

    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline().strip()
        if cmd:
            navi_speak(cmd)

    ctx.clear(0.0, 0.0, 0.0)

    t = time.time() - start_time
    prog["time"].value = t

    if is_talking_:
        # ---- Animate Mouth ----
        # advance mouth rhythm
        mouth_phase_ += 0.35
        # talking rhythm (open/close)
        talk_cycle = (math.sin(mouth_phase_) + 1.0) * 0.5
        # amplitude controls how wide the mouth opens
        amp = min(audio_level_ * 9.0, 1.0)
        target_mouth = talk_cycle * amp
        if target_mouth < 0.02:
            target_mouth = 0.0
        prog["gaze"].value = (0.0, 0.0)
        # ---- Head nod slightly ----
        prog["headTilt"].value = update_head(t) + math.sin(t * 5) * 0.01
    elif time.time() - last_talk_time_ < talk_hold_time_:
        # keep looking forward briefly after talking
        prog["gaze"].value = (0.0, 0.0)
        target_mouth = 0.0
    else:
        prog["mouthOpen"].value = 0.0
        # ---- idle eye movement ----
        prog["gaze"].value = update_gaze(t)
        # ---- gentle head tilt idle motion ----
        prog["headTilt"].value = update_head(t)
        target_mouth = 0.0

    # smooth transition
    current_mouth_ += (target_mouth - current_mouth_) * 0.45
    prog["mouthOpen"].value = current_mouth_

    # ---- Subtle breathing motion ----
    zoom = 1.35 + math.sin(t * 0.25) * 0.01
    prog["zoom"].value = zoom

    # ---- Blinking ----
    prog["blink"].value = update_blink(t)

    vao.render(moderngl.TRIANGLE_STRIP)

pyglet.app.run()