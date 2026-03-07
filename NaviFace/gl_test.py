import pyglet
import moderngl
import numpy as np
import time
import math

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
        uniform float time;
        uniform vec2 resolution;
        out vec4 fragColor;

float sdEllipse(vec2 p, vec2 r) {
    // Signed-ish distance for ellipse (good enough for soft masks)
    float k = length(p / r);
    return k - 1.0;
}

void main() {
    vec2 res = resolution;
    vec2 uv = (gl_FragCoord.xy - 0.5 * res) / res.y * 1.5;

    float eye_offset = 0.33;
    float eye_y = 0.14;

    float eye_width = 0.23;
    float eye_height = 0.18;

    // ---- Blink ----
    float blink = 0.0;
    float blink_cycle = mod(time, 4.0);
    if (blink_cycle < 0.12) {
        float phase = blink_cycle / 0.12;
        blink = sin(phase * 3.14159);
    }

    float dynamic_height = max(eye_height * (1.0 - blink), 0.01);

    vec2 left_uv  = uv - vec2(-eye_offset, eye_y);
    vec2 right_uv = uv - vec2( eye_offset, eye_y);

    float left_eye  = length(vec2(left_uv.x / eye_width,  left_uv.y / dynamic_height));
    float right_eye = length(vec2(right_uv.x / eye_width, right_uv.y / dynamic_height));

    float eye_mask = max(
        smoothstep(1.0, 0.88, left_eye),
        smoothstep(1.0, 0.88, right_eye)
    );

    // ---- Brighter cyan core ----
    float core_glow = max(
        smoothstep(0.7, 0.3, left_eye),
        smoothstep(0.7, 0.3, right_eye)
    );

    vec3 base_cyan = vec3(0.4, 1.0, 1.0);

    // ---- Rainbow cheeky arc (top tint) ----
    float arc = clamp(1.0 - uv.y * 2.5, 0.0, 1.0);
    vec3 rainbow_tint = vec3(0.8, 0.9, 1.0);

    vec3 eye_color = mix(base_cyan, rainbow_tint, arc * 0.4);

    eye_color += core_glow * 0.4;

    vec3 final_color = eye_color * eye_mask;
    // -------------------------
    // Blush: outward cheeks + horizontal ovals
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

    fragColor = vec4(final_color, 1.0);
}
    """,
)

prog["time"] = 0.0
start_time = time.time()

prog["resolution"].value = (width, height)

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

@window.event
def on_draw():
    ctx.clear(0.0, 0.0, 0.0)
    prog["time"].value = time.time() - start_time
    vao.render(moderngl.TRIANGLE_STRIP)

pyglet.app.run()