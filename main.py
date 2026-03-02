import threading
from collections import deque

import cv2
import numpy as np
import mido

# =========================
# Config
# =========================
CAM_INDEX = 0

PLANE_W, PLANE_H = 1280, 2600

NOTE_MIN, NOTE_MAX = 21, 108
NOTE_RANGE = NOTE_MAX - NOTE_MIN + 1

SPEED_PX = 42
FADE = 0.9976

BASE_ALPHA = 2.3

STAMP_H = None

# Glow: vertical-only (no horizontal bleed => no overlap across notes)
GLOW = True
GLOW_EXPAND_Y = 26
GLOW_EXPAND_X = 0

# Fly far
EXTEND_SCALE = 3.0
EXTEND_STEP = 0.25
EXTEND_MIN = 1.0
EXTEND_MAX = 12.0

# =========================
# Edit UX (two-click relocate)  --- SMALLER VISUALS
# =========================
EDIT_MODE = False

HANDLE_RADIUS = 5
HANDLE_HIT_RADIUS = 18
HANDLE_RING_RADIUS = 10
LINE_THICKNESS = 1

_hover_idx = -1
_selected_idx = -1

# =========================
# FX: Line glow + Particles (REALTIME FRIENDLY)
# =========================
FX_ENABLED = True

# If your OpenCV supports UMat (OpenCL/T-API), you can try True.
# On many mac setups this may or may not speed up; code works either way.
USE_UMAT = False

# ---- Fold line glow (on OUTPUT frame) ----
LINE_GLOW = True
LINE_GLOW_PASSES = 4       # more => brighter but heavier
LINE_GLOW_BASE_THICK = 2
LINE_GLOW_EXTRA_THICK = 14
LINE_GLOW_ALPHA = 0.95     # strength

# ---- Particle layer (low-res, warped to frame) ----
PARTICLE_FX = True
PART_SCALE = 2             # particle plane resolution = (PLANE_W/PART_SCALE, PLANE_H/PART_SCALE)
PART_FADE = 0.9925
PART_SPEED_DIV = 2         # particle scroll speed = SPEED_PX / PART_SPEED_DIV

PARTICLE_ALPHA = 1.2       # how strong particle layer adds onto output
PARTICLE_BLUR_EVERY = 2    # blur every N frames (0/1 => every frame)
PARTICLE_BLUR_K = 3        # blur kernel size (odd)

# Ambient fog along fold line
AMBIENT_FOG_RATE = 220     # particles per second-ish (scaled per frame)
AMBIENT_FOG_BRIGHT = 140

# Note sparkle
NOTE_SPARK_RATE = 6        # sparks per frame per held note (scaled by velocity)
NOTE_SPARK_BRIGHT = 220

# Spark spread
SPARK_X_JITTER = 2         # in particle-plane px
SPARK_Y_JITTER = 6

# Optional: subtle twinkle noise in particle plane bottom strip (cheap)
BOTTOM_NOISE = True
BOTTOM_NOISE_H = 14

# =========================
# MIDI thread-safe queue
# =========================
events = deque()
events_lock = threading.Lock()

active_notes = {}  # note -> velocity
active_lock = threading.Lock()

def list_midi_inputs():
    return mido.get_input_names()

def midi_worker(port_name: str):
    with mido.open_input(port_name) as inport:
        for msg in inport:
            if msg.type not in ("note_on", "note_off"):
                continue
            if msg.type == "note_on" and msg.velocity == 0:
                t, vel = "note_off", 0
            else:
                t, vel = msg.type, getattr(msg, "velocity", 0)
            with events_lock:
                events.append((t, msg.note, vel))

def choose_midi_port():
    ports = list_midi_inputs()
    if not ports:
        raise RuntimeError("No MIDI input ports found.")
    print("\nAvailable MIDI input ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    while True:
        s = input("\nSelect MIDI input by number: ").strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(ports):
                return ports[idx]
        print("Invalid selection.")

# =========================
# Fly-out plane calibration
# =========================
calib_mode = False
calib_points = []
fly_quad_base = None

CALIB_LABELS = ["L0 (emit-left)", "R0 (emit-right)", "L1 (far-left)", "R1 (far-right)"]
PT_NAMES_QUAD = ["L1", "R1", "R0", "L0"]  # indices in fly_quad_base

def start_calibration():
    global calib_mode, calib_points, fly_quad_base
    global _selected_idx, _hover_idx
    calib_mode = True
    calib_points = []
    fly_quad_base = None
    _selected_idx = -1
    _hover_idx = -1
    print("\n[CALIB] Click 4 points in order:")
    for i, lab in enumerate(CALIB_LABELS, 1):
        print(f"  {i}) {lab}")

def reset_calibration():
    global calib_mode, calib_points, fly_quad_base
    global EDIT_MODE, _selected_idx, _hover_idx
    calib_mode = False
    calib_points = []
    fly_quad_base = None
    EDIT_MODE = False
    _selected_idx = -1
    _hover_idx = -1
    print("[CALIB] Reset.")

def get_fly_quad_extended():
    global fly_quad_base, EXTEND_SCALE
    if fly_quad_base is None:
        return None
    L1, R1, R0, L0 = fly_quad_base.astype(np.float32)
    vL = (L1 - L0)
    vR = (R1 - R0)
    L1e = L0 + vL * float(EXTEND_SCALE)
    R1e = R0 + vR * float(EXTEND_SCALE)
    return np.float32([L1e, R1e, R0, L0])

# =========================
# Note -> x mapping (no overlap)
# =========================
KEY_W = PLANE_W / NOTE_RANGE

def note_to_cell(note: int):
    note = int(max(NOTE_MIN, min(NOTE_MAX, note)))
    i = note - NOTE_MIN
    x0 = int(np.floor(i * KEY_W))
    x1 = int(np.floor((i + 1) * KEY_W) - 1)
    x0 = max(0, min(PLANE_W - 1, x0))
    x1 = max(0, min(PLANE_W - 1, x1))
    if x1 < x0:
        x1 = x0
    return x0, x1

# =========================
# Waterfall rendering
# =========================
def scroll_and_fade(plane: np.ndarray):
    plane[:] = np.roll(plane, -SPEED_PX, axis=0)
    plane[-SPEED_PX:, :, :] = 0
    plane[:] = (plane.astype(np.float32) * FADE).astype(np.uint8)

def vel_gain(vel: int) -> float:
    v = np.clip(vel / 127.0, 0.0, 1.0)
    return float(v ** 0.30)

def stamp_color_bgr(vel: int):
    g = vel_gain(vel)
    base = np.array([210, 80, 255], dtype=np.float32)   # neon magenta-ish
    bright = (0.55 + 1.00 * g)                          # slightly brighter than before
    col = np.clip(base * bright, 0, 255).astype(np.uint8)
    return int(col[0]), int(col[1]), int(col[2])

def glow_color_bgr(vel: int):
    g = vel_gain(vel)
    base = np.array([255, 180, 255], dtype=np.float32)
    bright = (0.18 + 0.70 * g)
    col = np.clip(base * bright, 0, 255).astype(np.uint8)
    return int(col[0]), int(col[1]), int(col[2])

def draw_note_stamp_at_bottom(plane: np.ndarray, note: int, vel: int):
    x0, x1 = note_to_cell(note)
    if x1 - x0 >= 4:
        x0 += 1
        x1 -= 1

    stamp_h = SPEED_PX if STAMP_H is None else int(STAMP_H)
    y1 = PLANE_H - 1
    y0 = max(0, PLANE_H - stamp_h)

    c = stamp_color_bgr(vel)
    cv2.rectangle(plane, (x0, y0), (x1, y1), c, thickness=-1)

    if not GLOW:
        return

    ge_y = int(GLOW_EXPAND_Y * (0.7 + vel / 127.0))
    ge_x = int(GLOW_EXPAND_X)

    gx0 = max(0, x0 - ge_x)
    gx1 = min(PLANE_W - 1, x1 + ge_x)
    gx0 = max(gx0, x0)  # hard clamp to cell
    gx1 = min(gx1, x1)

    gy0 = max(0, y0 - ge_y)
    gy1 = min(PLANE_H - 1, y1 + ge_y)

    gc = glow_color_bgr(vel)
    cv2.rectangle(plane, (gx0, gy0), (gx1, gy1), gc, thickness=-1)

# =========================
# Particle plane (low-res) FX
# =========================
def make_particle_plane():
    pw = PLANE_W // PART_SCALE
    ph = PLANE_H // PART_SCALE
    return np.zeros((ph, pw, 3), dtype=np.uint8)

def scroll_and_fade_particles(pplane: np.ndarray, spx: int):
    pplane[:] = np.roll(pplane, -spx, axis=0)
    pplane[-spx:, :, :] = 0
    pplane[:] = (pplane.astype(np.float32) * PART_FADE).astype(np.uint8)

def particle_note_to_cell(note: int, pw: int):
    x0, x1 = note_to_cell(note)
    x0p = int(x0 / PART_SCALE)
    x1p = int(x1 / PART_SCALE)
    x0p = max(0, min(pw - 1, x0p))
    x1p = max(0, min(pw - 1, x1p))
    if x1p < x0p:
        x1p = x0p
    return x0p, x1p

def add_ambient_fog(pplane: np.ndarray):
    ph, pw = pplane.shape[:2]
    y_base = ph - 2
    n = max(0, int(AMBIENT_FOG_RATE / 60.0))
    if n <= 0:
        return
    xs = np.random.randint(0, pw, size=n)
    ys = np.random.randint(max(0, y_base - 6), y_base + 1, size=n)
    br = AMBIENT_FOG_BRIGHT

    pplane[ys, xs, 0] = np.clip(pplane[ys, xs, 0].astype(np.int16) + br, 0, 255).astype(np.uint8)
    pplane[ys, xs, 1] = np.clip(pplane[ys, xs, 1].astype(np.int16) + int(br * 0.9), 0, 255).astype(np.uint8)
    pplane[ys, xs, 2] = np.clip(pplane[ys, xs, 2].astype(np.int16) + int(br * 0.95), 0, 255).astype(np.uint8)

def add_note_sparks(pplane: np.ndarray, held_notes):
    ph, pw = pplane.shape[:2]
    y_base = ph - 2
    for note, vel in held_notes:
        x0p, x1p = particle_note_to_cell(note, pw)
        if x1p - x0p <= 0:
            xp = x0p
        else:
            xp = np.random.randint(x0p, x1p + 1)

        g = vel_gain(vel)
        k = int(NOTE_SPARK_RATE * (0.5 + 1.7 * g))
        if k <= 0:
            continue

        xs = np.random.randint(max(0, xp - SPARK_X_JITTER), min(pw, xp + SPARK_X_JITTER + 1), size=k)
        ys = np.random.randint(max(0, y_base - SPARK_Y_JITTER), y_base + 1, size=k)

        br = int(NOTE_SPARK_BRIGHT * (0.55 + 0.75 * g))
        add_b = br
        add_g = int(br * 0.75)
        add_r = int(br * 1.00)

        pplane[ys, xs, 0] = np.clip(pplane[ys, xs, 0].astype(np.int16) + add_b, 0, 255).astype(np.uint8)
        pplane[ys, xs, 1] = np.clip(pplane[ys, xs, 1].astype(np.int16) + add_g, 0, 255).astype(np.uint8)
        pplane[ys, xs, 2] = np.clip(pplane[ys, xs, 2].astype(np.int16) + add_r, 0, 255).astype(np.uint8)

def add_bottom_noise(pplane: np.ndarray):
    ph, pw = pplane.shape[:2]
    h = min(BOTTOM_NOISE_H, ph)
    if h <= 0:
        return
    noise = (np.random.randn(h, pw, 1) * 18.0 + 18.0).astype(np.int16)
    noise = np.clip(noise, 0, 40).astype(np.int16)
    strip = pplane[ph - h:ph, :, :].astype(np.int16)

    strip[:, :, 0] = np.clip(strip[:, :, 0] + noise[:, :, 0], 0, 255)
    strip[:, :, 1] = np.clip(strip[:, :, 1] + (noise[:, :, 0] * 0.7).astype(np.int16), 0, 255)
    strip[:, :, 2] = np.clip(strip[:, :, 2] + (noise[:, :, 0] * 0.8).astype(np.int16), 0, 255)
    pplane[ph - h:ph, :, :] = strip.astype(np.uint8)

# =========================
# Warp + blend
# =========================
def warp_plane_to_frame(plane: np.ndarray, frame_shape, dst_quad: np.ndarray):
    h, w = frame_shape[:2]
    src = np.float32([[0, 0],
                      [plane.shape[1] - 1, 0],
                      [plane.shape[1] - 1, plane.shape[0] - 1],
                      [0, plane.shape[0] - 1]])
    H = cv2.getPerspectiveTransform(src, dst_quad)
    return cv2.warpPerspective(plane, H, (w, h))

def blend_additive(frame: np.ndarray, overlay: np.ndarray, alpha: float):
    out = frame.astype(np.float32) + alpha * overlay.astype(np.float32)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

# =========================
# Line glow on output frame
# =========================
def draw_foldline_glow(out: np.ndarray, quad: np.ndarray):
    if quad is None:
        return out

    L1, R1, R0, L0 = quad.astype(np.int32)
    p0 = (int(L0[0]), int(L0[1]))
    p1 = (int(R0[0]), int(R0[1]))

    overlay = np.zeros_like(out)

    for i in range(LINE_GLOW_PASSES):
        t = LINE_GLOW_BASE_THICK + int((i / max(1, LINE_GLOW_PASSES - 1)) * LINE_GLOW_EXTRA_THICK)
        cv2.line(overlay, p0, p1, (255, 240, 255), thickness=t, lineType=cv2.LINE_AA)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=6.0, sigmaY=6.0)
    out2 = blend_additive(out, overlay, LINE_GLOW_ALPHA)
    return out2

# =========================
# Edit overlay & picking (two-click relocate)
# =========================
def _dist2(a, b):
    d = a - b
    return float(d[0]*d[0] + d[1]*d[1])

def pick_handle(pt: np.ndarray, quad: np.ndarray):
    best_i = -1
    best_d = 1e18
    for i in range(4):
        d = _dist2(quad[i], pt)
        if d < best_d:
            best_d = d
            best_i = i
    if best_d <= (HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS):
        return best_i
    return -1

def draw_edit_overlay(img: np.ndarray, quad: np.ndarray, hover_idx: int, selected_idx: int):
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255), LINE_THICKNESS)

    for i, p in enumerate(quad.astype(np.int32)):
        x, y = int(p[0]), int(p[1])

        cv2.circle(img, (x, y), HANDLE_RADIUS, (0, 255, 255), -1)

        if i == hover_idx:
            cv2.circle(img, (x, y), HANDLE_RING_RADIUS, (255, 255, 0), 1)

        if i == selected_idx:
            cv2.circle(img, (x, y), HANDLE_RING_RADIUS + 4, (0, 255, 0), 2)

        cv2.putText(img, PT_NAMES_QUAD[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    if selected_idx != -1:
        cv2.putText(img, f"Selected: {PT_NAMES_QUAD[selected_idx]} (click anywhere to move)",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def on_mouse(event, x, y, flags, param):
    global calib_mode, calib_points, fly_quad_base
    global EDIT_MODE, _hover_idx, _selected_idx

    pt = np.array([x, y], dtype=np.float32)

    if EDIT_MODE and fly_quad_base is not None:
        if event == cv2.EVENT_MOUSEMOVE:
            _hover_idx = pick_handle(pt, fly_quad_base)

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = pick_handle(pt, fly_quad_base)

            if _selected_idx == -1:
                if idx != -1:
                    _selected_idx = idx
                return

            if idx != -1:
                _selected_idx = idx
                return

            fly_quad_base[_selected_idx] = pt
            return

        return

    if calib_mode and event == cv2.EVENT_LBUTTONDOWN:
        idx = len(calib_points)
        if idx < 4:
            print(f"[CALIB] {CALIB_LABELS[idx]} = ({x}, {y})")
            calib_points.append((x, y))
            if len(calib_points) == 4:
                L0 = np.float32(calib_points[0])
                R0 = np.float32(calib_points[1])
                L1 = np.float32(calib_points[2])
                R1 = np.float32(calib_points[3])
                fly_quad_base = np.float32([L1, R1, R0, L0])
                calib_mode = False
                print("[CALIB] Completed fly-out plane calibration.")
        return

# =========================
# Main
# =========================
def main():
    global fly_quad_base, EDIT_MODE, _hover_idx, _selected_idx
    global SPEED_PX, FADE, BASE_ALPHA, EXTEND_SCALE
    global FX_ENABLED  # <-- FIX: declare before using

    port = choose_midi_port()
    th = threading.Thread(target=midi_worker, args=(port,), daemon=True)
    th.start()
    print(f"\nListening MIDI from: {port}")

    cap = cv2.VideoCapture(CAM_INDEX)
    ok, _ = cap.read()
    if not ok:
        print("Camera read failed.")
        return

    win = "Waterfall"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    plane = np.zeros((PLANE_H, PLANE_W, 3), dtype=np.uint8)

    pplane = make_particle_plane()
    p_spx = max(1, int(SPEED_PX / max(1, PART_SPEED_DIV)))

    frame_count = 0

    start_calibration()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        scroll_and_fade(plane)

        with events_lock:
            local = list(events)
            events.clear()

        if local:
            with active_lock:
                for t, note, vel in local:
                    if note < NOTE_MIN or note > NOTE_MAX:
                        continue
                    if t == "note_on":
                        active_notes[note] = int(vel)
                    elif t == "note_off":
                        active_notes.pop(note, None)

        with active_lock:
            held = list(active_notes.items())

        for note, vel in held:
            draw_note_stamp_at_bottom(plane, note, vel)

        if FX_ENABLED and PARTICLE_FX:
            p_spx = max(1, int(SPEED_PX / max(1, PART_SPEED_DIV)))
            scroll_and_fade_particles(pplane, p_spx)

            if BOTTOM_NOISE:
                add_bottom_noise(pplane)

            add_ambient_fog(pplane)
            add_note_sparks(pplane, held)

            if PARTICLE_BLUR_EVERY > 0 and (frame_count % PARTICLE_BLUR_EVERY == 0):
                k = PARTICLE_BLUR_K
                if k >= 3 and k % 2 == 1:
                    pplane[:] = cv2.blur(pplane, (k, k))

        overlay_bars = np.zeros_like(frame)
        overlay_particles = np.zeros_like(frame)

        fly_quad_ext = get_fly_quad_extended()
        if fly_quad_ext is not None:
            overlay_bars = warp_plane_to_frame(plane, frame.shape, fly_quad_ext)
            if FX_ENABLED and PARTICLE_FX:
                overlay_particles = warp_plane_to_frame(pplane, frame.shape, fly_quad_ext)

        out = blend_additive(frame, overlay_bars, BASE_ALPHA)

        if FX_ENABLED and PARTICLE_FX:
            out = blend_additive(out, overlay_particles, PARTICLE_ALPHA)

        if FX_ENABLED and LINE_GLOW and fly_quad_base is not None:
            out = draw_foldline_glow(out, fly_quad_base)

        if EDIT_MODE and fly_quad_base is not None:
            draw_edit_overlay(out, fly_quad_base, _hover_idx, _selected_idx)

        cv2.imshow(win, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('a'):
            if fly_quad_base is None:
                print("[EDIT] Need calibration first.")
            else:
                EDIT_MODE = True
                _hover_idx = -1
                _selected_idx = -1
                print("[EDIT] ON: click a point to select, then click anywhere to move it. Press 'f' to finish.")

        elif key == ord('f') or key == ord('F'):
            if EDIT_MODE:
                EDIT_MODE = False
                _hover_idx = -1
                _selected_idx = -1
                print("[EDIT] OFF.")

        elif key == ord('c'):
            start_calibration()

        elif key == ord('r'):
            reset_calibration()
            start_calibration()

        elif key == ord('['):
            EXTEND_SCALE = max(EXTEND_MIN, EXTEND_SCALE - EXTEND_STEP)
            print(f"[TUNE] EXTEND_SCALE={EXTEND_SCALE:.2f}")
        elif key == ord(']'):
            EXTEND_SCALE = min(EXTEND_MAX, EXTEND_SCALE + EXTEND_STEP)
            print(f"[TUNE] EXTEND_SCALE={EXTEND_SCALE:.2f}")

        elif key == ord('w'):
            SPEED_PX = min(260, SPEED_PX + 2)
            print(f"[TUNE] SPEED_PX={SPEED_PX}")
        elif key == ord('s'):
            SPEED_PX = max(2, SPEED_PX - 2)
            print(f"[TUNE] SPEED_PX={SPEED_PX}")

        elif key == ord('e'):
            FADE = min(0.9999, FADE + 0.0005)
            print(f"[TUNE] FADE={FADE:.4f}")
        elif key == ord('g'):
            FADE = max(0.90, FADE - 0.0005)
            print(f"[TUNE] FADE={FADE:.4f}")

        elif key == ord('z'):
            BASE_ALPHA = max(0.3, BASE_ALPHA - 0.1)
            print(f"[TUNE] BASE_ALPHA={BASE_ALPHA:.2f}")
        elif key == ord('x'):
            BASE_ALPHA = min(4.0, BASE_ALPHA + 0.1)
            print(f"[TUNE] BASE_ALPHA={BASE_ALPHA:.2f}")

        elif key == ord('p'):
            FX_ENABLED = not FX_ENABLED
            print(f"[FX] FX_ENABLED={FX_ENABLED}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
