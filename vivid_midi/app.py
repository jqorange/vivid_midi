import threading

import cv2

from .config import RenderConfig, RuntimeState
from .midi import apply_events, choose_midi_port, drain_events, get_held_notes, midi_worker
from .renderer import Renderer


def run():
    cfg = RenderConfig()
    state = RuntimeState()
    renderer = Renderer(cfg, state)
    renderer.enable_gpu_if_available()

    port = choose_midi_port()
    th = threading.Thread(target=midi_worker, args=(port,), daemon=True)
    th.start()
    print(f"\nListening MIDI from: {port}")

    cap = cv2.VideoCapture(cfg.cam_index)
    ok, _ = cap.read()
    if not ok:
        print("Camera read failed.")
        return

    win = "Waterfall"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda event, x, y, flags, param: renderer.handle_mouse(event, x, y))

    renderer.start_calibration()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        renderer.scroll_and_fade()

        local_events = drain_events()
        apply_events(local_events, cfg.note_min, cfg.note_max)
        held = get_held_notes()

        if cfg.show_bars:
            for note, vel in held:
                renderer.draw_note_stamp_at_bottom(note, vel)

        if cfg.fx_enabled and cfg.particle_fx:
            renderer.scroll_and_fade_particles()
            if cfg.bottom_noise:
                renderer.add_bottom_noise()
            renderer.add_ambient_fog()
            renderer.add_note_sparks(local_events)
            if cfg.particle_blur_every > 0 and (frame_count % cfg.particle_blur_every == 0):
                k = cfg.particle_blur_k
                if k >= 3 and k % 2 == 1:
                    renderer.pplane[:] = cv2.blur(renderer.pplane, (k, k))

        quad = renderer.get_fly_quad_extended()
        overlay_bars, overlay_particles = renderer.warp_to_frame(frame.shape, quad)

        out = frame
        if cfg.show_bars:
            out = renderer.blend_additive(out, overlay_bars, cfg.base_alpha)
        if cfg.fx_enabled and cfg.particle_fx:
            out = renderer.blend_additive(out, overlay_particles, cfg.particle_alpha)
        if cfg.fx_enabled and cfg.line_glow and state.fly_quad_base is not None:
            out = renderer.draw_foldline_glow(out)
        if cfg.edit_mode and state.fly_quad_base is not None:
            renderer.draw_edit_overlay(out)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            if state.fly_quad_base is None:
                print("[EDIT] Need calibration first.")
            else:
                cfg.edit_mode = True
                state.hover_idx = -1
                state.selected_idx = -1
                print("[EDIT] ON")
        elif key in (ord('f'), ord('F')):
            if cfg.edit_mode:
                cfg.edit_mode = False
                state.hover_idx = -1
                state.selected_idx = -1
                print("[EDIT] OFF")
        elif key == ord('c'):
            renderer.start_calibration()
        elif key == ord('r'):
            renderer.reset_calibration()
            renderer.start_calibration()
        elif key == ord('['):
            cfg.extend_scale = max(cfg.extend_min, cfg.extend_scale - cfg.extend_step)
            print(f"[TUNE] EXTEND_SCALE={cfg.extend_scale:.2f}")
        elif key == ord(']'):
            cfg.extend_scale = min(cfg.extend_max, cfg.extend_scale + cfg.extend_step)
            print(f"[TUNE] EXTEND_SCALE={cfg.extend_scale:.2f}")
        elif key == ord('w'):
            cfg.speed_px = min(260, cfg.speed_px + 2)
            print(f"[TUNE] SPEED_PX={cfg.speed_px}")
        elif key == ord('s'):
            cfg.speed_px = max(2, cfg.speed_px - 2)
            print(f"[TUNE] SPEED_PX={cfg.speed_px}")
        elif key == ord('e'):
            cfg.fade = min(0.9999, cfg.fade + 0.0005)
            print(f"[TUNE] FADE={cfg.fade:.4f}")
        elif key == ord('g'):
            cfg.fade = max(0.90, cfg.fade - 0.0005)
            print(f"[TUNE] FADE={cfg.fade:.4f}")
        elif key == ord('z'):
            cfg.base_alpha = max(0.3, cfg.base_alpha - 0.1)
            print(f"[TUNE] BASE_ALPHA={cfg.base_alpha:.2f}")
        elif key == ord('x'):
            cfg.base_alpha = min(4.0, cfg.base_alpha + 0.1)
            print(f"[TUNE] BASE_ALPHA={cfg.base_alpha:.2f}")
        elif key == ord('p'):
            cfg.fx_enabled = not cfg.fx_enabled
            print(f"[FX] FX_ENABLED={cfg.fx_enabled}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
