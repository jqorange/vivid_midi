import threading
import time

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

    cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.cam_buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
    if cfg.cam_use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    ok, _ = cap.read()
    if not ok:
        print("Camera read failed.")
        return

    win = "Waterfall"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda event, x, y, flags, param: renderer.handle_mouse(event, x, y))

    if cfg.hdmi_forward:
        cv2.namedWindow(cfg.hdmi_window_name, cv2.WINDOW_NORMAL)
        if cfg.hdmi_fullscreen:
            cv2.setWindowProperty(cfg.hdmi_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(cfg.hdmi_window_name, cfg.hdmi_width, cfg.hdmi_height)

    renderer.start_calibration()
    print("[HDMI] Press 'h' to toggle HDMI forwarding.")
    frame_count = 0

    fps_t0 = time.perf_counter()
    fps_frames = 0
    low_fps_hits = 0
    high_fps_hits = 0

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
            out = renderer.blend_additive_inplace(out, overlay_bars, cfg.base_alpha, renderer.overlay_bars_scaled)
        if cfg.fx_enabled and cfg.particle_fx:
            out = renderer.blend_additive_inplace(out, overlay_particles, cfg.particle_alpha, renderer.overlay_particles_scaled)
        if cfg.fx_enabled and cfg.line_glow and state.fly_quad_base is not None:
            out = renderer.draw_foldline_glow(out)
        if cfg.edit_mode and state.fly_quad_base is not None:
            renderer.draw_edit_overlay(out)

        cv2.imshow(win, out)
        if cfg.hdmi_forward:
            hdmi_out = out
            if out.shape[1] != cfg.hdmi_width or out.shape[0] != cfg.hdmi_height:
                hdmi_out = cv2.resize(out, (cfg.hdmi_width, cfg.hdmi_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(cfg.hdmi_window_name, hdmi_out)
        fps_frames += 1
        now = time.perf_counter()
        elapsed = now - fps_t0
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_t0 = now
            fps_frames = 0

            if fps < 58.0:
                low_fps_hits += 1
                high_fps_hits = 0
            elif fps > 66.0:
                high_fps_hits += 1
                low_fps_hits = 0
            else:
                low_fps_hits = 0
                high_fps_hits = 0

            if low_fps_hits >= 2:
                # Auto trade tiny quality for smoother frame rate.
                cfg.deband_blur_every = min(4, cfg.deband_blur_every + 1)
                cfg.particle_blur_every = min(6, cfg.particle_blur_every + 1)
                cfg.line_glow_passes = max(1, cfg.line_glow_passes - 1)
                cfg.firework_emit_count = max(8, int(cfg.firework_emit_count * 0.9))
                low_fps_hits = 0
                print(f"[PERF] fps={fps:.1f} -> deband_every={cfg.deband_blur_every}, part_blur_every={cfg.particle_blur_every}, glow_passes={cfg.line_glow_passes}, emit={cfg.firework_emit_count}")
            elif high_fps_hits >= 3:
                # Recover quality when sustained headroom exists.
                cfg.deband_blur_every = max(1, cfg.deband_blur_every - 1)
                cfg.particle_blur_every = max(2, cfg.particle_blur_every - 1)
                cfg.line_glow_passes = min(3, cfg.line_glow_passes + 1)
                cfg.firework_emit_count = min(20, int(cfg.firework_emit_count * 1.08) + 1)
                high_fps_hits = 0

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
        elif key in (ord('h'), ord('H')):
            cfg.hdmi_forward = not cfg.hdmi_forward
            if not cfg.hdmi_forward:
                close_hdmi_window()
            print(f"[HDMI] FORWARD={cfg.hdmi_forward}")

        frame_count += 1

    cap.release()
    close_hdmi_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
