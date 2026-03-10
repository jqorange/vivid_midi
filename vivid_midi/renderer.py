import cv2
import numpy as np

from .config import RenderConfig, RuntimeState

CALIB_LABELS = ["L0 (emit-left)", "R0 (emit-right)", "L1 (far-left)", "R1 (far-right)"]
PT_NAMES_QUAD = ["L1", "R1", "R0", "L0"]


class Renderer:
    def __init__(self, cfg: RenderConfig, state: RuntimeState):
        self.cfg = cfg
        self.state = state

        self.key_w = cfg.plane_w / cfg.note_range
        self.plane = np.zeros((cfg.plane_h, cfg.plane_w, 3), dtype=np.uint8)

        self.pw = cfg.plane_w // cfg.part_scale
        self.ph = cfg.plane_h // cfg.part_scale
        self.pplane = np.zeros((self.ph, self.pw, 3), dtype=np.uint8)

        self._src_plane = np.float32([[0, 0], [cfg.plane_w - 1, 0], [cfg.plane_w - 1, cfg.plane_h - 1], [0, cfg.plane_h - 1]])
        self._src_pplane = np.float32([[0, 0], [self.pw - 1, 0], [self.pw - 1, self.ph - 1], [0, self.ph - 1]])

        self._last_quad_sig = None
        self._H_bars = None
        self._H_parts = None

        self._tmp_fade = np.empty_like(self.plane)
        self._tmp_pfade = np.empty_like(self.pplane)

        self._frame_idx = 0
        self._line_glow_overlay = None
        self._line_glow_sig = None

        self.overlay_bars = None
        self.overlay_particles = None
        self.overlay_bars_scaled = None
        self.overlay_particles_scaled = None
        self.firework_particles: list[dict] = []

        self._note_cells = [self._compute_note_cell(n) for n in range(self.cfg.note_min, self.cfg.note_max + 1)]
        self._stamp_color_lut = [self._calc_stamp_color_bgr(v) for v in range(128)]
        self._glow_color_lut = [self._calc_glow_color_bgr(v) for v in range(128)]
        self._firework_color_lut = [self._calc_firework_color(v) for v in range(128)]

    def enable_gpu_if_available(self):
        cv2.ocl.setUseOpenCL(True)
        has_ocl = cv2.ocl.haveOpenCL()
        active_ocl = cv2.ocl.useOpenCL()
        print(f"[GPU] OpenCL available={has_ocl}, enabled={active_ocl}, use_umat={self.cfg.use_umat}")

    def ensure_overlays(self, frame_shape):
        h, w = frame_shape[:2]
        if self.overlay_bars is None or self.overlay_bars.shape[:2] != (h, w):
            self.overlay_bars = np.zeros((h, w, 3), dtype=np.uint8)
            self.overlay_particles = np.zeros((h, w, 3), dtype=np.uint8)
            self.overlay_bars_scaled = np.zeros((h, w, 3), dtype=np.uint8)
            self.overlay_particles_scaled = np.zeros((h, w, 3), dtype=np.uint8)

    def _compute_note_cell(self, note: int):
        note = int(max(self.cfg.note_min, min(self.cfg.note_max, note)))
        i = note - self.cfg.note_min
        x0 = int(np.floor(i * self.key_w))
        x1 = int(np.floor((i + 1) * self.key_w) - 1)
        x0 = max(0, min(self.cfg.plane_w - 1, x0))
        x1 = max(0, min(self.cfg.plane_w - 1, x1))
        if x1 < x0:
            x1 = x0
        return x0, x1

    def note_to_cell(self, note: int):
        note = int(max(self.cfg.note_min, min(self.cfg.note_max, note)))
        return self._note_cells[note - self.cfg.note_min]

    @staticmethod
    def vel_gain(vel: int) -> float:
        v = np.clip(vel / 127.0, 0.0, 1.0)
        return float(v ** 0.30)

    def _calc_stamp_color_bgr(self, vel: int):
        g = self.vel_gain(vel)
        base = np.array([210, 80, 255], dtype=np.float32)
        bright = 0.48 + 0.78 * g
        col = np.clip(base * bright, 0, 255).astype(np.uint8)
        return int(col[0]), int(col[1]), int(col[2])

    def stamp_color_bgr(self, vel: int):
        return self._stamp_color_lut[int(np.clip(vel, 0, 127))]

    def _calc_glow_color_bgr(self, vel: int):
        g = self.vel_gain(vel)
        base = np.array([255, 180, 255], dtype=np.float32)
        bright = 0.12 + 0.52 * g
        col = np.clip(base * bright, 0, 255).astype(np.uint8)
        return int(col[0]), int(col[1]), int(col[2])

    def glow_color_bgr(self, vel: int):
        return self._glow_color_lut[int(np.clip(vel, 0, 127))]

    def scroll_and_fade(self):
        self._frame_idx += 1
        spx = int(self.cfg.speed_px)
        if spx <= 0:
            return
        if spx >= self.cfg.plane_h:
            self.plane.fill(0)
        else:
            self.plane[:-spx, :, :] = self.plane[spx:, :, :]
            self.plane[-spx:, :, :] = 0
        cv2.convertScaleAbs(self.plane, self._tmp_fade, alpha=self.cfg.fade)
        self.plane[:] = self._tmp_fade
        k = self.cfg.deband_blur_k
        blur_every = max(1, int(self.cfg.deband_blur_every))
        if k >= 3 and k % 2 == 1 and (self._frame_idx % blur_every == 0):
            self.plane[:] = cv2.GaussianBlur(self.plane, (3, k), sigmaX=0.0, sigmaY=0.0)

    def scroll_and_fade_particles(self):
        self.pplane[:] = 0

    def _calc_firework_color(self, vel: int):
        base = np.array(self._calc_stamp_color_bgr(vel), dtype=np.float32)
        bloom = np.array([15, 30, 10], dtype=np.float32)
        col = np.clip(base * 1.08 + bloom, 0, 255).astype(np.uint8)
        return int(col[0]), int(col[1]), int(col[2])

    def _firework_color(self, vel: int):
        return self._firework_color_lut[int(np.clip(vel, 0, 127))]

    def emit_firework_from_note(self, note: int, vel: int):
        x0p, x1p = self.particle_note_to_cell(note)
        x_center = 0.5 * (x0p + x1p)
        y_base = float(self.ph - 2)
        g = self.vel_gain(vel)
        emit_count = max(6, int(self.cfg.firework_emit_count * (0.75 + g * 0.95)))
        color = self._firework_color(vel)

        for _ in range(emit_count):
            vx = np.random.normal(0.0, self.cfg.firework_emit_vx) * (0.55 + 0.35 * g)
            vy = -np.random.uniform(self.cfg.firework_emit_vy_min, self.cfg.firework_emit_vy_max)
            vy *= (0.75 + g * 0.6)
            vy -= abs(np.random.normal(0.0, self.cfg.firework_emit_spread))
            life = np.random.randint(self.cfg.firework_life_min, self.cfg.firework_life_max + 1)
            self.firework_particles.append({
                "x": x_center + np.random.normal(0.0, 0.45),
                "y": y_base,
                "px": x_center,
                "py": y_base,
                "vx": vx,
                "vy": vy,
                "life": life,
                "max_life": life,
                "color": color,
                "size": float(np.random.uniform(1.0, 2.2)),
            })

    def update_and_draw_fireworks(self):
        if not self.firework_particles:
            return

        alive_particles = []
        for p in self.firework_particles:
            p["px"] = p["x"]
            p["py"] = p["y"]
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= self.cfg.firework_drag
            p["vy"] += self.cfg.firework_gravity
            p["life"] -= 1

            if p["life"] <= 0:
                continue
            if p["x"] < 0 or p["x"] >= self.pw or p["y"] < 0 or p["y"] >= self.ph:
                continue

            fade = p["life"] / max(1, p["max_life"])
            bgr = tuple(int(c * fade) for c in p["color"])
            x0, y0 = int(p["px"]), int(p["py"])
            x1, y1 = int(p["x"]), int(p["y"])
            trail_thickness = 1 if fade < 0.65 else 2
            cv2.line(self.pplane, (x0, y0), (x1, y1), bgr, thickness=trail_thickness, lineType=cv2.LINE_8)
            r = max(1, int(p["size"] * (0.35 + fade)))
            cv2.circle(self.pplane, (x1, y1), r, bgr, thickness=-1, lineType=cv2.LINE_8)
            alive_particles.append(p)

        self.firework_particles = alive_particles

    def draw_note_stamp_at_bottom(self, note: int, vel: int):
        x0, x1 = self.note_to_cell(note)
        if x1 - x0 >= 4:
            x0 += 1
            x1 -= 1
        stamp_h = self.cfg.speed_px if self.cfg.stamp_h is None else int(self.cfg.stamp_h)
        y1 = self.cfg.plane_h - 1
        y0 = max(0, self.cfg.plane_h - stamp_h)
        cv2.rectangle(self.plane, (x0, y0), (x1, y1), self.stamp_color_bgr(vel), thickness=-1)

        if not self.cfg.glow:
            return
        ge_y = int(self.cfg.glow_expand_y * (0.7 + vel / 127.0))
        gx0, gx1 = x0, x1
        gy0 = max(0, y0 - ge_y)
        gy1 = min(self.cfg.plane_h - 1, y1 + ge_y)
        cv2.rectangle(self.plane, (gx0, gy0), (gx1, gy1), self.glow_color_bgr(vel), thickness=-1)

    def particle_note_to_cell(self, note: int):
        x0, x1 = self.note_to_cell(note)
        x0p = int(x0 / self.cfg.part_scale)
        x1p = int(x1 / self.cfg.part_scale)
        x0p = max(0, min(self.pw - 1, x0p))
        x1p = max(0, min(self.pw - 1, x1p))
        if x1p < x0p:
            x1p = x0p
        return x0p, x1p

    def add_ambient_fog(self):
        y_base = self.ph - 2
        n = max(0, int(self.cfg.ambient_fog_rate / 60.0))
        if n <= 0:
            return
        xs = np.random.randint(0, self.pw, size=n)
        ys = np.random.randint(max(0, y_base - 10), y_base + 1, size=n)
        br = self.cfg.ambient_fog_bright
        p = self.pplane[ys, xs, :].astype(np.int16)
        p[:, 0] = np.clip(p[:, 0] + br, 0, 255)
        p[:, 1] = np.clip(p[:, 1] + int(br * 0.95), 0, 255)
        p[:, 2] = np.clip(p[:, 2] + int(br * 1.0), 0, 255)
        self.pplane[ys, xs, :] = p.astype(np.uint8)

    def add_note_sparks(self, midi_events):
        for event_type, note, vel in midi_events:
            if event_type == "note_on" and vel > 0:
                self.emit_firework_from_note(note, vel)
        self.update_and_draw_fireworks()

    def add_bottom_noise(self):
        h = min(self.cfg.bottom_noise_h, self.ph)
        if h <= 0:
            return
        noise = (np.random.randn(h, self.pw, 1) * 18.0 + 18.0).astype(np.int16)
        noise = np.clip(noise, 0, 40)
        strip = self.pplane[self.ph - h:self.ph, :, :].astype(np.int16)
        strip[:, :, 0] = np.clip(strip[:, :, 0] + noise[:, :, 0], 0, 255)
        strip[:, :, 1] = np.clip(strip[:, :, 1] + (noise[:, :, 0] * 0.7).astype(np.int16), 0, 255)
        strip[:, :, 2] = np.clip(strip[:, :, 2] + (noise[:, :, 0] * 0.8).astype(np.int16), 0, 255)
        self.pplane[self.ph - h:self.ph, :, :] = strip.astype(np.uint8)

    def get_fly_quad_extended(self):
        if self.state.fly_quad_base is None:
            return None
        L1, R1, R0, L0 = self.state.fly_quad_base.astype(np.float32)
        vL, vR = (L1 - L0), (R1 - R0)
        L1e = L0 + vL * float(self.cfg.extend_scale)
        R1e = R0 + vR * float(self.cfg.extend_scale)
        return np.float32([L1e, R1e, R0, L0])

    def _ensure_homography(self, quad):
        sig = tuple(np.round(quad.flatten(), 2))
        if sig == self._last_quad_sig:
            return
        self._H_bars = cv2.getPerspectiveTransform(self._src_plane, quad)
        self._H_parts = cv2.getPerspectiveTransform(self._src_pplane, quad)
        self._last_quad_sig = sig

    def warp_to_frame(self, frame_shape, quad):
        self.ensure_overlays(frame_shape)
        if quad is None:
            self.overlay_bars.fill(0)
            self.overlay_particles.fill(0)
            return self.overlay_bars, self.overlay_particles

        self._ensure_homography(quad)

        if self.cfg.use_umat:
            ubars = cv2.UMat(self.plane)
            warped_bars = cv2.warpPerspective(ubars, self._H_bars, (frame_shape[1], frame_shape[0]))
            self.overlay_bars[:] = warped_bars.get()
            if self.cfg.fx_enabled and self.cfg.particle_fx:
                uparts = cv2.UMat(self.pplane)
                warped_parts = cv2.warpPerspective(uparts, self._H_parts, (frame_shape[1], frame_shape[0]))
                self.overlay_particles[:] = warped_parts.get()
        else:
            cv2.warpPerspective(self.plane, self._H_bars, (frame_shape[1], frame_shape[0]), dst=self.overlay_bars)
            if self.cfg.fx_enabled and self.cfg.particle_fx:
                cv2.warpPerspective(self.pplane, self._H_parts, (frame_shape[1], frame_shape[0]), dst=self.overlay_particles)
        return self.overlay_bars, self.overlay_particles

    @staticmethod
    def blend_additive(frame: np.ndarray, overlay: np.ndarray, alpha: float):
        return cv2.add(frame, cv2.convertScaleAbs(overlay, alpha=alpha))

    @staticmethod
    def blend_additive_inplace(frame: np.ndarray, overlay: np.ndarray, alpha: float, buffer: np.ndarray):
        cv2.convertScaleAbs(overlay, dst=buffer, alpha=alpha)
        cv2.add(frame, buffer, dst=frame)
        return frame

    def _build_line_glow_overlay(self, out_shape):
        quad = self.state.fly_quad_base
        if quad is None:
            self._line_glow_overlay = None
            self._line_glow_sig = None
            return
        h, w = out_shape[:2]
        sig = (tuple(np.round(quad.flatten(), 2)), h, w, self.cfg.line_glow_passes, self.cfg.line_glow_base_thick, self.cfg.line_glow_extra_thick)
        if sig == self._line_glow_sig and self._line_glow_overlay is not None:
            return
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        L1, R1, R0, L0 = quad.astype(np.int32)
        p0, p1 = (int(L0[0]), int(L0[1])), (int(R0[0]), int(R0[1]))
        for i in range(self.cfg.line_glow_passes):
            t = self.cfg.line_glow_base_thick + int((i / max(1, self.cfg.line_glow_passes - 1)) * self.cfg.line_glow_extra_thick)
            cv2.line(overlay, p0, p1, (240, 220, 245), thickness=t, lineType=cv2.LINE_AA)
        self._line_glow_overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=5.0, sigmaY=5.0)
        self._line_glow_sig = sig

    def draw_foldline_glow(self, out: np.ndarray):
        if self.state.fly_quad_base is None:
            return out
        self._build_line_glow_overlay(out.shape)
        if self._line_glow_overlay is None:
            return out
        return self.blend_additive(out, self._line_glow_overlay, self.cfg.line_glow_alpha)

    def start_calibration(self):
        self.state.calib_mode = True
        self.state.calib_points = []
        self.state.fly_quad_base = None
        self.state.selected_idx = -1
        self.state.hover_idx = -1
        print("\n[CALIB] Click 4 points in order:")
        for i, lab in enumerate(CALIB_LABELS, 1):
            print(f"  {i}) {lab}")

    def reset_calibration(self):
        self.state.calib_mode = False
        self.state.calib_points = []
        self.state.fly_quad_base = None
        self.cfg.edit_mode = False
        self.state.selected_idx = -1
        self.state.hover_idx = -1
        print("[CALIB] Reset.")

    def draw_calibration_overlay(self, img: np.ndarray):
        if self.state.calib_mode:
            h, w = img.shape[:2]
            step = max(40, min(h, w) // 12)
            for x in range(step, w, step):
                cv2.line(img, (x, 0), (x, h - 1), (40, 40, 40), 1)
            for y in range(step, h, step):
                cv2.line(img, (0, y), (w - 1, y), (40, 40, 40), 1)

            next_idx = len(self.state.calib_points)
            help_text = "Click 4 points: L0 -> R0 -> L1 -> R1"
            cv2.putText(img, help_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            if next_idx < 4:
                cv2.putText(img, f"Next: {CALIB_LABELS[next_idx]}", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

            for i, (x, y) in enumerate(self.state.calib_points):
                cv2.drawMarker(img, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(img, CALIB_LABELS[i], (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if self.state.fly_quad_base is not None:
            self.draw_edit_overlay(img)

    def pick_handle(self, pt: np.ndarray):
        quad = self.state.fly_quad_base
        best_i, best_d = -1, 1e18
        for i in range(4):
            d = float(np.sum((quad[i] - pt) ** 2))
            if d < best_d:
                best_d, best_i = d, i
        if best_d <= (self.cfg.handle_hit_radius * self.cfg.handle_hit_radius):
            return best_i
        return -1

    def draw_edit_overlay(self, img: np.ndarray):
        quad = self.state.fly_quad_base
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), self.cfg.line_thickness)
        for i, p in enumerate(quad.astype(np.int32)):
            x, y = int(p[0]), int(p[1])
            cv2.circle(img, (x, y), self.cfg.handle_radius, (0, 255, 255), -1)
            if i == self.state.hover_idx:
                cv2.circle(img, (x, y), self.cfg.handle_ring_radius, (255, 255, 0), 1)
            if i == self.state.selected_idx:
                cv2.circle(img, (x, y), self.cfg.handle_ring_radius + 4, (0, 255, 0), 2)
            cv2.putText(img, PT_NAMES_QUAD[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    def handle_mouse(self, event, x, y):
        pt = np.array([x, y], dtype=np.float32)
        if self.cfg.edit_mode and self.state.fly_quad_base is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                self.state.hover_idx = self.pick_handle(pt)
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self.pick_handle(pt)
                if self.state.selected_idx == -1:
                    if idx != -1:
                        self.state.selected_idx = idx
                    return
                if idx != -1:
                    self.state.selected_idx = idx
                    return
                self.state.fly_quad_base[self.state.selected_idx] = pt
                self._last_quad_sig = None
            return

        if self.state.calib_mode and event == cv2.EVENT_LBUTTONDOWN:
            idx = len(self.state.calib_points)
            if idx < 4:
                print(f"[CALIB] {CALIB_LABELS[idx]} = ({x}, {y})")
                self.state.calib_points.append((x, y))
                if len(self.state.calib_points) == 4:
                    L0 = np.float32(self.state.calib_points[0])
                    R0 = np.float32(self.state.calib_points[1])
                    L1 = np.float32(self.state.calib_points[2])
                    R1 = np.float32(self.state.calib_points[3])
                    self.state.fly_quad_base = np.float32([L1, R1, R0, L0])
                    self.state.calib_mode = False
                    self._last_quad_sig = None
                    print("[CALIB] Completed fly-out plane calibration.")
