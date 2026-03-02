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

        self.overlay_bars = None
        self.overlay_particles = None

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

    def note_to_cell(self, note: int):
        note = int(max(self.cfg.note_min, min(self.cfg.note_max, note)))
        i = note - self.cfg.note_min
        x0 = int(np.floor(i * self.key_w))
        x1 = int(np.floor((i + 1) * self.key_w) - 1)
        x0 = max(0, min(self.cfg.plane_w - 1, x0))
        x1 = max(0, min(self.cfg.plane_w - 1, x1))
        if x1 < x0:
            x1 = x0
        return x0, x1

    @staticmethod
    def vel_gain(vel: int) -> float:
        v = np.clip(vel / 127.0, 0.0, 1.0)
        return float(v ** 0.30)

    def stamp_color_bgr(self, vel: int):
        g = self.vel_gain(vel)
        base = np.array([210, 80, 255], dtype=np.float32)
        bright = 0.48 + 0.78 * g
        col = np.clip(base * bright, 0, 255).astype(np.uint8)
        return int(col[0]), int(col[1]), int(col[2])

    def glow_color_bgr(self, vel: int):
        g = self.vel_gain(vel)
        base = np.array([255, 180, 255], dtype=np.float32)
        bright = 0.12 + 0.52 * g
        col = np.clip(base * bright, 0, 255).astype(np.uint8)
        return int(col[0]), int(col[1]), int(col[2])

    def scroll_and_fade(self):
        spx = self.cfg.speed_px
        self.plane[:] = np.roll(self.plane, -spx, axis=0)
        self.plane[-spx:, :, :] = 0
        cv2.convertScaleAbs(self.plane, self._tmp_fade, alpha=self.cfg.fade)
        self.plane[:] = self._tmp_fade

    def scroll_and_fade_particles(self):
        spx = max(1, int(self.cfg.speed_px / max(1, self.cfg.part_speed_div)))
        self.pplane[:] = np.roll(self.pplane, -spx, axis=0)
        self.pplane[-spx:, :, :] = 0
        cv2.convertScaleAbs(self.pplane, self._tmp_pfade, alpha=self.cfg.part_fade)
        self.pplane[:] = self._tmp_pfade

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

    def add_note_sparks(self, held_notes):
        y_base = self.ph - 2
        for note, vel in held_notes:
            x0p, x1p = self.particle_note_to_cell(note)
            xp = x0p if x1p <= x0p else np.random.randint(x0p, x1p + 1)
            g = self.vel_gain(vel)
            k = int(self.cfg.note_spark_rate * (0.7 + 2.0 * g))
            if k <= 0:
                continue
            xs = np.random.randint(max(0, xp - self.cfg.spark_x_jitter), min(self.pw, xp + self.cfg.spark_x_jitter + 1), size=k)
            ys = np.random.randint(max(0, y_base - self.cfg.spark_y_jitter), y_base + 1, size=k)
            br = int(self.cfg.note_spark_bright * (0.6 + 0.8 * g))
            p = self.pplane[ys, xs, :].astype(np.int16)
            p[:, 0] = np.clip(p[:, 0] + br, 0, 255)
            p[:, 1] = np.clip(p[:, 1] + int(br * 0.8), 0, 255)
            p[:, 2] = np.clip(p[:, 2] + int(br * 1.0), 0, 255)
            self.pplane[ys, xs, :] = p.astype(np.uint8)

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
        self.overlay_bars.fill(0)
        self.overlay_particles.fill(0)
        if quad is None:
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

    def draw_foldline_glow(self, out: np.ndarray):
        quad = self.state.fly_quad_base
        if quad is None:
            return out
        L1, R1, R0, L0 = quad.astype(np.int32)
        p0, p1 = (int(L0[0]), int(L0[1])), (int(R0[0]), int(R0[1]))
        overlay = np.zeros_like(out)
        for i in range(self.cfg.line_glow_passes):
            t = self.cfg.line_glow_base_thick + int((i / max(1, self.cfg.line_glow_passes - 1)) * self.cfg.line_glow_extra_thick)
            cv2.line(overlay, p0, p1, (240, 220, 245), thickness=t, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=5.0, sigmaY=5.0)
        return self.blend_additive(out, overlay, self.cfg.line_glow_alpha)

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
