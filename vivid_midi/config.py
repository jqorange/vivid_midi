from dataclasses import dataclass


@dataclass
class RenderConfig:
    cam_index: int = 0
    plane_w: int = 1280
    plane_h: int = 2600

    note_min: int = 21
    note_max: int = 108

    speed_px: int = 42
    fade: float = 0.9976
    base_alpha: float = 1.65  # reduce bar brightness
    show_bars: bool = True
    deband_blur_k: int = 5

    stamp_h: int | None = None

    glow: bool = True
    glow_expand_y: int = 26
    glow_expand_x: int = 0

    extend_scale: float = 3.0
    extend_step: float = 0.25
    extend_min: float = 1.0
    extend_max: float = 12.0

    edit_mode: bool = False
    handle_radius: int = 5
    handle_hit_radius: int = 18
    handle_ring_radius: int = 10
    line_thickness: int = 1

    fx_enabled: bool = True

    # Try OpenCL + UMat to leverage mac GPU path when available.
    use_umat: bool = True

    line_glow: bool = True
    line_glow_passes: int = 3
    line_glow_base_thick: int = 2
    line_glow_extra_thick: int = 10
    line_glow_alpha: float = 0.68

    particle_fx: bool = True
    part_scale: int = 2
    part_fade: float = 0.9925
    part_speed_div: int = 2

    particle_alpha: float = 1.55
    particle_blur_every: int = 3
    particle_blur_k: int = 5

    firework_emit_count: int = 20
    firework_emit_spread: float = 1.2
    firework_emit_vy_min: float = 7.0
    firework_emit_vy_max: float = 13.5
    firework_emit_vx: float = 1.1
    firework_gravity: float = 0.33
    firework_drag: float = 0.985
    firework_life_min: int = 18
    firework_life_max: int = 34

    # Heavier fog visibility
    ambient_fog_rate: int = 0
    ambient_fog_bright: int = 185

    note_spark_rate: int = 7
    note_spark_bright: int = 220

    spark_x_jitter: int = 2
    spark_y_jitter: int = 6

    bottom_noise: bool = False
    bottom_noise_h: int = 14

    @property
    def note_range(self) -> int:
        return self.note_max - self.note_min + 1


@dataclass
class RuntimeState:
    calib_mode: bool = False
    calib_points: list[tuple[int, int]] | None = None
    fly_quad_base: object | None = None
    hover_idx: int = -1
    selected_idx: int = -1

    def __post_init__(self):
        if self.calib_points is None:
            self.calib_points = []
