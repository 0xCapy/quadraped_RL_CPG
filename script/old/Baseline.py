#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bittle baseline (structured, documented, RL-ready)
===================================================

This script keeps the *same baseline philosophy* as the current clean v2 run:
- fixed validated sign system
- fixed validated support working point
- low-dimensional joint-space diagonal trot around q_support
- actual foot trajectory export from simulation

The purpose of this v3 file is NOT to redesign the controller again.
It is to make the code easier to maintain, tune, and hand over to later RL work.

What is improved relative to the previous file
----------------------------------------------
1) Code organization
   - clear section layout
   - grouped parameter interfaces
   - less duplicated logic in logging / export

2) Parameter interface
   - all user-facing tuning parameters are concentrated in one place
   - support / sim / gait / tracking / export are explicitly separated
   - comments state what each parameter is for and what to tune first

3) Documentation
   - the baseline design and each gait sub-phase are documented in code
   - helper functions are named and commented by purpose

4) Minor implementation cleanup
   - track points are fetched once per simulation step, not once per leg
   - exports are written through dedicated helpers
   - stage-prim foot tracking remains the preferred route

Authoritative constraints retained
----------------------------------
- Do NOT auto-infer signs.
- Do NOT auto-calibrate support posture.
- Keep the validated sign/support truths explicit and fixed.

Expected output files
---------------------
All outputs are written to:
    D:\\Project\\RLCPG\\quadraped_RL_CPG\\script\\tem_doc

This script exports:
- *_metrics.csv      : root state + per-leg commanded/scalar/track data
- *_foot_traj.csv    : flattened per-leg foot trajectories (world + body frame)
- *_run_info.txt     : exact run configuration and tracked prims
- *_exec.log         : success/failure summary
"""

from __future__ import annotations

import csv
import inspect
import math
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omni.isaac.kit import SimulationApp

# -----------------------------------------------------------------------------
# Isaac Sim must be created before importing isaaclab / pxr / omni modules.
# -----------------------------------------------------------------------------
HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))
simulation_app = SimulationApp({"headless": HEADLESS})

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS, SUPPORT_POSTURE_OFFSETS  # noqa: E402


# =============================================================================
# Global fixed truths and robot naming
# =============================================================================

TEM_DOC = Path(r"D:\Project\RLCPG\quadraped_RL_CPG\script\tem_doc")

LEGS = ["LF", "RF", "LB", "RB"]
GROUP_A = ["LF", "RB"]
GROUP_B = ["RF", "LB"]
FRONT_LEGS = ["LF", "RF"]
BACK_LEGS = ["LB", "RB"]

# Validated sign truth. Do not replace with automation.
SH_SIGN = {"LF": -1.0, "LB": -1.0, "RF": +1.0, "RB": +1.0}
KN_SIGN = {"LF": -1.0, "LB": -1.0, "RF": +1.0, "RB": +1.0}

LEG_TO_SHOULDER_NAME = {
    "LF": "left_front_shoulder_joint",
    "RF": "right_front_shoulder_joint",
    "LB": "left_back_shoulder_joint",
    "RB": "right_back_shoulder_joint",
}
LEG_TO_KNEE_NAME = {
    "LF": "left_front_knee_joint",
    "RF": "right_front_knee_joint",
    "LB": "left_back_knee_joint",
    "RB": "right_back_knee_joint",
}

# Tokens used when searching stage prims / articulation body names for distal feet.
LEG_NAME_TOKENS = {
    "LF": ["left_front", "front_left", "leftfront", "lf"],
    "RF": ["right_front", "front_right", "rightfront", "rf"],
    "LB": ["left_back", "back_left", "leftback", "left_rear", "rear_left", "lb"],
    "RB": ["right_back", "back_right", "rightback", "right_rear", "rear_right", "rb"],
}
DISTAL_BODY_SCORE = [
    ("toeproxy", 140),
    ("toe_proxy", 138),
    ("toe", 132),
    ("foot", 126),
    ("paw", 120),
    ("tip", 112),
    ("contact", 104),
    ("end", 96),
    ("distal", 88),
    ("lower", 70),
    ("shin", 66),
    ("ankle", 62),
    ("knee", 46),
    ("leg", 30),
]


# =============================================================================
# Parameter interfaces
# =============================================================================

@dataclass
class SupportParams:
    """Fixed support working-point offsets, sourced from cfg truth."""

    front_sh_y: float = float(SUPPORT_POSTURE_OFFSETS["front_sh_y"])
    back_sh_y: float = float(SUPPORT_POSTURE_OFFSETS["back_sh_y"])
    front_kn_crouch: float = float(SUPPORT_POSTURE_OFFSETS["front_kn_crouch"])
    back_kn_crouch: float = float(SUPPORT_POSTURE_OFFSETS["back_kn_crouch"])


@dataclass
class SimParams:
    """Simulation-level settings.

    Tune here only if you need a different runtime, dt, camera, or device.
    """

    root_z0: float = 1.0
    dt: float = 1.0 / 120.0
    device: str = os.environ.get("BITTLE_DEVICE", "cuda:0")
    camera_eye: Tuple[float, float, float] = (2.0, 2.0, 1.25)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.12)
    warmup_time: float = 1.2
    run_time: float = 12.0
    print_every_n_steps: int = 20


@dataclass
class VisualParams:
    """Purely visual stage aids. Safe to disable if you want a cleaner stage."""

    show_grid: bool = True
    show_axes: bool = True
    half_extent: float = 2.0
    grid_spacing: float = 0.50
    z_offset: float = 0.0015
    grid_thickness: float = 0.0018
    axis_thickness: float = 0.0060


@dataclass
class GaitParams:
    """Low-dimensional gait shaping interface.

    Tune priority for future work:
    1) shoulder_amp
    2) knee_lift_amp
    3) freq_hz
    4) swing_ratio
    5) touchdown_buffer_ratio

    Keep the rest as shape refinements unless you are explicitly debugging phase
    continuity or touchdown quality.
    """

    # Main gait parameters.
    freq_hz: float = 1
    swing_ratio: float = 0.36
    shoulder_amp: float = 0.14
    knee_lift_amp: float = 0.26
    touchdown_buffer_ratio: float = 0.24
    ramp_time: float = 2.5

    # Swing / touchdown shape parameters.
    liftoff_ratio: float = 0.30
    predown_ratio: float = 0.22
    liftoff_shoulder_frac: float = 0.18
    transfer_knee_floor_frac: float = 0.74
    touchdown_knee_hold_frac: float = 0.16
    touchdown_shoulder_end_frac: float = 0.92


@dataclass
class JointLimitParams:
    """Conservative joint clamps used after composing q_cmd."""

    shoulder_lim: Tuple[float, float] = (-1.15, 1.15)
    knee_lim: Tuple[float, float] = (-1.55, 1.55)


@dataclass
class TrackingParams:
    """Tracking/export behavior.

    prefer_stage_prim=True means we prefer actual USD prim tracking such as
    toe_proxy, which is better for paper plots than stopping at articulation
    body names like knee_link.
    """

    prefer_stage_prim: bool = True
    export_body_frame: bool = True
    export_world_frame: bool = True


@dataclass
class ExportParams:
    """File naming interface only. Does not affect gait behavior."""

    file_stem: str = "bittle_baseline_clean_track_v3_structured"


@dataclass
class BaselineConfig:
    """Single top-level config object for this script."""

    support: SupportParams = field(default_factory=SupportParams)
    sim: SimParams = field(default_factory=SimParams)
    visual: VisualParams = field(default_factory=VisualParams)
    gait: GaitParams = field(default_factory=GaitParams)
    limits: JointLimitParams = field(default_factory=JointLimitParams)
    tracking: TrackingParams = field(default_factory=TrackingParams)
    export: ExportParams = field(default_factory=ExportParams)


CFG = BaselineConfig()


# =============================================================================
# Small utility helpers
# =============================================================================

def ensure_output_dir() -> None:
    TEM_DOC.mkdir(parents=True, exist_ok=True)


def clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def smoothstep01(x: float) -> float:
    """C1-smooth interpolation on [0, 1]."""
    x = clampf(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quat_to_euler_xyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Quaternion (wxyz) -> XYZ Euler angles, used only for logging."""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def quat_conjugate_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_mul_wxyz(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_rotate_wxyz(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    p = (0.0, float(v[0]), float(v[1]), float(v[2]))
    qp = quat_mul_wxyz(q, p)
    qpq = quat_mul_wxyz(qp, quat_conjugate_wxyz(q))
    return (qpq[1], qpq[2], qpq[3])


def world_to_body(
    root_pos: Tuple[float, float, float],
    root_quat_wxyz: Tuple[float, float, float, float],
    p_world: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Express a world point in the robot body frame defined by the root pose."""
    rel = (
        float(p_world[0]) - float(root_pos[0]),
        float(p_world[1]) - float(root_pos[1]),
        float(p_world[2]) - float(root_pos[2]),
    )
    return quat_rotate_wxyz(quat_conjugate_wxyz(root_quat_wxyz), rel)


# =============================================================================
# Joint map and posture construction
# =============================================================================

def build_joint_maps(joint_names: List[str]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Build fast joint-name lookup dictionaries."""
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    leg_to_sh = {leg: name_to_idx[LEG_TO_SHOULDER_NAME[leg]] for leg in LEGS}
    leg_to_kn = {leg: name_to_idx[LEG_TO_KNEE_NAME[leg]] for leg in LEGS}
    return name_to_idx, leg_to_sh, leg_to_kn


def build_stand_pose_from_cfg(jmap: Dict[str, int], default_joint_pos: torch.Tensor) -> torch.Tensor:
    """Recover the original stand pose explicitly from cfg truth."""
    q = default_joint_pos.clone()
    for joint_name, val in STAND_JOINT_POS.items():
        if joint_name in jmap:
            q[0, jmap[joint_name]] = float(val)
    return q


def apply_joint_limits(q: torch.Tensor, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int]) -> torch.Tensor:
    """Clamp composed command q_cmd to conservative shoulder/knee ranges."""
    q = q.clone()
    for leg in LEGS:
        q[0, leg_to_sh[leg]] = clampf(float(q[0, leg_to_sh[leg]]), *CFG.limits.shoulder_lim)
        q[0, leg_to_kn[leg]] = clampf(float(q[0, leg_to_kn[leg]]), *CFG.limits.knee_lim)
    return q


def apply_support_posture(q_stand: torch.Tensor, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int]) -> torch.Tensor:
    """Construct q_support from q_stand using the fixed validated support offsets."""
    q = q_stand.clone()
    for leg in FRONT_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * CFG.support.front_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * CFG.support.front_kn_crouch
    for leg in BACK_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * CFG.support.back_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * CFG.support.back_kn_crouch
    return apply_joint_limits(q, leg_to_sh, leg_to_kn)


# =============================================================================
# Gait phase logic
# =============================================================================

def leg_phase(gait_t: float, leg: str) -> float:
    """Per-leg phase in [0,1), with Group B shifted by half a cycle."""
    phase0 = 0.0 if leg in GROUP_A else 0.5
    return (phase0 + CFG.gait.freq_hz * gait_t) % 1.0


def leg_template_scalars(u: float) -> Tuple[float, float, str]:
    """Return (shoulder_scalar, knee_scalar, phase_name) for one leg.

    This is still a joint-space baseline, not a task-space foot planner.
    The design goal is simply to make the baseline cleaner than the old
    single-bump template, especially around liftoff and touchdown.

    Phase layout
    ------------
    Swing
      1) swing_liftoff   : get clearance first, but begin forward carry early
      2) swing_transfer  : carry leg forward while keeping useful clearance
      3) swing_predown   : lower the leg smoothly for touchdown

    Stance
      4) touchdown_buffer: short support-establishment hold
      5) stance_sweep    : sweep back toward q_support
    """
    sr = clampf(CFG.gait.swing_ratio, 0.10, 0.90)

    # -----------------------------
    # SWING
    # -----------------------------
    if u < sr:
        s = u / sr
        liftoff_end = clampf(CFG.gait.liftoff_ratio, 0.05, 0.80)
        predown_start = clampf(1.0 - CFG.gait.predown_ratio, liftoff_end + 0.05, 0.98)

        # 1) Liftoff: unload / rise, while already moving forward a little.
        if s < liftoff_end:
            a = s / liftoff_end
            shoulder_scalar = CFG.gait.shoulder_amp * CFG.gait.liftoff_shoulder_frac * smoothstep01(a)
            knee_scalar = CFG.gait.knee_lift_amp * 0.90 * smoothstep01(a)
            return shoulder_scalar, knee_scalar, "swing_liftoff"

        # 2) Transfer: bring the leg forward, keep clearance reasonably high.
        if s < predown_start:
            a = (s - liftoff_end) / max(1e-9, predown_start - liftoff_end)
            shoulder_scalar = CFG.gait.shoulder_amp * (
                CFG.gait.liftoff_shoulder_frac
                + (1.0 - CFG.gait.liftoff_shoulder_frac) * smoothstep01(a)
            )
            knee_floor = clampf(CFG.gait.transfer_knee_floor_frac, 0.35, 1.00)
            knee_scalar = CFG.gait.knee_lift_amp * (
                knee_floor + (1.0 - knee_floor) * math.sin(0.5 * math.pi * a)
            )
            return shoulder_scalar, knee_scalar, "swing_transfer"

        # 3) Pre-touchdown: keep shoulder forward, lower the leg smoothly.
        a = (s - predown_start) / max(1e-9, 1.0 - predown_start)
        shoulder_scalar = CFG.gait.shoulder_amp
        knee_scalar = CFG.gait.knee_lift_amp * (
            CFG.gait.touchdown_knee_hold_frac
            + (1.0 - CFG.gait.touchdown_knee_hold_frac) * (1.0 - smoothstep01(a))
        )
        return shoulder_scalar, knee_scalar, "swing_predown"

    # -----------------------------
    # STANCE
    # -----------------------------
    w = (u - sr) / max(1e-9, 1.0 - sr)
    td = clampf(CFG.gait.touchdown_buffer_ratio, 0.0, 0.80)

    # 4) Touchdown buffer: do not instantly kill knee flex at contact.
    if w < td:
        a = 0.0 if td < 1e-9 else (w / td)
        sh_end_frac = clampf(CFG.gait.touchdown_shoulder_end_frac, 0.70, 1.00)
        shoulder_scalar = CFG.gait.shoulder_amp * (1.0 - (1.0 - sh_end_frac) * smoothstep01(a))
        knee_scalar = CFG.gait.knee_lift_amp * CFG.gait.touchdown_knee_hold_frac * (1.0 - smoothstep01(a))
        return shoulder_scalar, knee_scalar, "touchdown_buffer"

    # 5) Stance sweep: support phase returns toward q_support.
    v = (w - td) / max(1e-9, 1.0 - td)
    sh_end_frac = clampf(CFG.gait.touchdown_shoulder_end_frac, 0.70, 1.00)
    shoulder_scalar = CFG.gait.shoulder_amp * sh_end_frac * (1.0 - smoothstep01(v))
    knee_scalar = 0.0
    return shoulder_scalar, knee_scalar, "stance_sweep"


# =============================================================================
# Simple stage visuals
# =============================================================================

def _set_translate_scale(prim, translate_xyz: Tuple[float, float, float], scale_xyz: Tuple[float, float, float]) -> None:
    xf = UsdGeom.Xformable(prim)
    try:
        xf.ClearXformOpOrder()
    except Exception:
        pass
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate_xyz))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale_xyz))


def spawn_floor_reference(stage) -> None:
    """Spawn optional grid / axes to make gait motion easier to inspect visually."""
    UsdGeom.Xform.Define(stage, "/World/FloorReference")
    half = float(CFG.visual.half_extent)
    spacing = max(1e-6, float(CFG.visual.grid_spacing))
    z0 = float(CFG.visual.z_offset)
    gth = float(CFG.visual.grid_thickness)
    ath = float(CFG.visual.axis_thickness)

    if CFG.visual.show_grid:
        n = int(round(half / spacing))
        for i in range(-n, n + 1):
            coord = i * spacing

            grid_x = UsdGeom.Cube.Define(stage, f"/World/FloorReference/grid_x_{i:+03d}".replace("+", "p").replace("-", "m"))
            grid_x.GetSizeAttr().Set(1.0)
            _set_translate_scale(grid_x.GetPrim(), (coord, 0.0, z0), (gth, half, gth))

            grid_y = UsdGeom.Cube.Define(stage, f"/World/FloorReference/grid_y_{i:+03d}".replace("+", "p").replace("-", "m"))
            grid_y.GetSizeAttr().Set(1.0)
            _set_translate_scale(grid_y.GetPrim(), (0.0, coord, z0), (half, gth, gth))

    if CFG.visual.show_axes:
        axis_x = UsdGeom.Cube.Define(stage, "/World/FloorReference/axis_x_red")
        axis_x.GetSizeAttr().Set(1.0)
        _set_translate_scale(axis_x.GetPrim(), (0.0, 0.0, z0 + 0.0006), (half, ath, ath))

        axis_y = UsdGeom.Cube.Define(stage, "/World/FloorReference/axis_y_green")
        axis_y.GetSizeAttr().Set(1.0)
        _set_translate_scale(axis_y.GetPrim(), (0.0, 0.0, z0 + 0.0006), (ath, half, ath))

        try:
            UsdGeom.Gprim(axis_x.GetPrim()).GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
            UsdGeom.Gprim(axis_y.GetPrim()).GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])
        except Exception:
            pass


def spawn_ground_plane(stage) -> None:
    """Version-tolerant ground plane spawning for Isaac Sim / Isaac Lab."""
    import omni.physx.scripts.physicsUtils as physicsUtils

    fn = physicsUtils.add_ground_plane
    sig = inspect.signature(fn)

    plane_path = "/World/GroundPlane"
    color = (0.16, 0.16, 0.16)
    kwargs = {}
    for name in sig.parameters.keys():
        low = name.lower()
        if "stage" in low:
            kwargs[name] = stage
        elif "planepath" in low or (("path" in low or "prim" in low) and "plane" in low):
            kwargs[name] = plane_path
        elif "color" in low or "colour" in low:
            kwargs[name] = color
        elif "axis" in low:
            kwargs[name] = "Z"
        elif "size" in low or "halfsize" in low:
            kwargs[name] = 50.0
        elif "position" in low:
            kwargs[name] = (0.0, 0.0, 0.0)
        elif "normal" in low:
            kwargs[name] = (0.0, 0.0, 1.0)
        elif ("static" in low and "friction" in low) or low == "staticfriction":
            kwargs[name] = 2.0
        elif ("dynamic" in low and "friction" in low) or low == "dynamicfriction":
            kwargs[name] = 1.8
        elif "restitution" in low:
            kwargs[name] = 0.0

    if "planePath" in sig.parameters and "planePath" not in kwargs:
        kwargs["planePath"] = plane_path
    if "color" in sig.parameters and "color" not in kwargs:
        kwargs["color"] = color

    fn(**kwargs)


def spawn_light(stage) -> None:
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(0.5)


# =============================================================================
# Track point discovery and world/body coordinate extraction
# =============================================================================

def get_body_names(robot: Articulation) -> List[str]:
    names = getattr(robot.data, "body_names", None)
    if names is None:
        raise AttributeError("robot.data.body_names is unavailable; cannot discover articulation bodies.")
    return list(names)


def get_body_pos_w(robot: Articulation) -> torch.Tensor:
    """Version-tolerant access to articulation body positions in world coordinates."""
    data = robot.data
    for attr in ("body_pos_w", "body_link_pos_w"):
        value = getattr(data, attr, None)
        if value is not None:
            return value[0]
    for attr in ("body_state_w", "body_link_state_w"):
        value = getattr(data, attr, None)
        if value is not None:
            return value[0, :, 0:3]
    raise AttributeError("No supported body world-position tensor found on robot.data.")


def _score_leg_track_name(leg: str, low: str) -> float:
    """Heuristic scorer for distal track-point discovery.

    We strongly prefer explicit toe/toe_proxy/foot-like names.
    Knee links are allowed only as a fallback.
    """
    score = -1e9
    if not any(tok in low for tok in LEG_NAME_TOKENS[leg]):
        return score

    score = 0.0
    for tok, pts in DISTAL_BODY_SCORE:
        if tok in low:
            score = max(score, float(pts))

    if any(tok in low for tok in ["toeproxy", "toe_proxy", "toe", "foot", "paw", "tip", "contact"]):
        score += 30.0
    if "visual" in low:
        score -= 4.0
    if "collision" in low or "collider" in low or "colliders" in low:
        score += 6.0
    if "mesh" in low:
        score += 2.0
    if "knee_link" in low or low.endswith("knee_link"):
        score -= 18.0
    elif "knee" in low:
        score -= 10.0

    if leg in FRONT_LEGS and "front" in low:
        score += 8.0
    if leg in BACK_LEGS and ("back" in low or "rear" in low):
        score += 8.0
    if "left" in low and leg in ("LF", "LB"):
        score += 4.0
    if "right" in low and leg in ("RF", "RB"):
        score += 4.0
    return score


def discover_track_prims(stage, robot_prim_path: str) -> Dict[str, str]:
    """Preferred route: discover distal foot-like prims directly from the USD stage."""
    robot_prefix = robot_prim_path.rstrip("/")
    all_prims = [prim for prim in stage.TraverseAll()]

    leg_to_path: Dict[str, str] = {}
    for leg in LEGS:
        best_score = -1e9
        best_path: Optional[str] = None

        for prim in all_prims:
            path_str = str(prim.GetPath())
            if not path_str.startswith(robot_prefix):
                continue
            if not prim.IsActive() or not prim.IsValid():
                continue

            low = path_str.lower()
            score = _score_leg_track_name(leg, low)
            if score > best_score:
                best_score = score
                best_path = path_str

        if best_path is None or best_score < -1e8:
            raise RuntimeError(f"Could not find any track prim under {robot_prim_path} for {leg}.")
        leg_to_path[leg] = best_path

    return leg_to_path


def discover_track_body_indices(body_names: List[str]) -> Dict[str, int]:
    """Fallback route only: articulation body names may stop at knee_link."""
    leg_to_idx: Dict[str, int] = {}
    for leg in LEGS:
        best_idx: Optional[int] = None
        best_score = -1e9
        for i, name in enumerate(body_names):
            score = _score_leg_track_name(leg, name.lower())
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None or best_score < -1e8:
            raise RuntimeError(
                f"Could not find a fallback articulation body for {leg}. Available body names: {body_names}"
            )
        leg_to_idx[leg] = best_idx
    return leg_to_idx


def get_track_points_world(
    robot: Articulation,
    stage,
    leg_to_track_prim: Optional[Dict[str, str]],
    leg_to_track_body: Optional[Dict[str, int]],
) -> Dict[str, Tuple[float, float, float, str]]:
    """Return actual tracked point positions for all four legs.

    Output dict maps each leg to:
        (x_w, y_w, z_w, source_name)
    """
    out: Dict[str, Tuple[float, float, float, str]] = {}

    if leg_to_track_prim is not None:
        xfc = UsdGeom.XformCache()
        for leg, path_str in leg_to_track_prim.items():
            prim = stage.GetPrimAtPath(path_str)
            if prim and prim.IsValid():
                M = xfc.GetLocalToWorldTransform(prim)
                p = M.ExtractTranslation()
                out[leg] = (float(p[0]), float(p[1]), float(p[2]), path_str)
        if len(out) == len(LEGS):
            return out

    body_pos_w = get_body_pos_w(robot)
    body_names = get_body_names(robot)
    if leg_to_track_body is None:
        leg_to_track_body = discover_track_body_indices(body_names)
    for leg, idx in leg_to_track_body.items():
        p = body_pos_w[idx].detach().cpu().numpy()
        out[leg] = (float(p[0]), float(p[1]), float(p[2]), body_names[idx])
    return out


# =============================================================================
# CSV field layout and export helpers
# =============================================================================

def build_metrics_fieldnames() -> List[str]:
    fieldnames = [
        "t", "stage", "root_x", "root_y", "root_z", "roll", "pitch", "yaw", "vx", "vy", "vz"
    ]
    for leg in LEGS:
        fieldnames += [
            f"{leg}_phase",
            f"{leg}_q_sh_target",
            f"{leg}_q_kn_target",
            f"{leg}_sh_scalar",
            f"{leg}_kn_scalar",
            f"{leg}_phase_name",
            f"{leg}_track_body_name",
            f"{leg}_foot_x_w",
            f"{leg}_foot_y_w",
            f"{leg}_foot_z_w",
            f"{leg}_foot_x_b",
            f"{leg}_foot_y_b",
            f"{leg}_foot_z_b",
        ]
    return fieldnames


def build_foot_traj_fieldnames() -> List[str]:
    return [
        "t",
        "stage",
        "leg",
        "phase",
        "phase_name",
        "track_body_name",
        "foot_x_w",
        "foot_y_w",
        "foot_z_w",
        "foot_x_b",
        "foot_y_b",
        "foot_z_b",
        "root_yaw",
    ]


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_run_info(
    path: Path,
    cfg,
    q_support: torch.Tensor,
    leg_to_sh: Dict[str, int],
    leg_to_kn: Dict[str, int],
    leg_to_track_prim: Optional[Dict[str, str]],
    leg_to_track_body: Optional[Dict[str, int]],
    robot: Articulation,
    metrics_csv: Path,
    foot_traj_csv: Path,
    exec_log_txt: Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("bittle_baseline_clean_track_structured\n")
        f.write(f"usd_path = {cfg.spawn.usd_path}\n")
        f.write(f"prim_path = {cfg.prim_path}\n")
        f.write(f"root_z0 = {float(cfg.init_state.pos[2])}\n")
        f.write(f"dt = {CFG.sim.dt}\n")
        f.write(f"device = {CFG.sim.device}\n")

        f.write("\n[validated_sign_system]\n")
        f.write(f"SH_SIGN = {SH_SIGN}\n")
        f.write(f"KN_SIGN = {KN_SIGN}\n")

        f.write("\n[support_offsets_from_cfg]\n")
        for k, v in vars(CFG.support).items():
            f.write(f"{k} = {v}\n")

        f.write("\n[gait_parameters]\n")
        for k, v in vars(CFG.gait).items():
            f.write(f"{k} = {v}\n")

        f.write("\n[q_support_targets]\n")
        for leg in LEGS:
            f.write(
                f"{leg}: sh={float(q_support[0, leg_to_sh[leg]]):+.6f} | "
                f"kn={float(q_support[0, leg_to_kn[leg]]):+.6f}\n"
            )

        f.write("\n[tracked_points]\n")
        if leg_to_track_prim is not None:
            f.write("method = stage_prim\n")
            for leg in LEGS:
                f.write(f"{leg}: {leg_to_track_prim[leg]}\n")
        else:
            f.write("method = articulation_body_fallback\n")
            body_names = get_body_names(robot)
            for leg in LEGS:
                f.write(f"{leg}: {body_names[leg_to_track_body[leg]]}\n")

        f.write("\n[files]\n")
        f.write(f"metrics_csv = {metrics_csv}\n")
        f.write(f"foot_traj_csv = {foot_traj_csv}\n")
        f.write(f"exec_log = {exec_log_txt}\n")

        f.write("\n[RL_handoff_note]\n")
        f.write("Recommended next stage: residual joint deltas on top of this fixed-support baseline.\n")


# =============================================================================
# Main simulation routine
# =============================================================================

def main() -> None:
    ensure_output_dir()

    # -------------------------------------------------------------------------
    # Simulation and stage setup
    # -------------------------------------------------------------------------
    cfg = BITTLE_CFG.copy()
    cfg.init_state.pos = (0.0, 0.0, CFG.sim.root_z0)

    sim_cfg = sim_utils.SimulationCfg(dt=CFG.sim.dt, device=CFG.sim.device)
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    sim.set_camera_view(list(CFG.sim.camera_eye), list(CFG.sim.camera_target))

    spawn_ground_plane(stage)
    spawn_floor_reference(stage)
    spawn_light(stage)

    robot = Articulation(cfg=cfg)

    # -------------------------------------------------------------------------
    # Reset robot cleanly and restore default root / joint state
    # -------------------------------------------------------------------------
    sim.reset()
    sim.step()
    robot.update(CFG.sim.dt)

    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    default_root[:, 0] = 0.0
    default_root[:, 1] = 0.0
    default_root[:, 2] = CFG.sim.root_z0

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.write_data_to_sim()
    for _ in range(2):
        sim.step()
        robot.update(CFG.sim.dt)
    robot.reset()

    # -------------------------------------------------------------------------
    # Build posture references
    # -------------------------------------------------------------------------
    joint_names = list(robot.data.joint_names)
    jmap, leg_to_sh, leg_to_kn = build_joint_maps(joint_names)

    q_stand = build_stand_pose_from_cfg(jmap, robot.data.default_joint_pos)
    q_support = apply_support_posture(q_stand, leg_to_sh, leg_to_kn)

    # -------------------------------------------------------------------------
    # Discover true foot tracking points
    # -------------------------------------------------------------------------
    leg_to_track_prim: Optional[Dict[str, str]] = None
    leg_to_track_body: Optional[Dict[str, int]] = None

    if CFG.tracking.prefer_stage_prim:
        try:
            leg_to_track_prim = discover_track_prims(stage, cfg.prim_path)
        except Exception:
            body_names = get_body_names(robot)
            leg_to_track_body = discover_track_body_indices(body_names)
    else:
        body_names = get_body_names(robot)
        leg_to_track_body = discover_track_body_indices(body_names)

    # -------------------------------------------------------------------------
    # Output paths and row buffers
    # -------------------------------------------------------------------------
    metrics_csv = TEM_DOC / f"{CFG.export.file_stem}_metrics.csv"
    foot_traj_csv = TEM_DOC / f"{CFG.export.file_stem}_foot_traj.csv"
    run_info_txt = TEM_DOC / f"{CFG.export.file_stem}_run_info.txt"
    exec_log_txt = TEM_DOC / f"{CFG.export.file_stem}_exec.log"

    metrics_fieldnames = build_metrics_fieldnames()
    foot_traj_fieldnames = build_foot_traj_fieldnames()

    metrics_rows: List[Dict[str, object]] = []
    foot_traj_rows: List[Dict[str, object]] = []

    t = 0.0
    step_counter = 0

    def step_with_target(q_cmd: torch.Tensor) -> None:
        nonlocal t, step_counter
        robot.set_joint_position_target(q_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(CFG.sim.dt)
        t += CFG.sim.dt
        step_counter += 1

    def log_step(
        stage_name: str,
        q_targets: Dict[str, Tuple[float, float]],
        sh_scalars: Dict[str, float],
        kn_scalars: Dict[str, float],
        phase_names: Dict[str, str],
    ) -> None:
        """Log root state, per-leg command state, and true tracked foot positions."""
        root_pos_np = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat_np = robot.data.root_quat_w[0].detach().cpu().numpy()
        root_lin_vel_np = robot.data.root_lin_vel_w[0].detach().cpu().numpy()

        root_pos = (float(root_pos_np[0]), float(root_pos_np[1]), float(root_pos_np[2]))
        root_quat = (
            float(root_quat_np[0]),
            float(root_quat_np[1]),
            float(root_quat_np[2]),
            float(root_quat_np[3]),
        )
        roll, pitch, yaw = quat_to_euler_xyz(*root_quat)

        row: Dict[str, object] = {
            "t": t,
            "stage": stage_name,
            "root_x": root_pos[0],
            "root_y": root_pos[1],
            "root_z": root_pos[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": float(root_lin_vel_np[0]),
            "vy": float(root_lin_vel_np[1]),
            "vz": float(root_lin_vel_np[2]),
        }

        gait_t = max(0.0, t - CFG.sim.warmup_time)
        track_points = get_track_points_world(robot, stage, leg_to_track_prim, leg_to_track_body)

        for leg in LEGS:
            phase_val = leg_phase(gait_t, leg)
            foot_w = (track_points[leg][0], track_points[leg][1], track_points[leg][2])
            foot_b = world_to_body(root_pos, root_quat, foot_w)
            track_name = track_points[leg][3]

            row[f"{leg}_phase"] = phase_val
            row[f"{leg}_q_sh_target"] = q_targets[leg][0]
            row[f"{leg}_q_kn_target"] = q_targets[leg][1]
            row[f"{leg}_sh_scalar"] = sh_scalars[leg]
            row[f"{leg}_kn_scalar"] = kn_scalars[leg]
            row[f"{leg}_phase_name"] = phase_names[leg]
            row[f"{leg}_track_body_name"] = track_name
            row[f"{leg}_foot_x_w"] = foot_w[0]
            row[f"{leg}_foot_y_w"] = foot_w[1]
            row[f"{leg}_foot_z_w"] = foot_w[2]
            row[f"{leg}_foot_x_b"] = foot_b[0]
            row[f"{leg}_foot_y_b"] = foot_b[1]
            row[f"{leg}_foot_z_b"] = foot_b[2]

            foot_traj_rows.append(
                {
                    "t": t,
                    "stage": stage_name,
                    "leg": leg,
                    "phase": phase_val,
                    "phase_name": phase_names[leg],
                    "track_body_name": track_name,
                    "foot_x_w": foot_w[0],
                    "foot_y_w": foot_w[1],
                    "foot_z_w": foot_w[2],
                    "foot_x_b": foot_b[0],
                    "foot_y_b": foot_b[1],
                    "foot_z_b": foot_b[2],
                    "root_yaw": yaw,
                }
            )

        metrics_rows.append(row)

        if step_counter % max(1, CFG.sim.print_every_n_steps) == 0:
            print(
                f"[{t:6.3f}s | {stage_name:18s}] "
                f"p=({row['root_x']:+.3f},{row['root_y']:+.3f},{row['root_z']:+.3f}) | "
                f"rpy=({row['roll']:+.3f},{row['pitch']:+.3f},{row['yaw']:+.3f}) | "
                f"v=({row['vx']:+.3f},{row['vy']:+.3f},{row['vz']:+.3f}) | "
                f"LF_yz_b=({row['LF_foot_y_b']:+.3f},{row['LF_foot_z_b']:+.3f}) via {row['LF_track_body_name']}"
            )

    exec_lines = [
        "START",
        f"usd_path = {cfg.spawn.usd_path}",
        f"prim_path = {cfg.prim_path}",
        f"device = {CFG.sim.device}",
    ]

    try:
        # ---------------------------------------------------------------------
        # Console summary at run start
        # ---------------------------------------------------------------------
        print("\n=== bittle_baseline_clean_track_structured ===")
        print(f"[usd_path] {cfg.spawn.usd_path}")
        print(f"[prim_path] {cfg.prim_path}")
        print(f"[root_z0] {float(cfg.init_state.pos[2]):.3f}")
        print(f"[device] {CFG.sim.device}")
        print("[validated sign system]")
        print(f"  SH_SIGN = {SH_SIGN}")
        print(f"  KN_SIGN = {KN_SIGN}")
        print("[support offsets from cfg]")
        print(
            f"  front_sh_y={CFG.support.front_sh_y:+.3f} back_sh_y={CFG.support.back_sh_y:+.3f} | "
            f"front_kn_crouch={CFG.support.front_kn_crouch:+.3f} back_kn_crouch={CFG.support.back_kn_crouch:+.3f}"
        )
        print("[gait parameters]")
        for k, v in vars(CFG.gait).items():
            print(f"  {k} = {v}")
        print("[tracked points]")
        if leg_to_track_prim is not None:
            print("  method = stage_prim")
            for leg in LEGS:
                print(f"  {leg}: {leg_to_track_prim[leg]}")
        else:
            print("  method = articulation_body_fallback")
            body_names = get_body_names(robot)
            for leg in LEGS:
                print(f"  {leg}: {body_names[leg_to_track_body[leg]]}")
        print("[q_support targets]")
        for leg in LEGS:
            print(
                f"  {leg}: sh={float(q_support[0, leg_to_sh[leg]]):+.3f} | "
                f"kn={float(q_support[0, leg_to_kn[leg]]):+.3f}"
            )

        # ---------------------------------------------------------------------
        # Warmup: hold q_support to settle into the support working point
        # ---------------------------------------------------------------------
        warmup_steps = int(round(CFG.sim.warmup_time / CFG.sim.dt))
        for _ in range(warmup_steps):
            step_with_target(q_support)
            zero_targets = {
                leg: (float(q_support[0, leg_to_sh[leg]]), float(q_support[0, leg_to_kn[leg]]))
                for leg in LEGS
            }
            zero_scalars = {leg: 0.0 for leg in LEGS}
            zero_phase_names = {leg: "warmup" for leg in LEGS}
            log_step("warmup_qsupport", zero_targets, zero_scalars, zero_scalars, zero_phase_names)

        # ---------------------------------------------------------------------
        # Main gait loop
        # ---------------------------------------------------------------------
        run_steps = int(round(CFG.sim.run_time / CFG.sim.dt))
        for _ in range(run_steps):
            gait_t = max(0.0, t - CFG.sim.warmup_time)
            alpha = min(1.0, gait_t / CFG.gait.ramp_time) if CFG.gait.ramp_time > 1e-9 else 1.0

            q_cmd = q_support.clone()
            q_targets: Dict[str, Tuple[float, float]] = {}
            sh_scalars: Dict[str, float] = {}
            kn_scalars: Dict[str, float] = {}
            phase_names: Dict[str, str] = {}

            for leg in LEGS:
                u = leg_phase(gait_t, leg)
                sh_scalar, kn_scalar, phase_name = leg_template_scalars(u)

                q_cmd[0, leg_to_sh[leg]] += alpha * SH_SIGN[leg] * sh_scalar
                q_cmd[0, leg_to_kn[leg]] += alpha * KN_SIGN[leg] * kn_scalar

                sh_scalars[leg] = sh_scalar
                kn_scalars[leg] = kn_scalar
                phase_names[leg] = phase_name

            q_cmd = apply_joint_limits(q_cmd, leg_to_sh, leg_to_kn)

            for leg in LEGS:
                q_targets[leg] = (
                    float(q_cmd[0, leg_to_sh[leg]]),
                    float(q_cmd[0, leg_to_kn[leg]]),
                )

            step_with_target(q_cmd)
            log_step("baseline_clean_track", q_targets, sh_scalars, kn_scalars, phase_names)

        # ---------------------------------------------------------------------
        # Export results
        # ---------------------------------------------------------------------
        write_csv(metrics_csv, metrics_fieldnames, metrics_rows)
        write_csv(foot_traj_csv, foot_traj_fieldnames, foot_traj_rows)
        write_run_info(
            run_info_txt,
            cfg,
            q_support,
            leg_to_sh,
            leg_to_kn,
            leg_to_track_prim,
            leg_to_track_body,
            robot,
            metrics_csv,
            foot_traj_csv,
            exec_log_txt,
        )

        exec_lines.append("SUCCESS")
        exec_lines.append(f"metrics_csv = {metrics_csv}")
        exec_lines.append(f"foot_traj_csv = {foot_traj_csv}")
        exec_lines.append(f"run_info_txt = {run_info_txt}")
        exec_lines.append(f"exec_log = {exec_log_txt}")
        exec_log_txt.write_text("\n".join(exec_lines), encoding="utf-8")

        print(f"\nSaved: {metrics_csv}")
        print(f"Saved: {foot_traj_csv}")
        print(f"Saved: {run_info_txt}")
        print(f"Saved: {exec_log_txt}")

    except Exception as e:
        exec_lines.append("FAILED")
        exec_lines.append(repr(e))
        exec_lines.append(traceback.format_exc())
        exec_log_txt.write_text("\n".join(exec_lines), encoding="utf-8")
        print(f"\nFAILED. See: {exec_log_txt}")
        raise


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
