from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

LEGS = ("LF", "RF", "LB", "RB")
GROUP_A = ("LF", "RB")
GROUP_B = ("RF", "LB")
FRONT_LEGS = ("LF", "RF")
BACK_LEGS = ("LB", "RB")

# -----------------------------------------------------------------------------
# Fixed truths that must stay explicit
# -----------------------------------------------------------------------------
# These names are written redundantly on purpose.
# The goal is that if the surrounding project memory is gone, the next reader can
# still reconstruct the intended Bittle control interface from this file alone.
#
# Confirmed joint interface names:
#   LF shoulder -> left_front_shoulder_joint
#   RF shoulder -> right_front_shoulder_joint
#   LB shoulder -> left_back_shoulder_joint
#   RB shoulder -> right_back_shoulder_joint
#   LF knee     -> left_front_knee_joint
#   RF knee     -> right_front_knee_joint
#   LB knee     -> left_back_knee_joint
#   RB knee     -> right_back_knee_joint
#
# Confirmed foot-reference frame names in the patched USD:
#   LF -> left_front_knee_link/toe_proxy
#   RF -> right_front_knee_link/toe_proxy
#   LB -> left_back_knee_link/toe_proxy
#   RB -> right_back_knee_link/toe_proxy
#
# Control-policy implication:
# - This file generates only the joint-space baseline target.
# - Foot/toe frame names are written here as documentation so the RL env layer
#   can align observation code with the same asset truths.

# Fixed validated sign truth. Keep explicit for sim-to-real consistency.
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

# Documentation-only foot frame names copied from the targeted USD inspection.
LEG_TO_TOE_FRAME_NAME = {
    "LF": "left_front_knee_link/toe_proxy",
    "RF": "right_front_knee_link/toe_proxy",
    "LB": "left_back_knee_link/toe_proxy",
    "RB": "right_back_knee_link/toe_proxy",
}


@dataclass
class SupportParams:
    front_sh_y: float = -0.45
    back_sh_y: float = -0.45
    front_kn_crouch: float = -0.40
    back_kn_crouch: float = -0.40


@dataclass
class GaitParams:
    freq_hz: float = 1.0
    swing_ratio: float = 0.36
    shoulder_amp: float = 0.14
    knee_lift_amp: float = 0.26
    touchdown_buffer_ratio: float = 0.24
    ramp_time: float = 2.5

    liftoff_ratio: float = 0.30
    predown_ratio: float = 0.22
    liftoff_shoulder_frac: float = 0.18
    transfer_knee_floor_frac: float = 0.74
    touchdown_knee_hold_frac: float = 0.16
    touchdown_shoulder_end_frac: float = 0.92


@dataclass
class JointLimitParams:
    shoulder_lim: Tuple[float, float] = (-1.15, 1.15)
    knee_lim: Tuple[float, float] = (-1.55, 1.55)


DEFAULT_SUPPORT = SupportParams()
DEFAULT_GAIT = GaitParams()
DEFAULT_LIMITS = JointLimitParams()


def clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def smoothstep01(x: float) -> float:
    x = clampf(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_joint_maps(joint_names: Iterable[str]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    leg_to_sh = {leg: name_to_idx[LEG_TO_SHOULDER_NAME[leg]] for leg in LEGS}
    leg_to_kn = {leg: name_to_idx[LEG_TO_KNEE_NAME[leg]] for leg in LEGS}
    return name_to_idx, leg_to_sh, leg_to_kn


def apply_support_posture(
    q_stand,
    leg_to_sh: Dict[str, int],
    leg_to_kn: Dict[str, int],
    support: SupportParams = DEFAULT_SUPPORT,
    limits: JointLimitParams = DEFAULT_LIMITS,
):
    q = q_stand.clone()
    for leg in FRONT_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * support.front_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * support.front_kn_crouch
    for leg in BACK_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * support.back_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * support.back_kn_crouch
    return apply_joint_limits(q, leg_to_sh, leg_to_kn, limits)


def apply_joint_limits(q, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int], limits: JointLimitParams = DEFAULT_LIMITS):
    q = q.clone()
    for leg in LEGS:
        q[0, leg_to_sh[leg]] = clampf(float(q[0, leg_to_sh[leg]]), *limits.shoulder_lim)
        q[0, leg_to_kn[leg]] = clampf(float(q[0, leg_to_kn[leg]]), *limits.knee_lim)
    return q


def leg_phase(gait_t: float, leg: str, gait: GaitParams = DEFAULT_GAIT) -> float:
    phase0 = 0.0 if leg in GROUP_A else 0.5
    return (phase0 + gait.freq_hz * gait_t) % 1.0


def leg_template_scalars(u: float, gait: GaitParams = DEFAULT_GAIT) -> Tuple[float, float, str]:
    sr = clampf(gait.swing_ratio, 0.10, 0.90)

    if u < sr:
        s = u / sr
        liftoff_end = clampf(gait.liftoff_ratio, 0.05, 0.80)
        predown_start = clampf(1.0 - gait.predown_ratio, liftoff_end + 0.05, 0.98)

        if s < liftoff_end:
            a = s / liftoff_end
            sh = gait.shoulder_amp * gait.liftoff_shoulder_frac * smoothstep01(a)
            kn = gait.knee_lift_amp * 0.90 * smoothstep01(a)
            return sh, kn, "swing_liftoff"

        if s < predown_start:
            a = (s - liftoff_end) / max(1e-9, predown_start - liftoff_end)
            sh = gait.shoulder_amp * (
                gait.liftoff_shoulder_frac
                + (1.0 - gait.liftoff_shoulder_frac) * smoothstep01(a)
            )
            knee_floor = clampf(gait.transfer_knee_floor_frac, 0.35, 1.00)
            kn = gait.knee_lift_amp * (knee_floor + (1.0 - knee_floor) * math.sin(0.5 * math.pi * a))
            return sh, kn, "swing_transfer"

        a = (s - predown_start) / max(1e-9, 1.0 - predown_start)
        sh = gait.shoulder_amp
        kn = gait.knee_lift_amp * (
            gait.touchdown_knee_hold_frac
            + (1.0 - gait.touchdown_knee_hold_frac) * (1.0 - smoothstep01(a))
        )
        return sh, kn, "swing_predown"

    w = (u - sr) / max(1e-9, 1.0 - sr)
    td = clampf(gait.touchdown_buffer_ratio, 0.0, 0.80)

    if w < td:
        a = 0.0 if td < 1e-9 else (w / td)
        sh_end_frac = clampf(gait.touchdown_shoulder_end_frac, 0.70, 1.00)
        sh = gait.shoulder_amp * (1.0 - (1.0 - sh_end_frac) * smoothstep01(a))
        kn = gait.knee_lift_amp * gait.touchdown_knee_hold_frac * (1.0 - smoothstep01(a))
        return sh, kn, "touchdown_buffer"

    v = (w - td) / max(1e-9, 1.0 - td)
    sh_end_frac = clampf(gait.touchdown_shoulder_end_frac, 0.70, 1.00)
    sh = gait.shoulder_amp * sh_end_frac * (1.0 - smoothstep01(v))
    return sh, 0.0, "stance_sweep"


def compose_cpg_target(
    q_support,
    leg_to_sh: Dict[str, int],
    leg_to_kn: Dict[str, int],
    gait_t: float,
    alpha: float,
    gait: GaitParams = DEFAULT_GAIT,
    limits: JointLimitParams = DEFAULT_LIMITS,
):
    q_cmd = q_support.clone()
    info = {}

    for leg in LEGS:
        u = leg_phase(gait_t, leg, gait)
        sh_scalar, kn_scalar, phase_name = leg_template_scalars(u, gait)

        q_cmd[0, leg_to_sh[leg]] += alpha * SH_SIGN[leg] * sh_scalar
        q_cmd[0, leg_to_kn[leg]] += alpha * KN_SIGN[leg] * kn_scalar

        info[leg] = {
            "phase": u,
            "phase_name": phase_name,
            "sh_scalar": sh_scalar,
            "kn_scalar": kn_scalar,
        }

    q_cmd = apply_joint_limits(q_cmd, leg_to_sh, leg_to_kn, limits)
    return q_cmd, info


def apply_residual_delta(
    q_cmd,
    residual,
    leg_to_sh: Dict[str, int],
    leg_to_kn: Dict[str, int],
    action_scale_sh: float = 0.10,
    action_scale_kn: float = 0.10,
    limits: JointLimitParams = DEFAULT_LIMITS,
):
    """Apply residual RL action on top of the baseline command.

    residual shape convention:
        [LF_sh, LF_kn, RF_sh, RF_kn, LB_sh, LB_kn, RB_sh, RB_kn]

    This ordering intentionally matches the fixed leg naming written above.
    Keep it explicit. Do not silently reorder actions from inferred body order.
    """
    q_out = q_cmd.clone()
    order = ("LF", "RF", "LB", "RB")
    for i, leg in enumerate(order):
        q_out[0, leg_to_sh[leg]] += action_scale_sh * float(residual[2 * i + 0])
        q_out[0, leg_to_kn[leg]] += action_scale_kn * float(residual[2 * i + 1])
    return apply_joint_limits(q_out, leg_to_sh, leg_to_kn, limits)
