from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo root = .../quadraped_RL_CPG
REPO_ROOT = Path(__file__).resolve().parents[1]

# Symmetry-fixed USD with rebuilt symmetric colliders.
BITTLE_USD_PATH = (
    REPO_ROOT / "bittle" / "configuration" / "bittle_shoulderpatch_toeproxy.usd"
).as_posix()

# -----------------------------------------------------------------------------
# Joint pose definitions
# -----------------------------------------------------------------------------
# Original stand pose used earlier during stand/pose debugging.
# Keep this around because it is still useful as a neutral reference.
STAND_JOINT_POS = {
    "left_front_shoulder_joint": 0.35,
    "right_front_shoulder_joint": -0.35,
    "left_back_shoulder_joint": 0.35,
    "right_back_shoulder_joint": -0.35,
    "left_front_knee_joint": 0.70,
    "right_front_knee_joint": -0.70,
    "left_back_knee_joint": 0.70,
    "right_back_knee_joint": -0.70,
}

# Support posture offsets that had been applied in the stance-test scripts.
# The goal here is maintenance simplicity: cfg now starts directly from the
# support-calibrated posture, so later scripts do not need to re-apply the same
# offsets in multiple places.
SUPPORT_POSTURE_OFFSETS = {
    "front_sh_y": -0.45,
    "back_sh_y": -0.45,
    "front_kn_crouch": -0.40,
    "back_kn_crouch": -0.40,
}

# Side sign convention already validated in your Bittle scripts.
SIDE_SIGN = {
    "LF": -1.0,
    "RF": +1.0,
    "LB": -1.0,
    "RB": +1.0,
}

# Support-calibrated default pose.
# Derived from STAND_JOINT_POS + SUPPORT_POSTURE_OFFSETS:
#   shoulders: ±0.35 -> ±0.80
#   knees    : ±0.70 -> ±0.30
SUPPORT_JOINT_POS = {
    "left_front_shoulder_joint": STAND_JOINT_POS["left_front_shoulder_joint"] + SIDE_SIGN["LF"] * SUPPORT_POSTURE_OFFSETS["front_sh_y"],
    "right_front_shoulder_joint": STAND_JOINT_POS["right_front_shoulder_joint"] + SIDE_SIGN["RF"] * SUPPORT_POSTURE_OFFSETS["front_sh_y"],
    "left_back_shoulder_joint": STAND_JOINT_POS["left_back_shoulder_joint"] + SIDE_SIGN["LB"] * SUPPORT_POSTURE_OFFSETS["back_sh_y"],
    "right_back_shoulder_joint": STAND_JOINT_POS["right_back_shoulder_joint"] + SIDE_SIGN["RB"] * SUPPORT_POSTURE_OFFSETS["back_sh_y"],
    "left_front_knee_joint": STAND_JOINT_POS["left_front_knee_joint"] - SIDE_SIGN["LF"] * SUPPORT_POSTURE_OFFSETS["front_kn_crouch"],
    "right_front_knee_joint": STAND_JOINT_POS["right_front_knee_joint"] - SIDE_SIGN["RF"] * SUPPORT_POSTURE_OFFSETS["front_kn_crouch"],
    "left_back_knee_joint": STAND_JOINT_POS["left_back_knee_joint"] - SIDE_SIGN["LB"] * SUPPORT_POSTURE_OFFSETS["back_kn_crouch"],
    "right_back_knee_joint": STAND_JOINT_POS["right_back_knee_joint"] - SIDE_SIGN["RB"] * SUPPORT_POSTURE_OFFSETS["back_kn_crouch"],
}

# Single switch for later maintenance.
# Change this to STAND_JOINT_POS if you want to go back to the old neutral pose.
DEFAULT_INIT_JOINT_POS = SUPPORT_JOINT_POS

BITTLE_CFG = ArticulationCfg(
    prim_path="/World/bittle",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BITTLE_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=50.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=0.3,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # NOTE: for your current scaled USD, z=1.0 is intentional.
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=6.0,
            stiffness=350.0,
            damping=32.0,
        )
    },
)
