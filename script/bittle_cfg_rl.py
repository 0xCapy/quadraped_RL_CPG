from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, ImuCfg

# Repo root = .../quadraped_RL_CPG
REPO_ROOT = Path(__file__).resolve().parents[1]

# Symmetry-fixed USD with rebuilt symmetric colliders.
BITTLE_USD_PATH = (
    REPO_ROOT / "bittle" / "configuration" / "bittle_shoulderpatch_toeproxy.usd"
).as_posix()

# -----------------------------------------------------------------------------
# Asset-interface truths confirmed by targeted USD inspection on 2026-04-10
# -----------------------------------------------------------------------------
# These comments are intentionally concrete and redundant.
# They are here so that even if all chat history is gone, the next person can
# still recover the exact training-side interface choices directly from this file.
#
# Confirmed authored root in the USD:
#   /bittle
#
# Confirmed runtime root used by Isaac Lab / scene spawning:
#   /World/bittle
#   In vectorized RL scenes, write this as:
#   {ENV_REGEX_NS}/bittle
#
# Confirmed best per-leg distal tracking frames:
#   LF -> /bittle/left_front_knee_link/toe_proxy
#   RF -> /bittle/right_front_knee_link/toe_proxy
#   LB -> /bittle/left_back_knee_link/toe_proxy
#   RB -> /bittle/right_back_knee_link/toe_proxy
#
# Confirmed explicit IMU-like prim:
#   /bittle/base_frame_link/mainboard_link/imu_link
#
# Important sensor-design implication from inspection:
# - toe_proxy prims are Xform frames only. They are good for foot-frame tracking,
#   FrameTransformer targets, and logging.
# - toe_proxy prims are NOT confirmed rigid bodies and should NOT be the first
#   choice for contact sensing.
# - The rigid-body-bearing leg links found by inspection are the shoulder_link
#   and knee_link Xforms. So first-pass RL contact sensing should attach to the
#   physical leg links (especially the knee links), not to toe_proxy.
#
# Collision note:
# - The targeted offline inspection saw named collisions groups such as
#     /bittle/left_front_knee_link/collisions
#   but did not surface authored CollisionAPI on the kept prim list.
# - Therefore this file keeps contact sensor hookup conservative:
#   use the rigid knee links first, and only move lower if a later runtime check
#   proves that deeper collider children expose contact reporting cleanly.

# Authored USD paths (for documentation / debugging only).
AUTHORED_ROOT_PATH = "/bittle"
AUTHORED_BASE_FRAME_PATH = "/bittle/base_frame_link"
AUTHORED_IMU_PATH = "/bittle/base_frame_link/mainboard_link/imu_link"
AUTHORED_TOE_PATHS = {
    "LF": "/bittle/left_front_knee_link/toe_proxy",
    "RF": "/bittle/right_front_knee_link/toe_proxy",
    "LB": "/bittle/left_back_knee_link/toe_proxy",
    "RB": "/bittle/right_back_knee_link/toe_proxy",
}

# Runtime scene paths used by Isaac Lab envs.
RUNTIME_ROOT_PATH = "{ENV_REGEX_NS}/bittle"
RUNTIME_BASE_FRAME_PATH = "/World/envs/env_.*/bittle/base_frame_link"
RUNTIME_IMU_PATH = "{ENV_REGEX_NS}/bittle/base_frame_link/mainboard_link/imu_link"
RUNTIME_TOE_PATHS = {
    "LF": "{ENV_REGEX_NS}/bittle/left_front_knee_link/toe_proxy",
    "RF": "{ENV_REGEX_NS}/bittle/right_front_knee_link/toe_proxy",
    "LB": "{ENV_REGEX_NS}/bittle/left_back_knee_link/toe_proxy",
    "RB": "{ENV_REGEX_NS}/bittle/right_back_knee_link/toe_proxy",
}

# First-pass physical contact bodies for RL.
# These are intentionally the knee links rather than toe_proxy.

RUNTIME_CONTACT_BODY_PATHS = {
    "LF": "/World/envs/env_.*/bittle/left_front_knee_link",
    "RF": "/World/envs/env_.*/bittle/right_front_knee_link",
    "LB": "/World/envs/env_.*/bittle/left_back_knee_link",
    "RB": "/World/envs/env_.*/bittle/right_back_knee_link",
}
# -----------------------------------------------------------------------------
# Joint pose definitions and validated fixed truths
# -----------------------------------------------------------------------------
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

SUPPORT_POSTURE_OFFSETS = {
    "front_sh_y": -0.45,
    "back_sh_y": -0.45,
    "front_kn_crouch": -0.40,
    "back_kn_crouch": -0.40,
}

# Validated side sign truth. Keep explicit. Do not auto-infer.
SIDE_SIGN = {
    "LF": -1.0,
    "RF": +1.0,
    "LB": -1.0,
    "RB": +1.0,
}

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

# Exact runtime toe frame paths.
# Keep these constants because they are confirmed, stable interface names.
LEG_TO_TOE_FRAME_PATH = dict(RUNTIME_TOE_PATHS)

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

DEFAULT_INIT_JOINT_POS = SUPPORT_JOINT_POS

# -----------------------------------------------------------------------------
# Core articulation cfg for training / evaluation
# -----------------------------------------------------------------------------
# Important training-side change relative to the old baseline cfg:
#   activate_contact_sensors=True
#
# Why it is enabled here:
# - RL observations will need contact state / net contact force.
# - This switch enables PhysX contact reporting on the robot asset so that
#   ContactSensorCfg in the scene can query contacts.
BITTLE_CFG_RL = ArticulationCfg(
    prim_path=RUNTIME_ROOT_PATH,
    spawn=sim_utils.UsdFileCfg(
        usd_path=BITTLE_USD_PATH,
        activate_contact_sensors=True,
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

# Backward-compatible alias so existing imports can be gradually migrated.
BITTLE_CFG = BITTLE_CFG_RL

# -----------------------------------------------------------------------------
# Reusable sensor cfg blocks for RL scene construction
# -----------------------------------------------------------------------------
# IMU:
# We now use the confirmed explicit IMU link rather than the robot root regex.
# Exact runtime path:
#   {ENV_REGEX_NS}/bittle/base_frame_link/mainboard_link/imu_link
BITTLE_IMU_CFG = ImuCfg(
    prim_path=RUNTIME_IMU_PATH,
    gravity_bias=(0.0, 0.0, 9.81),
    debug_vis=False,
)

# Foot contact sensor:
# First-pass RL contact sensing is attached to the four knee links because:
# 1) the inspection confirmed toe_proxy is only an Xform tracking frame;
# 2) the knee links are rigid-body-bearing leg links;
# 3) this is the safest stable interface before doing deeper collider probing.
#
# Known runtime contact body names:
#   {ENV_REGEX_NS}/bittle/left_front_knee_link
#   {ENV_REGEX_NS}/bittle/right_front_knee_link
#   {ENV_REGEX_NS}/bittle/left_back_knee_link
#   {ENV_REGEX_NS}/bittle/right_back_knee_link
#
# If a later runtime contact-debug script proves that lower collider children are
# cleanly reportable, this can be refined. Until then, keep this conservative.
BITTLE_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/bittle/(left_front_knee_link|right_front_knee_link|left_back_knee_link|right_back_knee_link)",
    update_period=0.0,
    history_length=6,
    debug_vis=False,
)

# Base->feet frame transformer:
# Use the exact confirmed toe_proxy paths. These are not guessed regexes anymore.
# They are the best available foot reference frames in the patched asset.
BITTLE_FRAME_TRANSFORMER_CFG = FrameTransformerCfg(
    prim_path=RUNTIME_BASE_FRAME_PATH,
    target_frames=[
        FrameTransformerCfg.FrameCfg(name="LF_toe", prim_path=RUNTIME_CONTACT_BODY_PATHS["LF"]),
        FrameTransformerCfg.FrameCfg(name="RF_toe", prim_path=RUNTIME_CONTACT_BODY_PATHS["RF"]),
        FrameTransformerCfg.FrameCfg(name="LB_toe", prim_path=RUNTIME_CONTACT_BODY_PATHS["LB"]),
        FrameTransformerCfg.FrameCfg(name="RB_toe", prim_path=RUNTIME_CONTACT_BODY_PATHS["RB"]),
    ],
    debug_vis=False,
)