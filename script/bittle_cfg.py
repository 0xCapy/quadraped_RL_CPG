# bittle_cfg.py
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo root = .../quadraped_RL_CPG
REPO_ROOT = Path(__file__).resolve().parents[1]

# NEW: point to your fixed collider USD
# Your file: D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_phyx.usd
BITTLE_USD_PATH = (REPO_ROOT / "bittle" / "bittle_phyx_NOGROUND.usd").as_posix()

# Symmetric stand pose (radians) for your joint names.
# If knees bend the wrong direction, flip the sign of ALL knee values.
STAND_JOINT_POS = {
    "left_front_shoulder_joint":  0.35,
    "right_front_shoulder_joint": -0.35,
    "left_back_shoulder_joint":   0.35,
    "right_back_shoulder_joint":  -0.35,
    "left_front_knee_joint":      0.70,
    "right_front_knee_joint":     -0.70,
    "left_back_knee_joint":       0.70,
    "right_back_knee_joint":      -0.70,
}

BITTLE_CFG = ArticulationCfg(
    # IMPORTANT: keep consistent with your working stand/CPG scripts
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
            enabled_self_collisions=False,  # keep off for Phase A/B stability
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Keep your 0.1m-scale start height (you already tuned around this)
        pos=(0.0, 0.0, 0.12),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=STAND_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=6.0,
            stiffness=40.0,
            damping=12.0,
        )
    },
)