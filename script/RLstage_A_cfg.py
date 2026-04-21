from __future__ import annotations

"""Stage-A project configuration for residual PPO training.

This file is intentionally self-contained.

Important maintenance rule
--------------------------
This module MUST NOT import ``RLstage_A_cfg`` anywhere inside itself.
A previous bad patch accidentally inserted::

    a self-import statement for this same module

inside the file, which causes a circular self-import on startup. This rewrite
keeps the file standalone so replacing it into the project cannot recreate that
error.

Why the PPO config looks different from older Isaac Lab examples
----------------------------------------------------------------
The user's current Isaac Lab / rsl_rl stack expects the newer runner schema with
TOP-LEVEL ``actor`` / ``critic`` / ``algorithm`` blocks. Official Isaac Lab docs
show these as attributes of ``RslRlOnPolicyRunnerCfg`` and mark the old
``policy`` block as deprecated for newer rsl_rl versions.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _as_plain_dict(obj: Any) -> Any:
    """Recursively convert nested dataclasses into plain Python containers."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _as_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain_dict(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Human-facing Stage-A configuration tree
# -----------------------------------------------------------------------------

@dataclass
class RLstage_A_CommandCfg:
    min_v_cmd_mps: float = 0.05
    max_v_cmd_mps: float = 0.30
    switch_interval_s_min: float = 2.0
    switch_interval_s_max: float = 3.0
    yaw_cmd_radps: float = 0.0
    # Project truth: BODY +Y is the robot forward direction.
    forward_axis: str = "y"


@dataclass
class RLstage_A_ActionCfg:
    # Action layout:
    #   8 joint residuals = [LF_sh, LF_kn, RF_sh, RF_kn, LB_sh, LB_kn, RB_sh, RB_kn]
    #   3 gait residual states = [delta_touchdown_buffer, delta_shoulder_amp, delta_knee_lift_amp]
    dim: int = 11

    # Fast joint-space residual scales.
    scale_sh_rad: float = 0.10
    scale_kn_rad: float = 0.10
    rate_limit_rad_per_ctrl_step: float = 0.04

    # Slow-varying gait-parameter residual scales.
    # These are updated every control step but rate-limited strongly so they act as
    # slowly changing gait states rather than twitchy per-step commands.
    scale_td_ratio: float = 0.10
    scale_sh_amp: float = 0.07
    scale_kn_amp: float = 0.10
    rate_limit_td_per_ctrl_step: float = 0.004
    rate_limit_sh_amp_per_ctrl_step: float = 0.003
    rate_limit_kn_amp_per_ctrl_step: float = 0.004

    decimation: int = 4
    residual_warmup_s: float = 0.50


@dataclass
class RLstage_A_ObsCfg:
    """Residual-policy observation with explicit heading, body linear velocity, and gait-state feedback.

    Final default dimension = 61
        base_ang_vel_b      3
        root_lin_vel_b      3
        projected_gravity   3
        joint_error         8
        joint_vel           8
        contact_binary      4
        toe_pos_body       12
        phase_features      4
        prev_action        11
        v_cmd               1
        heading_error       1
        gait_param_state    3
    """

    include_joint_error: bool = True
    include_joint_vel: bool = True
    include_contact_binary: bool = True
    include_toe_pos_body: bool = True
    include_prev_action: bool = True
    include_phase_features: bool = True
    include_command: bool = True
    include_root_lin_vel: bool = True
    include_heading_error: bool = True
    include_gait_param_state: bool = True
    obs_dim_policy: int = 61


@dataclass
class RLstage_A_ResetCfg:
    # User-confirmed placement height for the current asset/scaling in simulation.
    root_height_m: float = 1.03
    xy_jitter_m: float = 0.00
    yaw_jitter_rad: float = 0.06
    joint_jitter_rad: float = 0.02
    lin_vel_jitter_mps: float = 0.02
    ang_vel_jitter_radps: float = 0.05


@dataclass
class RLstage_A_TerminationCfg:
    min_base_height_m: float = 0.06
    max_abs_roll_rad: float = 0.35
    max_abs_pitch_rad: float = 0.35
    max_abs_heading_err_rad: float = 0.30
    episode_length_s: float = 15.0
    nan_terminate: bool = True

    # Success is stricter than mere timeout: the episode must finish while
    # keeping heading/tilt/bounce within acceptable stability bounds.
    success_max_heading_err_rad: float = 0.12
    success_max_tilt_rms_rad: float = 0.06
    success_max_vertical_bounce_rms_mps: float = 0.16


@dataclass
class RLstage_A_RewardCfg:
    """Residual reward for straight-line stabilization on compliant ground.

    Current priority order:
    1) keep heading error and yaw motion small,
    2) keep tilt and vertical bounce small,
    3) track forward speed,
    4) only then improve contact-cleanliness terms such as scuff / diagonal asymmetry.
    """

    vel_track_sigma_mps: float = 0.15
    w_forward_track: float = 0.70
    w_alive: float = 0.35
    w_support_continuity: float = 0.25

    # heading / stability first
    w_yaw_abs_pen: float = 2.40
    w_yaw_rate_pen: float = 2.20
    w_roll_pitch_pen: float = 2.80
    w_vertical_bounce_pen: float = 1.60

    # keep residuals controlled, but do not over-regularize away heading recovery
    w_action_mag_pen: float = 0.05
    w_action_rate_pen: float = 0.10

    # contact-quality terms are deliberately de-emphasized relative to heading/tilt/bounce
    w_scuff_pen: float = 0.10
    w_diag_asym_pen: float = 0.12

    contact_force_threshold_n: float = 0.5
    support_continuity_min_contacts: float = 2.0
    scuff_swing_ratio: float = 0.36


@dataclass
class RLstage_A_GroundCfg:
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class RLstage_A_TrainCfg:
    num_envs: int = 512
    env_spacing_m: float = 2.5
    sim_dt_s: float = 1.0 / 120.0
    policy_dt_s: float = 1.0 / 30.0
    max_iterations: int = 3000
    eval_every_iters: int = 20
    save_trace_episodes_per_eval: int = 4
    headless: bool = True
    seed: int = 42
    runner_class_name: str = "OnPolicyRunner"
    experiment_name: str = "RLstage_A"
    run_name: str = "straight_residual_ppo"

    # Conservative PPO defaults for first-pass debugging.
    num_steps_per_env: int = 120
    save_interval: int = 100
    empirical_normalization: bool = False
    policy_init_noise_std: float = 1.0

    actor_hidden_dims: tuple[int, ...] = (256, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"

    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    def to_rsl_rl_dict(self, num_obs: int, num_actions: int) -> Dict[str, Any]:
        """Build the newer top-level rsl_rl runner dictionary.

        Why this exact shape is used
        ----------------------------
        The user's stack raised ``KeyError: 'actor'`` inside
        ``rsl_rl.algorithms.ppo.PPO.construct_algorithm(...)``. That tells us the
        runner expects top-level ``actor`` and ``critic`` blocks rather than only
        the deprecated nested ``policy`` block.

        The structure below mirrors the current Isaac Lab documentation for
        ``RslRlOnPolicyRunnerCfg`` / ``RslRlPpoAlgorithmCfg``.
        """
        actor_cfg = {
            # Newer rsl_rl release notes explicitly mention the move toward
            # generic models like MLPModel instead of the old ActorCritic-only
            # path. The actor is stochastic, so it needs a Gaussian distribution.
            "class_name": "MLPModel",
            "hidden_dims": list(self.actor_hidden_dims),
            "activation": self.activation,
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": self.policy_init_noise_std,
                "std_type": "scalar",
            },
        }

        critic_cfg = {
            # Critic is deterministic scalar-value prediction, so no output
            # distribution is attached.
            "class_name": "MLPModel",
            "hidden_dims": list(self.critic_hidden_dims),
            "activation": self.activation,
            "obs_normalization": False,
            "distribution_cfg": None,
        }

        algorithm_cfg = {
            "class_name": "PPO",
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "learning_rate": self.learning_rate,
            "schedule": self.schedule,
            "gamma": self.gamma,
            "lam": self.lam,
            "entropy_coef": self.entropy_coef,
            "desired_kl": self.desired_kl,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": "adam",
            "value_loss_coef": self.value_loss_coef,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "clip_param": self.clip_param,
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            # Keep these explicit because some runner paths access them directly.
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }

        # The environment exposes observations under the single key "policy".
        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        return {
            # Base runner fields.
            "seed": self.seed,
            "device": "cuda:0",
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": self.empirical_normalization,
            "obs_groups": obs_groups,
            "clip_actions": None,
            "check_for_nan": True,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": "tensorboard",
            "neptune_project": "isaaclab",
            "wandb_project": "isaaclab",
            "resume": False,
            "load_run": ".*",
            "load_checkpoint": "model_.*.pt",
            # On-policy runner fields.
            "class_name": self.runner_class_name,
            "actor": actor_cfg,
            "critic": critic_cfg,
            "algorithm": algorithm_cfg,
            # Optional field used by distributed launches. Keep explicit for logs.
            "multi_gpu": None,
            # Lightweight metadata for debug printing only.
            "meta": {
                "num_obs": num_obs,
                "num_actions": num_actions,
            },
        }


@dataclass
class RLstage_A_LogCfg:
    print_debug_every_ctrl_steps: int = 25
    save_summaries_csv: bool = True
    save_traces_npz: bool = True
    save_bundles_npz: bool = True
    save_failure_dump_npz: bool = True
    summary_filename: str = "summaries.csv"
    human_summary_filename: str = "summaries.txt"


@dataclass
class RLstage_A_DebugCfg:
    rollout_steps: int = 360
    print_startup_banner: bool = True
    print_joint_maps: bool = True
    assert_finite_obs: bool = True
    assert_finite_reward: bool = True
    assert_finite_action: bool = True
    detect_all_zero_contacts_window: int = 60
    detect_frozen_toe_window: int = 60


@dataclass
class RLstage_A_Config:
    command: RLstage_A_CommandCfg = field(default_factory=RLstage_A_CommandCfg)
    action: RLstage_A_ActionCfg = field(default_factory=RLstage_A_ActionCfg)
    obs: RLstage_A_ObsCfg = field(default_factory=RLstage_A_ObsCfg)
    reset: RLstage_A_ResetCfg = field(default_factory=RLstage_A_ResetCfg)
    termination: RLstage_A_TerminationCfg = field(default_factory=RLstage_A_TerminationCfg)
    reward: RLstage_A_RewardCfg = field(default_factory=RLstage_A_RewardCfg)
    ground: RLstage_A_GroundCfg = field(default_factory=RLstage_A_GroundCfg)
    train: RLstage_A_TrainCfg = field(default_factory=RLstage_A_TrainCfg)
    log: RLstage_A_LogCfg = field(default_factory=RLstage_A_LogCfg)
    debug: RLstage_A_DebugCfg = field(default_factory=RLstage_A_DebugCfg)

    def to_dict(self) -> Dict[str, Any]:
        return _as_plain_dict(self)
