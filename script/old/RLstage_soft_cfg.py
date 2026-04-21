from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


def _as_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _as_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain_dict(v) for v in obj]
    return obj


@dataclass
class RLstage_A_CommandCfg:
    min_v_cmd_mps: float = 0.05
    max_v_cmd_mps: float = 0.30
    switch_interval_s_min: float = 2.0
    switch_interval_s_max: float = 3.0
    yaw_cmd_radps: float = 0.0
    # Project truth: BODY +Y is the robot forward direction.
    # Keep this default aligned with debug/reward/summary code in RLstage_A_env.py.
    forward_axis: str = "y"


@dataclass
class RLstage_A_ActionCfg:
    dim: int = 8
    scale_sh_rad: float = 0.10
    scale_kn_rad: float = 0.10
    rate_limit_rad_per_ctrl_step: float = 0.04
    decimation: int = 4
    residual_warmup_s: float = 0.50


@dataclass
class RLstage_A_ObsCfg:
    """Compact policy observation for the soft-ground residual stage.

    The intent is to keep the policy input easy to explain in a paper:
    - body stability state,
    - deviation from baseline,
    - contact/support state,
    - foot location proxy,
    - gait phase,
    - previous residual,
    - commanded speed.

    Final dimension breakdown (current default = 51):
        base_ang_vel_b      3
        projected_gravity   3
        joint_error         8
        joint_vel           8
        contact_binary      4
        toe_pos_body       12
        phase_features      4
        prev_action         8
        v_cmd               1
    """

    include_joint_error: bool = True
    include_joint_vel: bool = True
    include_contact_binary: bool = True
    include_toe_pos_body: bool = True
    include_prev_action: bool = True
    include_phase_features: bool = True
    include_command: bool = True
    obs_dim_policy: int = 51


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
    max_abs_roll_rad: float = 0.55
    max_abs_pitch_rad: float = 0.55
    episode_length_s: float = 10.0
    nan_terminate: bool = True


@dataclass
class RLstage_A_RewardCfg:
    """Soft-ground-aligned residual reward.

    Priorities are intentionally simple and paper-friendly:
    1) track the commanded forward speed,
    2) stay upright with small roll/pitch and small yaw drift,
    3) avoid vertical body bounce on compliant ground,
    4) keep support transfer clean (continuity / low scuff / low diagonal asymmetry),
    5) keep residual corrections small and smooth.
    """

    vel_track_sigma_mps: float = 0.15
    w_forward_track: float = 1.20
    w_alive: float = 1.00
    w_support_continuity: float = 0.80

    # Straight-line stabilization terms.
    w_yaw_abs_pen: float = 0.40
    w_yaw_rate_pen: float = 0.90
    w_roll_pitch_pen: float = 1.35
    w_vertical_bounce_pen: float = 0.35

    # Residual regularization terms.
    w_action_mag_pen: float = 0.06
    w_action_rate_pen: float = 0.12

    # Contact-quality terms.
    w_scuff_pen: float = 0.70
    w_diag_asym_pen: float = 0.60

    contact_force_threshold_n: float = 0.5
    support_continuity_min_contacts: float = 2.0


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

    # rsl_rl PPO defaults chosen to be conservative and easy to debug first.
    num_steps_per_env: int = 24
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
        """Build an RSL-RL config dictionary that is compatible with newer Isaac Lab / rsl_rl.

        Why this helper looks slightly redundant
        ----------------------------------------
        Older project versions used the legacy three-block layout:
            {"algorithm": ..., "policy": ..., "runner": ...}

        The currently installed Isaac Lab / rsl_rl stack expects the newer flattened
        runner layout instead:
            - top-level "actor"
            - top-level "critic"
            - top-level "algorithm"
            - top-level runner fields such as "num_steps_per_env" and "save_interval"
            - top-level "obs_groups"

        If we only provide the old "policy" block, current rsl_rl fails during
        OnPolicyRunner(...) construction with errors such as:
            KeyError: 'actor'

        Therefore this function now emits the *new* keys required by current
        rsl_rl, while also keeping a small legacy "policy" block as an explicit
        comment-carrying compatibility shim for older code paths / debugging.
        """

        # ------------------------------------------------------------------
        # New-style model configs required by current rsl_rl (Isaac Lab 2.x /
        # recent rsl_rl releases). The actor is stochastic, so it needs an
        # output distribution. The critic is deterministic and therefore keeps
        # distribution_cfg=None.
        # ------------------------------------------------------------------
        actor_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": list(self.actor_hidden_dims),
            "activation": self.activation,
            "obs_normalization": False,
            # rsl_rl >= 5 expects distribution_cfg instead of the old
            # init_noise_std / noise_std_type fields on the model itself.
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": self.policy_init_noise_std,
                "std_type": "scalar",
            },
        }
        critic_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": list(self.critic_hidden_dims),
            "activation": self.activation,
            "obs_normalization": False,
            "distribution_cfg": None,
        }

        # ------------------------------------------------------------------
        # New-style PPO algorithm config. Keep rnd_cfg / symmetry_cfg present
        # explicitly as None because newer OnPolicyRunner.learn() accesses
        # cfg["algorithm"]["rnd_cfg"] directly.
        # ------------------------------------------------------------------
        algorithm_cfg = {
            "class_name": "PPO",
            "clip_param": self.clip_param,
            "desired_kl": self.desired_kl,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "lam": self.lam,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "schedule": self.schedule,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "value_loss_coef": self.value_loss_coef,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }

        # ------------------------------------------------------------------
        # Observation-group routing for current rsl_rl model construction.
        # Our wrapped env exposes the compact policy observation under the
        # "policy" key, and both actor and critic consume that same tensor.
        # ------------------------------------------------------------------
        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        return {
            # New-style top-level runner fields used directly by current
            # OnPolicyRunner.learn().
            "class_name": self.runner_class_name,
            "seed": self.seed,
            "device": "cuda:0",
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "empirical_normalization": self.empirical_normalization,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": "tensorboard",
            "resume": False,
            "load_run": ".*",
            "load_checkpoint": "model_.*.pt",
            "obs_groups": obs_groups,
            "algorithm": algorithm_cfg,
            "actor": actor_cfg,
            "critic": critic_cfg,
            # multi_gpu is normally injected by OnPolicyRunner itself. We keep
            # the key absent here and let the runner populate it.
            # ------------------------------------------------------------------
            # Legacy compatibility/debug block.
            # This is not used by current rsl_rl for algorithm construction, but
            # keeping it here makes the config self-explanatory and easier to
            # compare against older Isaac Lab examples and past project logs.
            # ------------------------------------------------------------------
            "policy": {
                "class_name": "ActorCritic",
                "activation": self.activation,
                "actor_hidden_dims": list(self.actor_hidden_dims),
                "critic_hidden_dims": list(self.critic_hidden_dims),
                "init_noise_std": self.policy_init_noise_std,
            },
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
