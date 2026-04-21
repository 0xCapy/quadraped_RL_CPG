from __future__ import annotations

"""RLstage_A Isaac Lab environment.

This file intentionally carries more comments than strictly necessary.
The goal is that if project memory is lost later, the next reader can still
recover the design choices, known pitfalls, and the reasons behind them.

Key Stage A truths implemented here
-----------------------------------
1) RL is residual-only. It corrects a fixed CPG baseline and does NOT own gait structure.
2) Stage A is straight-line only on flat rigid ground.
3) Sign system, support posture, diagonal phasing, and touchdown guard shape remain fixed.
4) Debuggability is a first-class requirement: human-readable summaries and stable export keys.
5) The current soft-ground branch uses a compact observation and a stability-first reward.
6) IMPORTANT coordinate convention for this project: BODY +Y is the forward direction.\n   Whenever Stage A computes commanded forward tracking, debug v_fwd_body, episode summaries,\n   or exported trace fields, it must use the body-frame Y velocity component, not X.\n
Known Isaac Lab / Isaac Sim integration pitfalls handled here
-------------------------------------------------------------
- DirectRLEnv already owns read-only properties such as ``device`` and ``num_envs``.
  Do NOT assign to them in this subclass.
- The original ``toe_proxy`` frames are Xforms, not rigid bodies. FrameTransformer only
  supports rigid-body endpoints. For Stage A we therefore route the frame-transformer to the
  four knee links while keeping output names stable (LF_toe / RF_toe / LB_toe / RB_toe).
- The first generated version of this file used chained tensor indexing in the CPG phase
  shaping logic (e.g. ``x[mask1][mask2] = ...``). That writes into a temporary tensor and
  silently drops the assignment. The phase-scalar computation below avoids that bug by working
  on flattened tensors and assigning with direct boolean masks.
- Reset writers must not write pose and velocity separately via ``write_root_state_to_sim``
  using zero-filled placeholders, because the second call would overwrite the first. We always
  prefer dedicated pose/velocity writers, and only fall back to a single combined root-state
  write when necessary.
"""

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from RLstage_A_cfg import RLstage_A_Config
from RLstage_A_log import (
    DONE_REASON_STR,
    build_episode_summary,
    format_episode_summary,
    make_copy_paste_eval_bundle,
    make_episode_stats,
    reset_episode_stats,
    save_human_summaries,
    save_summaries_csv,
    save_trace_npz,
    update_episode_stats,
)
from RLstage_A_obs import (
    build_policy_obs,
    contact_binary_and_force_mag,
    first_attr,
    forward_speed_from_body_velocity,
    phase_features_from_groups,
    quat_to_euler_xyz,
    wrap_to_pi,
    yaw_to_quat_wxyz,
)
from RLstage_A_rew import compute_stage_a_reward, diag_support_asymmetry, support_continuity

from bittle_cpg_core import DEFAULT_GAIT, DEFAULT_LIMITS, KN_SIGN, SH_SIGN, build_joint_maps
from bittle_cfg_rl import (
    BITTLE_CFG_RL,
    BITTLE_CONTACT_SENSOR_CFG,
    BITTLE_FRAME_TRANSFORMER_CFG,
    BITTLE_IMU_CFG,
)

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, FrameTransformer, Imu
    from isaaclab.utils import configclass
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "RLstage_A_env.py requires Isaac Lab to be available in the runtime environment."
    ) from exc


# -----------------------------
# Done / termination codes
# -----------------------------
DONE_RUNNING = 0
DONE_TIMEOUT = 1
DONE_LOW_BASE = 2
DONE_ROLL = 3
DONE_PITCH = 4
DONE_NAN = 5
DONE_EXCEPTION = 6
DONE_HEADING = 7


# Stage A rigid-body distal frame fallback.
# Keep exported names later as LF_toe / RF_toe / ... even though the runtime source
# body is the knee link. This preserves downstream observation / logging interfaces.
# Stage A note:
# FrameTransformer in Isaac Lab needs GLOBAL prim-path regexes that start with "/".
# Do NOT use {ENV_REGEX_NS} here. It is not expanded by FrameTransformer init in this path.
# We therefore use explicit replicated-env global regexes under /World/envs/env_.*/...
_STAGEA_RIGID_DISTAL_PATHS = {
    "LF": "/World/envs/env_.*/bittle/left_front_knee_link",
    "RF": "/World/envs/env_.*/bittle/right_front_knee_link",
    "LB": "/World/envs/env_.*/bittle/left_back_knee_link",
    "RB": "/World/envs/env_.*/bittle/right_back_knee_link",
}


def _control_dt_from_cfg(cfg: DirectRLEnvCfg) -> float:
    """Stage control dt = physics dt * decimation."""
    return float(cfg.sim.dt) * int(cfg.decimation)


def _spawn_ground_plane(stage_cfg: RLstage_A_Config) -> None:
    """Version-tolerant ground-plane spawn.

    Primary path: Isaac Lab GroundPlaneCfg.
    Fallback path: omni.physx helper.
    """
    try:
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=stage_cfg.ground.static_friction,
            dynamic_friction=stage_cfg.ground.dynamic_friction,
            restitution=stage_cfg.ground.restitution,
        )
        gp = sim_utils.GroundPlaneCfg(physics_material=mat)
        if hasattr(gp, "func"):
            gp.func("/World/GroundPlane", gp)
            return
    except Exception:
        pass

    try:  # pragma: no cover
        from omni.physx.scripts import physicsUtils

        physicsUtils.add_ground_plane(
            stage=None,
            planePath="/World/GroundPlane",
            axis="Z",
            size=10.0,
            position=torch.tensor([0.0, 0.0, 0.0]),
            color=torch.tensor([0.5, 0.5, 0.5]),
        )
        return
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to spawn a ground plane. Replace _spawn_ground_plane with your local helper if needed."
        ) from exc


@configclass
class _SceneCfg(InteractiveSceneCfg):
    # Defaults here are placeholders. The real number of envs is overridden in make_isaac_env_cfg().
    num_envs = 512
    env_spacing = 2.5
    replicate_physics = True

    robot = BITTLE_CFG_RL
    imu = BITTLE_IMU_CFG
    contact_sensor = BITTLE_CONTACT_SENSOR_CFG
    frame_transformer = BITTLE_FRAME_TRANSFORMER_CFG


@configclass
class RLstage_A_IsaacCfg(DirectRLEnvCfg):
    scene = _SceneCfg()
    sim = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    decimation = 4
    episode_length_s = 10.0
    action_space = 8
    observation_space = 51
    state_space = 0


def make_isaac_env_cfg(stage_cfg: RLstage_A_Config, num_envs: int | None = None) -> RLstage_A_IsaacCfg:
    """Build the Isaac-side env cfg from the project-side Stage A cfg.

    ``stage_cfg`` is the human-facing configuration tree.
    ``RLstage_A_IsaacCfg`` is the DirectRLEnv config actually consumed by Isaac Lab.
    """
    cfg = RLstage_A_IsaacCfg()
    cfg.scene.num_envs = int(num_envs if num_envs is not None else stage_cfg.train.num_envs)
    cfg.scene.env_spacing = stage_cfg.train.env_spacing_m
    cfg.decimation = stage_cfg.action.decimation
    cfg.sim.dt = stage_cfg.train.sim_dt_s
    cfg.sim.render_interval = stage_cfg.action.decimation
    cfg.episode_length_s = stage_cfg.termination.episode_length_s
    cfg.observation_space = stage_cfg.obs.obs_dim_policy
    cfg.action_space = stage_cfg.action.dim
    cfg.state_space = 0
    cfg.stage_cfg = stage_cfg
    return cfg


class RLstage_AEnv(DirectRLEnv):
    cfg: RLstage_A_IsaacCfg

    def __init__(self, cfg: RLstage_A_IsaacCfg, render_mode: str | None = None, **kwargs):
        # Keep a direct pointer to the human-facing Stage A cfg.
        self.stage_cfg: RLstage_A_Config = cfg.stage_cfg
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Do NOT assign to self.device or self.num_envs; DirectRLEnv already owns those.
        self.num_actions = self.stage_cfg.action.dim
        self.num_obs = self.stage_cfg.obs.obs_dim_policy

        self.joint_names = list(self.robot.data.joint_names)
        self.name_to_idx, self.leg_to_sh, self.leg_to_kn = build_joint_maps(self.joint_names)
        self.action_order = ("LF", "RF", "LB", "RB")

        self._build_runtime_constants()
        self._build_runtime_buffers()

        self._trace_root = Path.cwd() / "RLstage_A_outputs"
        self._trace_root.mkdir(parents=True, exist_ok=True)

        if self.stage_cfg.debug.print_startup_banner:
            self._print_startup_banner()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        _spawn_ground_plane(self.stage_cfg)

        # Force the frame-transformer endpoints onto rigid bodies.
        # The original toe_proxy frames are Xforms, not rigid bodies, so FrameTransformer
        # rejects them at runtime. Stage A therefore tracks knee links as a conservative proxy.
        try:
            rigid_paths = [
                _STAGEA_RIGID_DISTAL_PATHS["LF"],
                _STAGEA_RIGID_DISTAL_PATHS["RF"],
                _STAGEA_RIGID_DISTAL_PATHS["LB"],
                _STAGEA_RIGID_DISTAL_PATHS["RB"],
            ]
            for frame_cfg, rigid_path in zip(self.cfg.scene.frame_transformer.target_frames, rigid_paths):
                frame_cfg.prim_path = rigid_path
        except Exception:
            # Keep the env robust even if local config-class mutation differs across versions.
            pass

        self.robot = Articulation(self.cfg.scene.robot)
        self.imu = Imu(self.cfg.scene.imu)
        self.contact_sensor = ContactSensor(self.cfg.scene.contact_sensor)
        self.frame_transformer = FrameTransformer(self.cfg.scene.frame_transformer)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["imu"] = self.imu
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["frame_transformer"] = self.frame_transformer

        self.scene.clone_environments(copy_from_source=False)
        if hasattr(self.scene, "filter_collisions"):
            self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

    # ------------------------------------------------------------------
    # Runtime constants / buffers
    # ------------------------------------------------------------------
    def _build_runtime_constants(self) -> None:
        dtype = torch.float32
        device = self.device
        num_joints = self.robot.num_joints

        self._sh_sign = torch.tensor([SH_SIGN[leg] for leg in self.action_order], device=device, dtype=dtype)
        self._kn_sign = torch.tensor([KN_SIGN[leg] for leg in self.action_order], device=device, dtype=dtype)
        self._leg_sh_idx = torch.tensor([self.leg_to_sh[leg] for leg in self.action_order], device=device, dtype=torch.long)
        self._leg_kn_idx = torch.tensor([self.leg_to_kn[leg] for leg in self.action_order], device=device, dtype=torch.long)

        self._shoulder_limits = torch.tensor(DEFAULT_LIMITS.shoulder_lim, device=device, dtype=dtype)
        self._knee_limits = torch.tensor(DEFAULT_LIMITS.knee_lim, device=device, dtype=dtype)

        default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.q_support = default_joint_pos.clone()
        self._q_default = default_joint_pos.clone()
        self._q_zero = torch.zeros(self.num_envs, num_joints, device=device, dtype=dtype)
        self._root_quat_identity = torch.zeros(self.num_envs, 4, device=device, dtype=dtype)
        self._root_quat_identity[:, 0] = 1.0

        # Per-dimension normalized action-rate limits.
        # The config stores physical per-control-step budgets, while the policy output is normalized.
        sh_rate_lim_norm = self.stage_cfg.action.rate_limit_rad_per_ctrl_step / max(1.0e-6, self.stage_cfg.action.scale_sh_rad)
        kn_rate_lim_norm = self.stage_cfg.action.rate_limit_rad_per_ctrl_step / max(1.0e-6, self.stage_cfg.action.scale_kn_rad)
        td_rate_lim_norm = self.stage_cfg.action.rate_limit_td_per_ctrl_step / max(1.0e-6, self.stage_cfg.action.scale_td_ratio)
        sh_amp_rate_lim_norm = self.stage_cfg.action.rate_limit_sh_amp_per_ctrl_step / max(1.0e-6, self.stage_cfg.action.scale_sh_amp)
        kn_amp_rate_lim_norm = self.stage_cfg.action.rate_limit_kn_amp_per_ctrl_step / max(1.0e-6, self.stage_cfg.action.scale_kn_amp)
        self._action_rate_limit_norm = torch.tensor(
            [sh_rate_lim_norm, kn_rate_lim_norm, sh_rate_lim_norm, kn_rate_lim_norm,
             sh_rate_lim_norm, kn_rate_lim_norm, sh_rate_lim_norm, kn_rate_lim_norm,
             td_rate_lim_norm, sh_amp_rate_lim_norm, kn_amp_rate_lim_norm],
            device=device,
            dtype=dtype,
        )
        self._gait_param_min = torch.tensor([-0.08, -0.06, -0.08], device=device, dtype=dtype)
        self._gait_param_max = torch.tensor([+0.10, +0.07, +0.10], device=device, dtype=dtype)
        self._gait_param_default = torch.tensor(
            [DEFAULT_GAIT.touchdown_buffer_ratio, DEFAULT_GAIT.shoulder_amp, DEFAULT_GAIT.knee_lift_amp],
            device=device,
            dtype=dtype,
        )

    def _build_runtime_buffers(self) -> None:
        d = self.device
        num_joints = self.robot.num_joints

        self.gait_t = torch.zeros(self.num_envs, device=d)
        self.episode_time = torch.zeros(self.num_envs, device=d)
        self.next_cmd_switch_t = torch.zeros(self.num_envs, device=d)
        self.v_cmd = torch.full((self.num_envs, 1), self.stage_cfg.command.min_v_cmd_mps, device=d)
        self.yaw_cmd = torch.zeros((self.num_envs, 1), device=d)

        # Group-A starts at 0.0, Group-B starts at 0.5 to preserve diagonal phasing truth.
        self.group_a_phase = torch.zeros(self.num_envs, device=d)
        self.group_b_phase = torch.full((self.num_envs,), 0.5, device=d)

        self.action_residual = torch.zeros(self.num_envs, self.num_actions, device=d)
        self.prev_action_residual = torch.zeros_like(self.action_residual)
        self.action_after_rate_limit = torch.zeros_like(self.action_residual)

        self.q_baseline = self.q_support.clone()
        self.q_final = self.q_support.clone()
        self.last_rewards = torch.zeros(self.num_envs, device=d)
        self.last_reward_terms: Dict[str, torch.Tensor] = {}
        self.last_done_reason = torch.zeros(self.num_envs, dtype=torch.long, device=d)
        self.last_obs = torch.zeros(self.num_envs, self.num_obs, device=d)
        self.heading_ref_yaw = torch.zeros(self.num_envs, device=d)
        self.gait_param_state = torch.zeros(self.num_envs, 3, device=d)
        self.gait_param_effective = self._gait_param_default.unsqueeze(0).repeat(self.num_envs, 1)

        self.episode_stats = make_episode_stats(self.num_envs, d)
        self.trace_cache: Dict[int, Dict[str, list]] = {}
        self._sample_trace_env_ids = torch.arange(
            min(self.stage_cfg.train.save_trace_episodes_per_eval, self.num_envs), device=d
        )

        # Completed-episode archive for training-time convergence tracking.
        #
        # Why this exists:
        # - The original summaries.csv only saved a tiny fixed subset of envs at the end.
        # - For convergence analysis we want a rolling stream of completed episodes, so a
        #   background monitor can aggregate yaw / roll / pitch / bounce metrics over time.
        self.completed_episode_summaries: list[Dict[str, Any]] = []
        self._completed_episode_counter = 0
        self._completed_episode_lock = threading.Lock()


    def _print_startup_banner(self) -> None:
        print("=" * 88)
        print("RLstage_A startup")
        print(f"USD path      : {self.cfg.scene.robot.spawn.usd_path}")
        print(f"runtime root  : {self.cfg.scene.robot.prim_path}")
        print(f"IMU path      : {self.cfg.scene.imu.prim_path}")
        print(f"contact path  : {self.cfg.scene.contact_sensor.prim_path}")
        print(f"toe frames    : {[f.name for f in self.cfg.scene.frame_transformer.target_frames]}")
        print(f"action order  : {self.action_order}")
        print(f"num envs      : {self.num_envs}")
        print(f"num joints    : {self.robot.num_joints}")
        print(f"num obs       : {self.num_obs}")
        print(f"num actions   : {self.num_actions}")
        print(f"sim dt        : {self.stage_cfg.train.sim_dt_s:.6f}")
        print(f"policy dt     : {_control_dt_from_cfg(self.cfg):.6f}")
        print(f"root height   : {self.stage_cfg.reset.root_height_m:.3f}")
        print(f"v_cmd range   : [{self.stage_cfg.command.min_v_cmd_mps:.2f}, {self.stage_cfg.command.max_v_cmd_mps:.2f}]")
        if self.stage_cfg.debug.print_joint_maps:
            print("joint map     :", self.name_to_idx)
        print("=" * 88)

    # ------------------------------------------------------------------
    # Command schedule
    # ------------------------------------------------------------------
    def _sample_command_speed(self, env_ids: torch.Tensor) -> None:
        lo = self.stage_cfg.command.min_v_cmd_mps
        hi = self.stage_cfg.command.max_v_cmd_mps
        n = int(env_ids.numel())
        if n == 0:
            return
        self.v_cmd[env_ids, 0] = lo + (hi - lo) * torch.rand(n, device=self.device)
        self.yaw_cmd[env_ids, 0] = self.stage_cfg.command.yaw_cmd_radps
        self.next_cmd_switch_t[env_ids] = self.episode_time[env_ids] + (
            self.stage_cfg.command.switch_interval_s_min
            + (self.stage_cfg.command.switch_interval_s_max - self.stage_cfg.command.switch_interval_s_min)
            * torch.rand(n, device=self.device)
        )

    def _maybe_resample_commands(self) -> None:
        switch_ids = torch.nonzero(self.episode_time >= self.next_cmd_switch_t, as_tuple=False).squeeze(-1)
        if switch_ids.numel() > 0:
            self._sample_command_speed(switch_ids)

    # ------------------------------------------------------------------
    # Baseline CPG target generation
    # ------------------------------------------------------------------
    def _vectorized_leg_phase(self) -> torch.Tensor:
        freq = DEFAULT_GAIT.freq_hz
        group_a = (freq * self.gait_t) % 1.0
        group_b = (0.5 + freq * self.gait_t) % 1.0
        return torch.stack([group_a, group_b, group_b, group_a], dim=-1)

    def _phase_scalars(
        self,
        u: torch.Tensor,
        *,
        touchdown_buffer_ratio: torch.Tensor,
        shoulder_amp: torch.Tensor,
        knee_lift_amp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shoulder/knee template scalars for each leg phase.

        ``u`` has shape [N, 4] with one normalized phase value per leg.
        The three gait-shaping arguments are per-env vectors and are broadcast to all four legs.
        Implementation note:
        We flatten to 1D and assign with direct boolean masks to avoid the classic
        chained-indexing bug ``x[mask1][mask2] = ...``.
        """
        gait = DEFAULT_GAIT
        sr = float(max(0.10, min(0.90, gait.swing_ratio)))

        uf = u.reshape(-1)
        sh_amp_flat = shoulder_amp.unsqueeze(-1).expand_as(u).reshape(-1)
        kn_amp_flat = knee_lift_amp.unsqueeze(-1).expand_as(u).reshape(-1)
        td_flat = touchdown_buffer_ratio.unsqueeze(-1).expand_as(u).reshape(-1)

        shf = torch.zeros_like(uf)
        knf = torch.zeros_like(uf)

        swing = uf < sr
        if swing.any():
            s = uf[swing] / sr
            liftoff_end = float(max(0.05, min(0.80, gait.liftoff_ratio)))
            predown_start = float(max(liftoff_end + 0.05, min(0.98, 1.0 - gait.predown_ratio)))

            swing_idx = torch.nonzero(swing, as_tuple=False).squeeze(-1)

            liftoff = s < liftoff_end
            if liftoff.any():
                a = s[liftoff] / liftoff_end
                smooth = a * a * (3.0 - 2.0 * a)
                idx = swing_idx[liftoff]
                shf[idx] = sh_amp_flat[idx] * gait.liftoff_shoulder_frac * smooth
                knf[idx] = kn_amp_flat[idx] * 0.90 * smooth

            transfer = (s >= liftoff_end) & (s < predown_start)
            if transfer.any():
                a = (s[transfer] - liftoff_end) / max(1.0e-9, predown_start - liftoff_end)
                smooth = a * a * (3.0 - 2.0 * a)
                knee_floor = float(max(0.35, min(1.00, gait.transfer_knee_floor_frac)))
                idx = swing_idx[transfer]
                shf[idx] = sh_amp_flat[idx] * (
                    gait.liftoff_shoulder_frac + (1.0 - gait.liftoff_shoulder_frac) * smooth
                )
                knf[idx] = kn_amp_flat[idx] * (
                    knee_floor + (1.0 - knee_floor) * torch.sin(0.5 * math.pi * a)
                )

            predown = s >= predown_start
            if predown.any():
                a = (s[predown] - predown_start) / max(1.0e-9, 1.0 - predown_start)
                smooth = a * a * (3.0 - 2.0 * a)
                idx = swing_idx[predown]
                shf[idx] = sh_amp_flat[idx]
                knf[idx] = kn_amp_flat[idx] * (
                    gait.touchdown_knee_hold_frac
                    + (1.0 - gait.touchdown_knee_hold_frac) * (1.0 - smooth)
                )

        stance = ~swing
        if stance.any():
            w = (uf[stance] - sr) / max(1.0e-9, 1.0 - sr)
            td = torch.clamp(td_flat[stance], 0.0, 0.80)
            sh_end_frac = float(max(0.70, min(1.0, gait.touchdown_shoulder_end_frac)))
            stance_idx = torch.nonzero(stance, as_tuple=False).squeeze(-1)

            td_mask = w < td
            if td_mask.any():
                a = w[td_mask] / torch.clamp(td[td_mask], min=1.0e-9)
                smooth = a * a * (3.0 - 2.0 * a)
                idx = stance_idx[td_mask]
                shf[idx] = sh_amp_flat[idx] * (1.0 - (1.0 - sh_end_frac) * smooth)
                knf[idx] = kn_amp_flat[idx] * gait.touchdown_knee_hold_frac * (1.0 - smooth)

            sweep = ~td_mask
            if sweep.any():
                v = (w[sweep] - td[sweep]) / torch.clamp(1.0 - td[sweep], min=1.0e-9)
                smooth = v * v * (3.0 - 2.0 * v)
                idx = stance_idx[sweep]
                shf[idx] = sh_amp_flat[idx] * sh_end_frac * (1.0 - smooth)

        sh = shf.reshape_as(u)
        kn = knf.reshape_as(u)
        return sh, kn

    def _effective_gait_params(self, gait_param_state: torch.Tensor) -> torch.Tensor:
        gait_param_state = torch.clamp(gait_param_state, self._gait_param_min, self._gait_param_max)
        eff = self._gait_param_default.unsqueeze(0) + gait_param_state
        # Final safety clamps around the intended operating window.
        eff[:, 0] = torch.clamp(eff[:, 0], 0.16, 0.34)
        eff[:, 1] = torch.clamp(eff[:, 1], 0.08, 0.21)
        eff[:, 2] = torch.clamp(eff[:, 2], 0.18, 0.36)
        return eff

    def _build_baseline_targets(self, gait_param_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_support.clone()
        leg_phase = self._vectorized_leg_phase()  # [N, 4] in action_order = LF, RF, LB, RB
        gait_eff = self._effective_gait_params(gait_param_state)
        self.gait_param_effective.copy_(gait_eff)
        sh_scalar, kn_scalar = self._phase_scalars(
            leg_phase,
            touchdown_buffer_ratio=gait_eff[:, 0],
            shoulder_amp=gait_eff[:, 1],
            knee_lift_amp=gait_eff[:, 2],
        )

        alpha = torch.clamp(
            self.episode_time / max(1.0e-6, DEFAULT_GAIT.ramp_time), 0.0, 1.0
        ).unsqueeze(-1)
        for i, leg in enumerate(self.action_order):
            q[:, self.leg_to_sh[leg]] += alpha[:, 0] * self._sh_sign[i] * sh_scalar[:, i]
            q[:, self.leg_to_kn[leg]] += alpha[:, 0] * self._kn_sign[i] * kn_scalar[:, i]

        q[:, self._leg_sh_idx] = torch.clamp(q[:, self._leg_sh_idx], self._shoulder_limits[0], self._shoulder_limits[1])
        q[:, self._leg_kn_idx] = torch.clamp(q[:, self._leg_kn_idx], self._knee_limits[0], self._knee_limits[1])
        return q, leg_phase[:, 0], leg_phase[:, 1]

    def _apply_residual(self, q_baseline: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        q = q_baseline.clone()
        joint_residual = residual[:, :8]
        for i, leg in enumerate(self.action_order):
            q[:, self.leg_to_sh[leg]] += self.stage_cfg.action.scale_sh_rad * joint_residual[:, 2 * i + 0]
            q[:, self.leg_to_kn[leg]] += self.stage_cfg.action.scale_kn_rad * joint_residual[:, 2 * i + 1]
        q[:, self._leg_sh_idx] = torch.clamp(q[:, self._leg_sh_idx], self._shoulder_limits[0], self._shoulder_limits[1])
        q[:, self._leg_kn_idx] = torch.clamp(q[:, self._leg_kn_idx], self._knee_limits[0], self._knee_limits[1])
        return q

    # ------------------------------------------------------------------
    # DirectRLEnv hooks
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.num_actions:
            raise ValueError(f"Expected action dim {self.num_actions}, got {tuple(actions.shape)}")

        # Isaac Lab DirectRLEnv does not call a user-defined _post_physics_step() hook in the
        # same way some earlier internal templates did. Therefore Stage A must advance its own
        # control-time clocks here in _pre_physics_step(), once per policy/control step.
        ctrl_dt = _control_dt_from_cfg(self.cfg)
        self.episode_time += ctrl_dt
        self.gait_t += ctrl_dt
        self._maybe_resample_commands()

        actions = torch.clamp(actions, -1.0, 1.0)
        if self.stage_cfg.debug.assert_finite_action and not torch.isfinite(actions).all():
            raise RuntimeError("NaN/Inf in policy action.")

        self.prev_action_residual.copy_(self.action_after_rate_limit)
        delta = torch.clamp(
            actions - self.action_after_rate_limit,
            -self._action_rate_limit_norm,
            self._action_rate_limit_norm,
        )
        self.action_after_rate_limit = torch.clamp(self.action_after_rate_limit + delta, -1.0, 1.0)

        residual_gate = torch.clamp(
            self.episode_time / max(1.0e-6, self.stage_cfg.action.residual_warmup_s), 0.0, 1.0
        ).unsqueeze(-1)
        residual_to_apply = residual_gate * self.action_after_rate_limit

        self.gait_param_state = torch.stack(
            [
                self.stage_cfg.action.scale_td_ratio * residual_to_apply[:, 8],
                self.stage_cfg.action.scale_sh_amp * residual_to_apply[:, 9],
                self.stage_cfg.action.scale_kn_amp * residual_to_apply[:, 10],
            ],
            dim=-1,
        )

        self.q_baseline, self.group_a_phase, self.group_b_phase = self._build_baseline_targets(self.gait_param_state)
        self.q_final = self._apply_residual(self.q_baseline, residual_to_apply)
        self.action_residual.copy_(residual_to_apply)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.q_final)
        self.robot.write_data_to_sim()

    def _post_physics_step(self) -> None:
        # Intentionally left as a no-op for Stage A.
        # Keep the method only as documentation because earlier generated versions used it,
        # but the active clock/phase advancement is now done in _pre_physics_step().
        return

    # ------------------------------------------------------------------
    # Sensor readers
    # ------------------------------------------------------------------
    def _root_quat_w(self) -> torch.Tensor:
        data = self.robot.data
        return first_attr(data, ["root_quat_w", "body_quat_w"], self._root_quat_identity)

    def _root_pos_w(self) -> torch.Tensor:
        data = self.robot.data
        return first_attr(data, ["root_pos_w", "body_pos_w"], torch.zeros(self.num_envs, 3, device=self.device))

    def _root_lin_vel_b(self) -> torch.Tensor:
        data = self.robot.data
        return first_attr(data, ["root_lin_vel_b", "body_lin_vel_b"], torch.zeros(self.num_envs, 3, device=self.device))

    def _root_ang_vel_b(self) -> torch.Tensor:
        data = self.robot.data
        return first_attr(data, ["root_ang_vel_b", "body_ang_vel_b"], torch.zeros(self.num_envs, 3, device=self.device))

    def _imu_ang_vel_b(self) -> torch.Tensor:
        data = self.imu.data
        return first_attr(data, ["ang_vel_b", "angular_velocity"], self._root_ang_vel_b())

    def _projected_gravity_b(self) -> torch.Tensor:
        data = self.imu.data
        g = first_attr(data, ["projected_gravity_b", "projected_gravity"], None)
        if g is not None:
            return g
        out = torch.zeros(self.num_envs, 3, device=self.device)
        out[:, 2] = -1.0
        return out

    def _contact_force_xyz(self) -> torch.Tensor:
        data = self.contact_sensor.data
        x = first_attr(data, ["net_forces_w", "net_forces", "force_matrix_w", "force_matrix"], None)
        if x is None:
            return torch.zeros(self.num_envs, 4, 3, device=self.device)
        if x.ndim == 4:
            x = x[:, :, 0, :]
        if x.shape[1] > 4:
            x = x[:, :4, :]
        return x

    def _toe_pos_vel_body(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.frame_transformer.data
        pos = first_attr(data, ["target_pos_source", "target_pos_source_frame", "target_pos_source_b"], None)
        vel = first_attr(data, ["target_vel_source", "target_vel_source_frame", "target_lin_vel_source_b"], None)
        if pos is None:
            pos = torch.zeros(self.num_envs, 4, 3, device=self.device)
        if vel is None:
            vel = torch.zeros(self.num_envs, 4, 3, device=self.device)
        if pos.shape[1] > 4:
            pos = pos[:, :4, :]
        if vel.shape[1] > 4:
            vel = vel[:, :4, :]
        return pos, vel

    def _heading_error(self, yaw_world: torch.Tensor) -> torch.Tensor:
        return wrap_to_pi(yaw_world - self.heading_ref_yaw)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        root_lin_vel_b = self._root_lin_vel_b()
        base_ang_vel_b = self._imu_ang_vel_b()
        projected_gravity_b = self._projected_gravity_b()
        quat_w = self._root_quat_w()
        _, _, yaw = quat_to_euler_xyz(quat_w)
        heading_error = self._heading_error(yaw)
        contact_xyz = self._contact_force_xyz()
        contact_binary, contact_force_mag = contact_binary_and_force_mag(
            contact_xyz, self.stage_cfg.reward.contact_force_threshold_n
        )
        toe_pos_b, toe_vel_b = self._toe_pos_vel_body()
        phase_feat = phase_features_from_groups(self.group_a_phase, self.group_b_phase)

        obs_dict = build_policy_obs(
            base_ang_vel_b=base_ang_vel_b,
            root_lin_vel_b=root_lin_vel_b,
            projected_gravity_b=projected_gravity_b,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            q_baseline=self.q_baseline,
            prev_action=self.prev_action_residual,
            v_cmd=self.v_cmd,
            heading_error=heading_error,
            gait_param_state=self.gait_param_state,
            contact_binary=contact_binary,
            contact_force_mag=contact_force_mag,
            toe_pos_b=toe_pos_b,
            toe_vel_b=toe_vel_b,
            phase_features=phase_feat,
        )
        self.last_obs = obs_dict["policy"]
        if self.stage_cfg.debug.assert_finite_obs and not torch.isfinite(self.last_obs).all():
            raise RuntimeError("NaN/Inf in observation.")
        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._root_lin_vel_b()
        root_ang_vel_b = self._root_ang_vel_b()
        quat_w = self._root_quat_w()
        root_pos_w = self._root_pos_w()
        roll, pitch, yaw = quat_to_euler_xyz(quat_w)
        heading_error = self._heading_error(yaw)

        v_fwd = forward_speed_from_body_velocity(root_lin_vel_b, axis=self.stage_cfg.command.forward_axis)
        yaw_rate = root_ang_vel_b[:, 2]

        contact_xyz = self._contact_force_xyz()
        contact_binary, _ = contact_binary_and_force_mag(contact_xyz, self.stage_cfg.reward.contact_force_threshold_n)
        leg_phase = self._vectorized_leg_phase()

        reward, terms = compute_stage_a_reward(
            v_fwd_body=v_fwd,
            v_cmd=self.v_cmd,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
            roll=roll,
            pitch=pitch,
            root_lin_vel_b=root_lin_vel_b,
            action_residual=self.action_residual,
            prev_action_residual=self.prev_action_residual,
            contact_binary=contact_binary,
            leg_phase=leg_phase,
            reward_cfg=self.stage_cfg.reward,
        )
        if self.stage_cfg.debug.assert_finite_reward and not torch.isfinite(reward).all():
            raise RuntimeError("NaN/Inf in reward.")

        self.last_rewards = reward
        self.last_reward_terms = terms

        update_episode_stats(
            self.episode_stats,
            env_ids=None,
            v_cmd=self.v_cmd,
            v_fwd=v_fwd,
            yaw=heading_error,
            yaw_rate=yaw_rate,
            roll=roll,
            pitch=pitch,
            vertical_vel=root_lin_vel_b[:, 2],
            base_z=root_pos_w[:, 2],
            contact_continuity=support_continuity(
                contact_binary, self.stage_cfg.reward.support_continuity_min_contacts
            ),
            diag_asym=diag_support_asymmetry(contact_binary),
            scuff=terms["p_scuff"],
            action_residual=self.action_residual,
            action_rate=self.action_residual - self.prev_action_residual,
            joint_vel=self.robot.data.joint_vel,
            reward_total=reward,
        )

        toe_pos_b, toe_vel_b = self._toe_pos_vel_body()
        self._maybe_capture_trace(
            v_fwd=v_fwd,
            yaw=heading_error,
            yaw_rate=yaw_rate,
            roll=roll,
            pitch=pitch,
            base_z=root_pos_w[:, 2],
            contact_binary=contact_binary,
            toe_pos_b=toe_pos_b,
            toe_vel_b=toe_vel_b,
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat_w = self._root_quat_w()
        root_pos_w = self._root_pos_w()
        roll, pitch, yaw = quat_to_euler_xyz(quat_w)
        heading_error = self._heading_error(yaw)

        time_out = self.episode_time >= self.stage_cfg.termination.episode_length_s
        low_base = root_pos_w[:, 2] < self.stage_cfg.termination.min_base_height_m
        bad_roll = torch.abs(roll) > self.stage_cfg.termination.max_abs_roll_rad
        bad_pitch = torch.abs(pitch) > self.stage_cfg.termination.max_abs_pitch_rad
        bad_heading = torch.abs(heading_error) > self.stage_cfg.termination.max_abs_heading_err_rad

        nan_mask = ~torch.isfinite(self.last_obs).all(dim=-1)
        if self.stage_cfg.termination.nan_terminate:
            nan_mask |= ~torch.isfinite(self.last_rewards)

        terminated = low_base | bad_roll | bad_pitch | bad_heading | nan_mask
        done_mask = terminated | time_out

        reason = torch.full((self.num_envs,), DONE_RUNNING, device=self.device, dtype=torch.long)
        reason = torch.where(low_base, torch.full_like(reason, DONE_LOW_BASE), reason)
        reason = torch.where(bad_roll, torch.full_like(reason, DONE_ROLL), reason)
        reason = torch.where(bad_pitch, torch.full_like(reason, DONE_PITCH), reason)
        reason = torch.where(bad_heading, torch.full_like(reason, DONE_HEADING), reason)
        reason = torch.where(nan_mask, torch.full_like(reason, DONE_NAN), reason)
        timeout_only = time_out & (~terminated)
        reason = torch.where(timeout_only, torch.full_like(reason, DONE_TIMEOUT), reason)

        self.last_done_reason = torch.where(done_mask, reason, self.last_done_reason)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        # Archive summaries for any envs that are being reset *after* they have already
        # collected at least one control step. This avoids logging the very first startup reset
        # as a fake completed episode.
        active_mask = self.episode_stats["steps"][env_ids] > 0.0
        if torch.any(active_mask):
            done_env_ids = env_ids[active_mask]
            done_reasons = self.last_done_reason[done_env_ids]
            current_time_s = self.episode_time[done_env_ids].detach().clone()
            completed = build_episode_summary(
                self.episode_stats,
                done_env_ids,
                done_reasons,
                self.stage_cfg.termination.episode_length_s,
                current_time_s=current_time_s,
                success_heading_thresh_rad=self.stage_cfg.termination.success_max_heading_err_rad,
                success_tilt_rms_thresh_rad=self.stage_cfg.termination.success_max_tilt_rms_rad,
                success_vertical_bounce_rms_thresh_mps=self.stage_cfg.termination.success_max_vertical_bounce_rms_mps,
            )
            wall_time_s = time.time()
            with self._completed_episode_lock:
                for row in completed:
                    row["completed_episode_index"] = int(self._completed_episode_counter)
                    row["completed_wall_time_s"] = float(wall_time_s)
                    self.completed_episode_summaries.append(row)
                    self._completed_episode_counter += 1

        self._sample_command_speed(env_ids)
        self.gait_t[env_ids] = 0.0
        self.group_a_phase[env_ids] = 0.0
        self.group_b_phase[env_ids] = 0.5
        self.episode_time[env_ids] = 0.0
        self.prev_action_residual[env_ids] = 0.0
        self.action_residual[env_ids] = 0.0
        self.action_after_rate_limit[env_ids] = 0.0
        self.gait_param_state[env_ids] = 0.0
        self.gait_param_effective[env_ids] = self._gait_param_default.unsqueeze(0)
        self.last_done_reason[env_ids] = DONE_RUNNING
        self.last_rewards[env_ids] = 0.0
        reset_episode_stats(self.episode_stats, env_ids)
        self._clear_trace(env_ids)

        env_origins = self.scene.env_origins[env_ids]
        root_pos = env_origins.clone()
        root_pos[:, 2] += self.stage_cfg.reset.root_height_m
        if self.stage_cfg.reset.xy_jitter_m > 0.0:
            root_pos[:, :2] += self.stage_cfg.reset.xy_jitter_m * (
                2.0 * torch.rand(len(env_ids), 2, device=self.device) - 1.0
            )

        yaw_jitter = self.stage_cfg.reset.yaw_jitter_rad * (
            2.0 * torch.rand(len(env_ids), device=self.device) - 1.0
        )
        root_quat = yaw_to_quat_wxyz(yaw_jitter)
        root_lin_vel = self.stage_cfg.reset.lin_vel_jitter_mps * (
            2.0 * torch.rand(len(env_ids), 3, device=self.device) - 1.0
        )
        root_ang_vel = self.stage_cfg.reset.ang_vel_jitter_radps * (
            2.0 * torch.rand(len(env_ids), 3, device=self.device) - 1.0
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if self.stage_cfg.reset.joint_jitter_rad > 0.0:
            joint_pos += self.stage_cfg.reset.joint_jitter_rad * (2.0 * torch.rand_like(joint_pos) - 1.0)

        self.q_baseline[env_ids] = joint_pos
        self.q_final[env_ids] = joint_pos
        self.heading_ref_yaw[env_ids] = yaw_jitter

        self._write_reset_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

        super()._reset_idx(env_ids)

    def _write_reset_state(
        self,
        *,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Write the full reset state without pose/velocity overwrite bugs."""
        if hasattr(self.robot, "write_root_pose_to_sim") and hasattr(self.robot, "write_root_velocity_to_sim"):
            pose = torch.cat([root_pos, root_quat], dim=-1)
            vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
            self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
        elif hasattr(self.robot, "write_root_state_to_sim"):
            state = torch.zeros(len(env_ids), 13, device=self.device)
            state[:, 0:3] = root_pos
            state[:, 3:7] = root_quat
            state[:, 7:10] = root_lin_vel
            state[:, 10:13] = root_ang_vel
            self.robot.write_root_state_to_sim(state, env_ids=env_ids)
        else:
            raise RuntimeError("Robot has no supported root state writer.")

        if hasattr(self.robot, "write_joint_state_to_sim"):
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        elif hasattr(self.robot, "write_joint_position_to_sim") and hasattr(self.robot, "write_joint_velocity_to_sim"):
            self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
        else:
            raise RuntimeError("Robot has no supported joint state writer.")

    # ------------------------------------------------------------------
    # Debug / logging interface
    # ------------------------------------------------------------------
    def get_debug_step_dict(self, env_ids: Sequence[int] | None = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = [0]
        env_id = int(env_ids[0])
        quat_w = self._root_quat_w()[env_id : env_id + 1]
        roll, pitch, yaw_world = quat_to_euler_xyz(quat_w)
        heading_error = self._heading_error(yaw_world)
        # Same forward-axis convention as the main reward path: BODY +Y is forward.
        v_fwd = forward_speed_from_body_velocity(
            self._root_lin_vel_b()[env_id : env_id + 1], axis=self.stage_cfg.command.forward_axis
        )
        root_pos = self._root_pos_w()[env_id]
        zeros_like_reward = torch.zeros_like(self.last_rewards)
        return {
            "t": float(self.episode_time[env_id].item()),
            "env_id": env_id,
            "ep_step": int(self.episode_stats["steps"][env_id].item()),
            "v_cmd": float(self.v_cmd[env_id, 0].item()),
            "v_fwd_body": float(v_fwd[0].item()),
            "yaw": float(heading_error[0].item()),
            "yaw_world": float(yaw_world[0].item()),
            "yaw_rate": float(self._root_ang_vel_b()[env_id, 2].item()),
            "roll": float(roll[0].item()),
            "pitch": float(pitch[0].item()),
            "base_z": float(root_pos[2].item()),
            "reward_total": float(self.last_rewards[env_id].item()),
            "reward_track": float(self.last_reward_terms.get("r_track", zeros_like_reward)[env_id].item()),
            "reward_alive": float(self.last_reward_terms.get("r_alive", zeros_like_reward)[env_id].item()),
            "pen_yaw": float(self.last_reward_terms.get("p_yaw", zeros_like_reward)[env_id].item()),
            "pen_yaw_rate": float(self.last_reward_terms.get("p_yaw_rate", zeros_like_reward)[env_id].item()),
            "pen_rp": float(self.last_reward_terms.get("p_rp", zeros_like_reward)[env_id].item()),
            "pen_bounce": float(self.last_reward_terms.get("p_bounce", zeros_like_reward)[env_id].item()),
            "pen_scuff": float(self.last_reward_terms.get("p_scuff", zeros_like_reward)[env_id].item()),
            "pen_asym": float(self.last_reward_terms.get("p_diag", zeros_like_reward)[env_id].item()),
            "td_ratio": float(self.gait_param_effective[env_id, 0].item()),
            "shoulder_amp": float(self.gait_param_effective[env_id, 1].item()),
            "knee_lift_amp": float(self.gait_param_effective[env_id, 2].item()),
            "done_reason": DONE_REASON_STR.get(int(self.last_done_reason[env_id].item()), "running"),
        }

    def get_episode_summary_dict(self, env_ids: Sequence[int] | None = None) -> list[Dict[str, Any]]:
        if env_ids is None:
            env_ids = [0]
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        done_reasons = self.last_done_reason[ids]
        current_time_s = self.episode_time[ids].detach().clone()
        return build_episode_summary(
            self.episode_stats,
            ids,
            done_reasons,
            self.stage_cfg.termination.episode_length_s,
            current_time_s=current_time_s,
            success_heading_thresh_rad=self.stage_cfg.termination.success_max_heading_err_rad,
            success_tilt_rms_thresh_rad=self.stage_cfg.termination.success_max_tilt_rms_rad,
            success_vertical_bounce_rms_thresh_mps=self.stage_cfg.termination.success_max_vertical_bounce_rms_mps,
        )

    def format_episode_summary(self, env_ids: Sequence[int] | None = None, prefix: str = "eval") -> str:
        summaries = self.get_episode_summary_dict(env_ids=env_ids)
        return "\n\n".join(format_episode_summary(s, prefix=prefix, episode_idx=i) for i, s in enumerate(summaries))

    def _normalize_trace_env_ids(self, env_ids=None, env_id: int | None = None) -> list[int]:
        """Normalize trace-selection arguments.

        Why this helper exists:
        - Earlier debug scripts called get_episode_trace_dict(env_ids=[...]) while the first
          generated env file only accepted env_id=0.
        - To keep later scripts robust and copy-paste friendly, we now accept either form.
        - For now the trace exporter still returns a single-env trace dict, because that is the
          format expected by save_trace_npz() and the copy-paste evaluation bundle.
        """
        if env_ids is not None:
            if isinstance(env_ids, slice):
                start = 0 if env_ids.start is None else int(env_ids.start)
                stop = min(self.num_envs, int(env_ids.stop) if env_ids.stop is not None else self.num_envs)
                step = 1 if env_ids.step is None else int(env_ids.step)
                ids = list(range(start, stop, step))
            elif isinstance(env_ids, torch.Tensor):
                ids = [int(x) for x in env_ids.detach().cpu().flatten().tolist()]
            elif isinstance(env_ids, (list, tuple)):
                ids = [int(x) for x in env_ids]
            else:
                ids = [int(env_ids)]
        elif env_id is not None:
            ids = [int(env_id)]
        else:
            ids = [0]
        if len(ids) == 0:
            ids = [0]
        return ids

    def get_episode_trace_dict(self, env_ids=None, env_id: int = 0) -> Dict[str, torch.Tensor]:
        ids = self._normalize_trace_env_ids(env_ids=env_ids, env_id=env_id)
        trace = self.trace_cache.get(int(ids[0]), {})
        out = {}
        for k, v in trace.items():
            if len(v) == 0:
                continue
            out[k] = torch.stack(v, dim=0)
        return out

    def save_episode_trace(self, path: str | Path, env_ids=None, env_id: int = 0) -> None:
        trace = self.get_episode_trace_dict(env_ids=env_ids, env_id=env_id)
        save_trace_npz(path, trace)

    def save_episode_summaries(self, root: str | Path, prefix: str = "eval") -> None:
        root = Path(root)
        summaries = self.get_episode_summary_dict(env_ids=list(range(min(4, self.num_envs))))
        save_summaries_csv(root / self.stage_cfg.log.summary_filename, summaries)
        save_human_summaries(root / self.stage_cfg.log.human_summary_filename, summaries, prefix=prefix)

    def make_copy_paste_eval_bundle(self, env_ids=None, env_id: int = 0) -> Dict[str, torch.Tensor]:
        return make_copy_paste_eval_bundle(self.get_episode_trace_dict(env_ids=env_ids, env_id=env_id))

    def pop_completed_episode_summaries(self) -> list[Dict[str, Any]]:
        """Return and clear the completed-episode archive.

        Why this method exists:
        - Training uses auto-resetting vectorized envs, so completed episodes can pass by quickly.
        - A background convergence monitor thread can call this method periodically and append the
          returned rows into long-horizon CSV files without interfering with training itself.
        """
        with self._completed_episode_lock:
            rows = list(self.completed_episode_summaries)
            self.completed_episode_summaries.clear()
        return rows

    def peek_completed_episode_count(self) -> int:
        with self._completed_episode_lock:
            return len(self.completed_episode_summaries)

    def _clear_trace(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            self.trace_cache[env_id] = {
                "time": [],
                "v_cmd": [],
                "v_fwd_body": [],
                "yaw": [],
                "yaw_rate": [],
                "roll": [],
                "pitch": [],
                "base_z": [],
                "contact_4": [],
                "toe_pos_body": [],
                "toe_vel_body": [],
                "q_baseline": [],
                "action_residual": [],
                "q_final": [],
                "gait_param_state": [],
                "gait_param_effective": [],
                "reward_total": [],
                "done_reason": [],
            }

    def _maybe_capture_trace(
        self,
        *,
        v_fwd: torch.Tensor,
        yaw: torch.Tensor,
        yaw_rate: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        base_z: torch.Tensor,
        contact_binary: torch.Tensor,
        toe_pos_b: torch.Tensor,
        toe_vel_b: torch.Tensor,
    ) -> None:
        for env_id in self._sample_trace_env_ids.tolist():
            if env_id not in self.trace_cache:
                self._clear_trace(torch.tensor([env_id], device=self.device))
            tr = self.trace_cache[env_id]
            tr["time"].append(self.episode_time[env_id].detach().clone())
            tr["v_cmd"].append(self.v_cmd[env_id].detach().clone())
            tr["v_fwd_body"].append(v_fwd[env_id].detach().clone())
            tr["yaw"].append(yaw[env_id].detach().clone())
            tr["yaw_rate"].append(yaw_rate[env_id].detach().clone())
            tr["roll"].append(roll[env_id].detach().clone())
            tr["pitch"].append(pitch[env_id].detach().clone())
            tr["base_z"].append(base_z[env_id].detach().clone())
            tr["contact_4"].append(contact_binary[env_id].detach().clone())
            tr["toe_pos_body"].append(toe_pos_b[env_id].detach().clone())
            tr["toe_vel_body"].append(toe_vel_b[env_id].detach().clone())
            tr["q_baseline"].append(self.q_baseline[env_id].detach().clone())
            tr["action_residual"].append(self.action_residual[env_id].detach().clone())
            tr["q_final"].append(self.q_final[env_id].detach().clone())
            tr["gait_param_state"].append(self.gait_param_state[env_id].detach().clone())
            tr["gait_param_effective"].append(self.gait_param_effective[env_id].detach().clone())
            tr["reward_total"].append(self.last_rewards[env_id].detach().clone())
            tr["done_reason"].append(self.last_done_reason[env_id].detach().clone())
