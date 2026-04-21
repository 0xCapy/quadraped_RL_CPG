from __future__ import annotations

"""Unified soft-ground residual RL entrypoint.

This rewrite focuses on three goals:
1) keep the soft-ground path in one file;
2) provide short, explicit stage logs for every major initialization step;
3) avoid hidden failure points when debugging Isaac Lab / RSL-RL startup.
"""

import argparse
import csv
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

from RLstage_A_cfg import RLstage_A_Config
from RLstage_A_log import (
    aggregate_episode_summaries,
    append_row_csv,
    append_rows_csv,
    save_trace_npz,
)

_DEFAULT_RUN_OUTDIR = (Path(__file__).resolve().parent / "tem_doc" / "soft").as_posix()
_DEFAULT_TRAIN_LOGDIR = "RLstage_soft_outputs/train"
_STAGE_LOG_PATH: Path | None = None


# -----------------------------------------------------------------------------
# Stage logger
# -----------------------------------------------------------------------------

def _set_stage_log_path(path: str | Path | None) -> None:
    global _STAGE_LOG_PATH
    _STAGE_LOG_PATH = None if path is None else Path(path)
    if _STAGE_LOG_PATH is not None:
        _STAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _stage_log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[soft-stage {stamp}] {msg}"
    print(line, flush=True)
    if _STAGE_LOG_PATH is not None:
        with _STAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _stage_exception(where: str, exc: BaseException) -> None:
    _stage_log(f"{where} | EXCEPTION: {type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    print(tb, flush=True)
    if _STAGE_LOG_PATH is not None:
        with _STAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(tb + "\n")


# -----------------------------------------------------------------------------
# Soft-ground config
# -----------------------------------------------------------------------------

@dataclass
class RLstage_SoftGroundCfg:
    enabled: bool = True
    size_m: float = 40.0
    color: tuple[float, float, float] = (0.32, 0.24, 0.18)

    static_friction: float = 1.15
    dynamic_friction: float = 1.05
    restitution: float = 0.0

    # Softer -> lower stiffness / lower damping.
    compliant_contact_stiffness: float = 200.0
    compliant_contact_damping: float = 100.0


@dataclass
class RLstage_Soft_Config(RLstage_A_Config):
    soft_ground: RLstage_SoftGroundCfg = field(default_factory=RLstage_SoftGroundCfg)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _make_rsl_wrapper(env):
    """Import and build the RSL-RL wrapper.

    Important debug note:
    RslRlVecEnvWrapper construction may call ``env.reset()`` internally. So if
    startup hangs here, the true root cause is often inside the environment reset
    path rather than inside PPO itself.
    """
    _stage_log("rsl_wrapper | enter")
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        _stage_log("rsl_wrapper | imported isaaclab_rl.rsl_rl")
        _stage_log("rsl_wrapper | constructing wrapper (may call env.reset internally)")
        wrapped = RslRlVecEnvWrapper(env)
        _stage_log("rsl_wrapper | constructed wrapper")
        return wrapped
    except Exception as exc1:
        _stage_log(f"rsl_wrapper | first path failed: {type(exc1).__name__}: {exc1}")

    try:
        from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
        _stage_log("rsl_wrapper | imported isaaclab_tasks wrapper")
        _stage_log("rsl_wrapper | constructing wrapper (may call env.reset internally)")
        wrapped = RslRlVecEnvWrapper(env)
        _stage_log("rsl_wrapper | constructed wrapper")
        return wrapped
    except Exception as exc2:
        raise ImportError(
            "Could not import RslRlVecEnvWrapper. Check your Isaac Lab / IsaacLab RL installation."
        ) from exc2


def _save_rows_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _list_checkpoint_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.rglob("model_*.pt")) + sorted(log_dir.rglob("checkpoint_*.pt"))


def _latest_mtime_ns(root: Path) -> int:
    latest = 0
    if not root.exists():
        return latest
    for path in root.rglob("*"):
        try:
            latest = max(latest, path.stat().st_mtime_ns)
        except OSError:
            pass
    return latest


def _train_heartbeat(log_dir: Path, stop_event: threading.Event, heartbeat_sec: float) -> None:
    t0 = time.time()
    _stage_log(f"train_heartbeat | started | heartbeat_sec={heartbeat_sec:.1f} | log_dir={log_dir}")
    last_ckpt_count = -1
    last_mtime = -1
    while not stop_event.wait(max(1.0, heartbeat_sec)):
        elapsed = time.time() - t0
        ckpts = _list_checkpoint_files(log_dir)
        latest_mtime = _latest_mtime_ns(log_dir)
        changed = latest_mtime != last_mtime
        latest_name = ckpts[-1].name if ckpts else "none"
        if len(ckpts) != last_ckpt_count or changed:
            _stage_log(
                "train_heartbeat | alive | "
                f"elapsed={elapsed/60.0:.1f} min | checkpoints={len(ckpts)} | latest={latest_name} | "
                f"logdir_changed={int(changed)}"
            )
        else:
            _stage_log(
                "train_heartbeat | alive | "
                f"elapsed={elapsed/60.0:.1f} min | checkpoints={len(ckpts)} | no new files yet"
            )
        last_ckpt_count = len(ckpts)
        last_mtime = latest_mtime




def _to_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _compact_episode_row(row: dict[str, float | int | str], ep_idx: int) -> dict[str, float | int | str]:
    return {
        'ep': int(ep_idx),
        'succ': int(_to_float(row.get('success', 0), 0.0)),
        'rsn': str(row.get('done_reason', '')),
        'T': _to_float(row.get('duration_sec', float('nan'))),
        'vfwd': _to_float(row.get('v_fwd_mean', float('nan'))),
        'vrmse': _to_float(row.get('v_fwd_rmse', float('nan'))),
        'yaw': _to_float(row.get('yaw_drift_abs_final', float('nan'))),
        'yr': _to_float(row.get('yaw_rate_abs_mean', float('nan'))),
        'tilt': _to_float(row.get('tilt_rms', float('nan'))),
        'bnc': _to_float(row.get('vertical_bounce_rms', float('nan'))),
        'scf': _to_float(row.get('swing_scuff_count', float('nan'))),
        'ret': _to_float(row.get('return_total', float('nan'))),
    }


def _compact_live_row(rows: list[dict[str, float | int | str]], pt_idx: int) -> dict[str, float | int | str]:
    def m(key: str) -> float:
        vals = [_to_float(r.get(key, float('nan'))) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')

    succ_vals = [_to_float(r.get('succ', 0.0), 0.0) for r in rows]
    succ = float(np.mean(succ_vals)) if succ_vals else float('nan')
    return {
        'pt': int(pt_idx),
        'ep': int(rows[-1]['ep']) if rows else -1,
        'n': int(len(rows)),
        'succ': succ,
        'T': m('T'),
        'vfwd': m('vfwd'),
        'vrmse': m('vrmse'),
        'yaw': m('yaw'),
        'yr': m('yr'),
        'tilt': m('tilt'),
        'bnc': m('bnc'),
        'scf': m('scf'),
        'ret': m('ret'),
    }


def _read_csv_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _build_dense_trend_rows(rows: list[dict[str, float | int | str]], stride_episodes: int, window_episodes: int) -> list[dict[str, float | int | str]]:
    rows = sorted(rows, key=lambda r: int(r['ep']))
    if not rows:
        return []
    stride = max(1, int(stride_episodes))
    window = max(1, int(window_episodes))
    out: list[dict[str, float | int | str]] = []
    pt = 0
    for end in range(1, len(rows) + 1):
        is_sample = (end == 1) or (end == len(rows)) or (end % stride == 0)
        if not is_sample:
            continue
        chunk = rows[max(0, end - window):end]
        out.append(_compact_live_row(chunk, pt_idx=pt))
        pt += 1
    return out


def _plot_training_trends(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    xs = [int(r['ep']) for r in rows]
    specs = [
        ('yaw', 'yaw'),
        ('tilt', 'tilt'),
        ('bnc', 'bounce'),
        ('vrmse', 'v rmse'),
        ('scf', 'scuff'),
        ('ret', 'return'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (key, title) in zip(axes.ravel(), specs):
        ys = [_to_float(r.get(key, float('nan'))) for r in rows]
        ax.plot(xs, ys, marker='o', linewidth=1.2, markersize=2.8)
        ax.set_title(title)
        ax.set_xlabel('episode')
        ax.grid(True, alpha=0.3)
    fig.suptitle('training trend', fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _finalize_training_outputs(log_dir: Path, env, stride_episodes: int, window_episodes: int) -> None:
    raw_csv = log_dir / 'training_episode_summaries.csv'
    conv_csv = log_dir / 'training_convergence.csv'

    existing_rows = _read_csv_rows(raw_csv)
    next_ep = 0
    if existing_rows:
        try:
            next_ep = int(existing_rows[-1]['ep']) + 1
        except Exception:
            next_ep = len(existing_rows)

    remaining_rows = env.pop_completed_episode_summaries()
    if remaining_rows:
        compact_remaining = [_compact_episode_row(r, next_ep + i) for i, r in enumerate(remaining_rows)]
        append_rows_csv(raw_csv, compact_remaining)

    compact_rows = _read_csv_rows(raw_csv)
    trend_rows = _build_dense_trend_rows(compact_rows, stride_episodes=stride_episodes, window_episodes=window_episodes)
    _save_rows_csv(conv_csv, trend_rows)
    _plot_training_trends(trend_rows, log_dir / 'training_trends.png')

    _stage_log(
        'train_finalize | ' +
        f'episodes={len(compact_rows)} | trend_points={len(trend_rows)} | ' +
        f'stride_ep={int(stride_episodes)} | window_ep={int(window_episodes)}'
    )


def _train_metrics_monitor(
    log_dir: Path,
    env,
    stop_event: threading.Event,
    metrics_sec: float,
    convergence_window: int,
) -> None:
    t0 = time.time()
    raw_csv = log_dir / 'training_episode_summaries.csv'
    conv_csv = log_dir / 'training_convergence.csv'
    rolling_rows: deque[dict[str, float | int | str]] = deque(maxlen=max(1, int(convergence_window)))
    total_completed = 0
    live_pt = 0

    _stage_log(
        'train_metrics | started | '
        f'metrics_sec={metrics_sec:.1f} | convergence_window={int(convergence_window)} | log_dir={log_dir}'
    )

    while not stop_event.wait(max(1.0, metrics_sec)):
        try:
            new_rows = env.pop_completed_episode_summaries()
        except Exception as exc:  # pragma: no cover
            _stage_log(f'train_metrics | pop_completed_episode_summaries failed: {type(exc).__name__}: {exc}')
            continue

        if new_rows:
            compact_new = [_compact_episode_row(r, total_completed + i) for i, r in enumerate(new_rows)]
            total_completed += len(compact_new)
            append_rows_csv(raw_csv, compact_new)
            for row in compact_new:
                rolling_rows.append(row)

        if not rolling_rows:
            _stage_log('train_metrics | alive | no completed episodes yet')
            continue

        conv_row = _compact_live_row(list(rolling_rows), pt_idx=live_pt)
        live_pt += 1
        append_row_csv(conv_csv, conv_row)

        _stage_log(
            'train_metrics | snapshot | '
            f'ep={int(conv_row.get("ep", -1))} | n={int(conv_row.get("n", 0))} | '
            f'yaw={_to_float(conv_row.get("yaw", float("nan"))):.4f} | '
            f'tilt={_to_float(conv_row.get("tilt", float("nan"))):.4f} | '
            f'bnc={_to_float(conv_row.get("bnc", float("nan"))):.4f} | '
            f'scf={_to_float(conv_row.get("scf", float("nan"))):.2f} | '
            f'succ={_to_float(conv_row.get("succ", float("nan"))):.3f}'
        )


# -----------------------------------------------------------------------------
# Deferred Isaac-Lab-dependent imports
# -----------------------------------------------------------------------------

def _load_soft_env_components() -> Tuple[type, Callable[[RLstage_Soft_Config, int | None], object]]:
    _stage_log("load_soft_env_components | start")

    from RLstage_A_env import RLstage_AEnv, RLstage_A_IsaacCfg, _STAGEA_RIGID_DISTAL_PATHS
    _stage_log("load_soft_env_components | imported RLstage_A_env")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, FrameTransformer, Imu
    _stage_log("load_soft_env_components | imported isaaclab sim/assets/sensors")

    def _spawn_soft_ground_plane(stage_cfg: RLstage_Soft_Config) -> None:
        try:
            mat = sim_utils.RigidBodyMaterialCfg(
                static_friction=stage_cfg.soft_ground.static_friction,
                dynamic_friction=stage_cfg.soft_ground.dynamic_friction,
                restitution=stage_cfg.soft_ground.restitution,
                compliant_contact_stiffness=(
                    stage_cfg.soft_ground.compliant_contact_stiffness if stage_cfg.soft_ground.enabled else 0.0
                ),
                compliant_contact_damping=(
                    stage_cfg.soft_ground.compliant_contact_damping if stage_cfg.soft_ground.enabled else 0.0
                ),
            )
            gp = sim_utils.GroundPlaneCfg(
                size=(stage_cfg.soft_ground.size_m, stage_cfg.soft_ground.size_m),
                color=stage_cfg.soft_ground.color,
                physics_material=mat,
            )
            if hasattr(gp, "func"):
                gp.func("/World/GroundPlane", gp)
                return
        except Exception:
            pass

        try:
            from omni.physx.scripts import physicsUtils

            physicsUtils.add_ground_plane(
                stage=None,
                planePath="/World/GroundPlane",
                axis="Z",
                size=stage_cfg.soft_ground.size_m,
                position=(0.0, 0.0, 0.0),
                color=stage_cfg.soft_ground.color,
            )
            return
        except Exception as exc:
            raise RuntimeError("Failed to spawn a soft ground plane.") from exc

    class RLstage_SoftEnv(RLstage_AEnv):
        cfg: RLstage_A_IsaacCfg

        def __init__(self, cfg: RLstage_A_IsaacCfg, render_mode: str | None = None, **kwargs):
            self.stage_cfg: RLstage_Soft_Config = cfg.stage_cfg
            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self) -> None:
            _spawn_soft_ground_plane(self.stage_cfg)
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

    def make_soft_env_cfg(stage_cfg: RLstage_Soft_Config, num_envs: int | None = None) -> RLstage_A_IsaacCfg:
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

    _stage_log("load_soft_env_components | done")
    return RLstage_SoftEnv, make_soft_env_cfg


# -----------------------------------------------------------------------------
# Mode implementations
# -----------------------------------------------------------------------------

def _run_mode(args) -> None:
    try:
        _stage_log("run_mode | start")
        from RLstage_A_obs import contact_binary_and_force_mag

        RLstage_SoftEnv, make_soft_env_cfg = _load_soft_env_components()

        cfg = RLstage_Soft_Config()
        cfg.train.num_envs = int(args.num_envs)
        cfg.train.headless = bool(getattr(args, "headless", False))
        cfg.soft_ground.compliant_contact_stiffness = float(args.ground_k)
        cfg.soft_ground.compliant_contact_damping = float(args.ground_d)
        _stage_log("run_mode | config built")

        env_cfg = make_soft_env_cfg(cfg)
        _stage_log("run_mode | create env")
        env = RLstage_SoftEnv(env_cfg)
        _stage_log("run_mode | env created")

        policy = None
        policy_mode = "baseline"
        if args.checkpoint:
            _stage_log("run_mode | wrap env")
            wrapped_env = _make_rsl_wrapper(env)
            _stage_log("run_mode | import OnPolicyRunner")
            from rsl_rl.runners import OnPolicyRunner

            runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
            _stage_log(f"run_mode | runner_cfg keys={sorted(runner_cfg.keys())}")
            runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=args.outdir, device=env.device)
            runner.load(args.checkpoint)
            policy = runner.get_inference_policy(device=env.device)
            policy_mode = "policy"
            _stage_log("run_mode | policy loaded")

        outdir = Path(args.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        obs, _ = env.reset()
        _stage_log("run_mode | env reset complete")

        rows: list[dict[str, float | int | str]] = []
        first_done_printed = False

        print("=" * 100)
        print(f"soft-run mode      : {policy_mode}")
        print(f"num_envs           : {env.num_envs}")
        print(f"steps              : {args.steps}")
        print(f"ground_k           : {cfg.soft_ground.compliant_contact_stiffness:.3f}")
        print(f"ground_d           : {cfg.soft_ground.compliant_contact_damping:.3f}")
        print(f"print_every        : {args.print_every}")
        print(f"outdir             : {outdir}")
        print("Note: this runner prints live debug values each N control steps and saves a CSV.")
        print("=" * 100)

        for step in range(int(args.steps)):
            with torch.no_grad():
                if policy is None:
                    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                else:
                    action = policy(obs)

            obs, rew, terminated, time_outs, extras = env.step(action)

            dbg = env.get_debug_step_dict(env_ids=[0])
            contact_xyz = env._contact_force_xyz()[0:1]
            contact_binary, contact_force_mag = contact_binary_and_force_mag(
                contact_xyz, cfg.reward.contact_force_threshold_n
            )
            toe_pos_b, toe_vel_b = env._toe_pos_vel_body()
            toe_pos_b0 = toe_pos_b[0]
            toe_vel_b0 = toe_vel_b[0]

            q_delta = env.q_final[0] - env.q_baseline[0]
            row = {
                "step": int(step),
                "t": float(dbg["t"]),
                "v_cmd": float(dbg["v_cmd"]),
                "v_fwd_body": float(dbg["v_fwd_body"]),
                "yaw": float(dbg["yaw"]),
                "yaw_rate": float(dbg["yaw_rate"]),
                "roll": float(dbg["roll"]),
                "pitch": float(dbg["pitch"]),
                "base_z": float(dbg["base_z"]),
                "reward_total": float(dbg["reward_total"]),
                "reward_track": float(dbg["reward_track"]),
                "reward_alive": float(dbg["reward_alive"]),
                "pen_yaw": float(dbg["pen_yaw"]),
                "pen_rp": float(dbg["pen_rp"]),
                "pen_scuff": float(dbg["pen_scuff"]),
                "pen_asym": float(dbg["pen_asym"]),
                "contact_count": int(contact_binary[0].sum().item()),
                "contact_force_mean": float(torch.mean(contact_force_mag[0]).item()),
                "contact_force_max": float(torch.max(contact_force_mag[0]).item()),
                "toe_z_min_b": float(torch.min(toe_pos_b0[:, 2]).item()),
                "toe_speed_mean_b": float(torch.mean(torch.linalg.norm(toe_vel_b0, dim=-1)).item()),
                "residual_norm": float(torch.linalg.norm(env.action_residual[0]).item()),
                "q_delta_norm": float(torch.linalg.norm(q_delta).item()),
                "action_abs_max": float(torch.max(torch.abs(env.action_residual[0])).item()),
                "done_reason": str(dbg["done_reason"]),
                "terminated": int(bool(terminated[0].item())),
                "timeout": int(bool(time_outs[0].item())),
            }
            rows.append(row)

            if step % max(1, int(args.print_every)) == 0:
                print(
                    f"[step {step:04d}] "
                    f"t={row['t']:.2f}s | cmd={row['v_cmd']:.2f} | vfwd={row['v_fwd_body']:.3f} | "
                    f"yaw={row['yaw']:+.3f} | yr={row['yaw_rate']:+.3f} | "
                    f"r/p=({row['roll']:+.3f},{row['pitch']:+.3f}) | z={row['base_z']:.3f} | "
                    f"contacts={row['contact_count']} | fmax={row['contact_force_max']:.2f} | "
                    f"toe_zmin={row['toe_z_min_b']:+.3f} | res={row['residual_norm']:.3f} | "
                    f"R={row['reward_total']:+.3f}"
                )

            if bool(terminated[0].item()) or bool(time_outs[0].item()):
                first_done_printed = True
                status = "timeout" if row["timeout"] else "terminated"
                print(
                    f"[episode_end env0] status={status} | reason={row['done_reason']} | "
                    f"t={row['t']:.2f}s | z={row['base_z']:.3f} | yaw={row['yaw']:+.3f} | "
                    f"vfwd={row['v_fwd_body']:.3f}"
                )
                print("[note] episode_end values can reflect the post-step auto-reset snapshot on some Isaac Lab versions.")

        prefix = "soft_policy" if policy is not None else "soft_baseline"
        csv_path = outdir / f"{prefix}_debug_env0.csv"
        _save_rows_csv(csv_path, rows)
        env.save_episode_trace(outdir / "trace_env0.npz", env_id=0)
        bundle = env.make_copy_paste_eval_bundle(env_id=0)
        save_trace_npz(outdir / "bundle_env0.npz", bundle)
        _stage_log("run_mode | saved outputs")

        if not first_done_printed:
            print("\nNo episode end was reached during the requested step window.")
            print("Use the live debug lines / CSV above to judge motion quality.")
    except BaseException as exc:
        _stage_exception("run_mode", exc)
        raise



def _train_mode(args) -> None:
    stop_event: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    metrics_thread: threading.Thread | None = None
    try:
        _stage_log("train_mode | start")
        RLstage_SoftEnv, make_soft_env_cfg = _load_soft_env_components()

        cfg = RLstage_Soft_Config()
        cfg.train.num_envs = int(args.num_envs)
        cfg.train.max_iterations = int(args.max_iterations)
        cfg.train.headless = bool(getattr(args, "headless", False))
        if getattr(args, "save_interval", None) is not None:
            cfg.train.save_interval = int(args.save_interval)
        cfg.soft_ground.compliant_contact_stiffness = float(args.ground_k)
        cfg.soft_ground.compliant_contact_damping = float(args.ground_d)
        _stage_log("train_mode | config built")

        env_cfg = make_soft_env_cfg(cfg)
        _stage_log("train_mode | create env")
        env = RLstage_SoftEnv(env_cfg)
        _stage_log("train_mode | env created")

        _stage_log("train_mode | wrap env")
        wrapped_env = _make_rsl_wrapper(env)
        _stage_log("train_mode | wrapped env created")

        _stage_log("train_mode | import OnPolicyRunner")
        from rsl_rl.runners import OnPolicyRunner

        runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
        _stage_log(f"train_mode | runner_cfg keys={sorted(runner_cfg.keys())}")
        _stage_log(f"train_mode | algorithm keys={sorted(runner_cfg['algorithm'].keys())}")
        _stage_log(f"train_mode | actor keys={sorted(runner_cfg['actor'].keys())}")
        _stage_log(f"train_mode | critic keys={sorted(runner_cfg['critic'].keys())}")

        log_dir = Path(args.logdir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 100)
        print("soft-train mode    : residual PPO on compliant soft ground")
        print(f"num_envs           : {cfg.train.num_envs}")
        print(f"max_iterations     : {cfg.train.max_iterations}")
        print(f"ground_k           : {cfg.soft_ground.compliant_contact_stiffness:.3f}")
        print(f"ground_d           : {cfg.soft_ground.compliant_contact_damping:.3f}")
        print(f"policy_obs_dim     : {env.num_obs}")
        print(f"action_dim         : {env.num_actions}")
        print(f"sim_dt             : {cfg.train.sim_dt_s:.6f}")
        print(f"decimation         : {cfg.action.decimation}")
        print(f"control_dt         : {cfg.train.sim_dt_s * cfg.action.decimation:.6f}")
        print(f"logdir             : {log_dir}")
        print(f"save_interval      : {cfg.train.save_interval}")
        print(f"heartbeat_sec      : {float(args.heartbeat_sec):.1f}")
        print(f"metrics_sec        : {float(args.metrics_sec):.1f}")
        print(f"convergence_window : {int(args.convergence_window)}")
        print(f"trend_stride_ep    : {int(args.trend_stride_episodes)}")
        print(f"trend_window_ep    : {int(args.trend_window_episodes)}")
        print("Note: RSL-RL may stay quiet for stretches. Heartbeat and training-metric lines below confirm the run is still alive.")
        print("=" * 100)

        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_train_heartbeat,
            args=(log_dir, stop_event, float(args.heartbeat_sec)),
            daemon=True,
        )
        _stage_log("train_mode | start heartbeat thread")
        heartbeat_thread.start()

        metrics_thread = threading.Thread(
            target=_train_metrics_monitor,
            args=(log_dir, env, stop_event, float(args.metrics_sec), int(args.convergence_window)),
            daemon=True,
        )
        _stage_log("train_mode | start metrics thread")
        metrics_thread.start()

        _stage_log("train_mode | create OnPolicyRunner")
        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(log_dir), device=env.device)
        _stage_log("train_mode | OnPolicyRunner created")

        _stage_log("train_mode | enter runner.learn")
        runner.learn(num_learning_iterations=cfg.train.max_iterations, init_at_random_ep_len=True)
        _stage_log("train_mode | runner.learn finished")

        ckpts = _list_checkpoint_files(log_dir)
        print("\nTraining finished.")
        print(f"checkpoint_count   : {len(ckpts)}")
        if ckpts:
            print(f"latest_checkpoint  : {ckpts[-1]}")

        # We intentionally do NOT write the old final_eval bundle here.
        # Why this is disabled:
        # 1) in this project, the training-side convergence CSV is now the primary
        #    monitoring artifact;
        # 2) the old final_eval output was often misleading / empty because it did
        #    not run a dedicated fixed-condition evaluation pass;
        # 3) keeping it would duplicate outputs without adding trustworthy signal.
        # Use training_convergence.csv for online monitoring, and use the separate
        # checkpoint-evaluation script for post-training analysis.
        _stage_log("train_mode | skip legacy final_eval output")
        _stage_log("train_mode | done")
    except BaseException as exc:
        _stage_exception("train_mode", exc)
        raise
    finally:
        if stop_event is not None:
            _stage_log("train_mode | stop heartbeat thread")
            stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)
        if metrics_thread is not None:
            metrics_thread.join(timeout=2.0)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified compliant soft-ground RL entrypoint: training and live run/debug."
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--mode", choices=("run", "train"), required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--outdir", type=str, default=_DEFAULT_RUN_OUTDIR)
    parser.add_argument("--logdir", type=str, default=_DEFAULT_TRAIN_LOGDIR)
    parser.add_argument("--ground_k", type=float, default=3500.0)
    parser.add_argument("--ground_d", type=float, default=180.0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--heartbeat_sec", type=float, default=20.0)
    # save_interval is exposed on the CLI because convergence analysis is much more informative
    # when we keep checkpoints more densely than the debug-first default of 100 iterations.
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--metrics_sec", type=float, default=20.0)
    parser.add_argument("--convergence_window", type=int, default=128)
    parser.add_argument("--trend_stride_episodes", type=int, default=2)
    parser.add_argument("--trend_window_episodes", type=int, default=8)
    return parser



def main() -> None:
    simulation_app = None
    try:
        parser = _build_parser()
        _stage_log("main | parser built")
        args = parser.parse_args()

        # Configure stage-log file as soon as mode/path are known.
        if args.mode == "train":
            _set_stage_log_path(Path(args.logdir) / "stage_debug.log")
        else:
            _set_stage_log_path(Path(args.outdir) / "stage_debug.log")
        _stage_log(f"main | args parsed | mode={args.mode}")

        if args.num_envs is None:
            args.num_envs = 16 if args.mode == "run" else 512
        _stage_log(f"main | resolved num_envs={args.num_envs}")

        from isaaclab.app import AppLauncher

        _stage_log("main | create AppLauncher")
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        _stage_log("main | AppLauncher created")

        if args.mode == "run":
            _run_mode(args)
        elif args.mode == "train":
            _train_mode(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except BaseException as exc:
        _stage_exception("main", exc)
        raise
    finally:
        if simulation_app is not None:
            _stage_log("main | close simulation_app")
            simulation_app.close()


if __name__ == "__main__":
    main()
