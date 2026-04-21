from __future__ import annotations

"""Checkpoint-sweep convergence evaluation for Stage-A soft-ground training.

This version fixes two practical issues seen in prior iterations:
1) It prints clear progress during each checkpoint evaluation.
2) It reuses a single OnPolicyRunner across checkpoints instead of repeatedly
   constructing a new runner on the same wrapped environment.

Why reuse the runner?
---------------------
In the user's logs, baseline evaluation completed, model_0 entered and printed
its model summary, and then the sweep advanced to model_25 before hanging with
no further model-summary print. That strongly suggests the stall occurs during
repeated OnPolicyRunner construction on an already-wrapped environment, not in
checkpoint scanning itself. Reusing one runner keeps the model architecture
constant while only swapping weights via runner.load(...).

Important observation API note
------------------------------
For this Isaac Lab / rsl_rl version, get_inference_policy() expects the full
observation dictionary keyed by observation-group name(s), e.g. {"policy": ...},
not the raw obs tensor obs["policy"].
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from RLstage_soft import RLstage_Soft_Config, _load_soft_env_components, _make_rsl_wrapper


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(?:model|checkpoint)_(\d+)\.pt$", path.name)
    idx = int(m.group(1)) if m else -1
    return idx, path.name


def _list_checkpoints(log_dir: Path) -> list[Path]:
    return sorted(
        list(log_dir.rglob("model_*.pt")) + list(log_dir.rglob("checkpoint_*.pt")),
        key=_checkpoint_sort_key,
    )


def _build_app_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a whole checkpoint series and export convergence metrics.")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--steps_per_episode", type=int, default=360)
    parser.add_argument("--ground_k", type=float, default=3500.0)
    parser.add_argument("--ground_d", type=float, default=180.0)
    parser.add_argument("--max_checkpoints", type=int, default=0)
    parser.add_argument("--include_baseline", action="store_true", default=False)
    parser.add_argument("--progress_every_steps", type=int, default=240)
    parser.add_argument("--progress_every_sec", type=float, default=2.0)
    parser.add_argument("--checkpoint_stride", type=int, default=1)
    return parser


def _save_rows_csv(path: str | Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_rows_csv(path: str | Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _array_metric(summaries: Sequence[Dict[str, float | int | str]], key: str) -> np.ndarray:
    vals = [float(s[key]) for s in summaries if key in s]
    return np.asarray(vals, dtype=np.float64)


def _stats_dict(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90.0)),
        f"{prefix}_max": float(np.max(values)),
    }


def _augment_episode_rows(rows: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    out: List[Dict[str, float | int | str]] = []
    for row in rows:
        r = dict(row)
        if "tilt_rms" not in r and "roll_rms" in r and "pitch_rms" in r:
            rr = float(r["roll_rms"])
            pr = float(r["pitch_rms"])
            r["tilt_rms"] = float(np.sqrt(rr * rr + pr * pr))
        # Prefer current key if present; otherwise create a placeholder NaN so aggregation stays consistent.
        if "vertical_bounce_rms" not in r:
            r["vertical_bounce_rms"] = float("nan")
        out.append(r)
    return out


def _aggregate_episode_summaries(
    summaries: Sequence[Dict[str, float | int | str]],
    *,
    prefix: str = "eval",
) -> Dict[str, float | int | str]:
    summaries = _augment_episode_rows(list(summaries))
    out: Dict[str, float | int | str] = {
        f"{prefix}_episode_count": int(len(summaries)),
    }
    if not summaries:
        return out

    success = _array_metric(summaries, "success")
    out[f"{prefix}_success_rate"] = float(np.mean(success)) if success.size else float("nan")

    for reason_name in sorted({str(s["done_reason"]) for s in summaries if "done_reason" in s}):
        count = sum(1 for s in summaries if str(s.get("done_reason")) == reason_name)
        out[f"{prefix}_done_{reason_name}"] = int(count)

    metric_keys = [
        "survival_ratio",
        "v_fwd_mean",
        "v_fwd_rmse",
        "yaw_drift_abs_final",
        "yaw_rate_abs_mean",
        "roll_rms",
        "pitch_rms",
        "tilt_rms",
        "vertical_bounce_rms",
        "contact_continuity_score",
        "diag_support_asym_score",
        "swing_scuff_count",
        "residual_rms",
        "residual_rate_rms",
        "joint_vel_rms",
        "return_total",
    ]
    for key in metric_keys:
        out.update(_stats_dict(_array_metric(summaries, key), f"{prefix}_{key}"))
    return out


def _plot_metric(rows: list[dict[str, float | int | str]], x_key: str, y_key: str, path: Path, title: str, ylabel: str) -> None:
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _render_progress(label: str, episodes_done: int, episodes_total: int, step: int, steps_total: int, elapsed_s: float) -> str:
    frac = 0.0 if episodes_total <= 0 else float(episodes_done) / float(episodes_total)
    frac = max(0.0, min(1.0, frac))
    width = 28
    n_full = int(round(frac * width))
    bar = "#" * n_full + "-" * (width - n_full)
    return (
        f"[eval-progress] {label} | eps {episodes_done}/{episodes_total} [{bar}] | "
        f"steps {step}/{steps_total} | elapsed={elapsed_s:.1f}s | pct={100.0 * frac:5.1f}%"
    )


def _evaluate_policy(
    env,
    policy,
    *,
    label: str,
    episodes: int,
    max_total_steps: int,
    progress_every_steps: int,
    progress_every_sec: float,
) -> list[dict[str, float | int | str]]:
    env.pop_completed_episode_summaries()
    obs, _ = env.reset()
    env.pop_completed_episode_summaries()

    summaries: list[dict[str, float | int | str]] = []
    t0 = time.time()
    last_print_t = -1.0e9
    print(_render_progress(label, 0, episodes, 0, max_total_steps, 0.0), flush=True)

    for step in range(1, max_total_steps + 1):
        with torch.no_grad():
            if policy is None:
                action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            else:
                action = policy(obs)
        obs, rew, terminated, time_outs, extras = env.step(action)
        new_rows = env.pop_completed_episode_summaries()
        if new_rows:
            summaries.extend(_augment_episode_rows(new_rows))
        now = time.time()
        should_print = False
        if step == 1 or step % max(1, progress_every_steps) == 0:
            should_print = True
        if (now - last_print_t) >= max(0.1, progress_every_sec):
            should_print = True
        if new_rows:
            should_print = True
        if should_print:
            print(_render_progress(label, min(len(summaries), episodes), episodes, step, max_total_steps, now - t0), flush=True)
            last_print_t = now
        if len(summaries) >= episodes:
            break
    print(f"[eval-done] {label} | collected {min(len(summaries), episodes)} episodes", flush=True)
    return summaries[:episodes]


def _save_scan_file(out_dir: Path, log_dir: Path, checkpoints: Sequence[Path], include_baseline: bool) -> Path:
    scan_path = out_dir / "checkpoint_scan.txt"
    with scan_path.open("w", encoding="utf-8") as f:
        f.write(f"log_dir={log_dir}\n")
        f.write(f"checkpoint_count={len(checkpoints)}\n")
        f.write(f"include_baseline={int(bool(include_baseline))}\n")
        for ckpt in checkpoints:
            f.write(str(ckpt) + "\n")
    return scan_path

def _select_core_mean_row(row: Dict[str, float | int | str]) -> Dict[str, float | int | str]:
    return {
        "ckpt_i": row.get("checkpoint_index"),
        "ckpt": row.get("checkpoint_name"),
        "n_ep": row.get("eval_episode_count"),
        "succ": row.get("eval_success_rate"),
        "surv": row.get("eval_survival_ratio_mean"),
        "vfwd": row.get("eval_v_fwd_mean_mean"),
        "vrmse": row.get("eval_v_fwd_rmse_mean"),
        "yaw": row.get("eval_yaw_drift_abs_final_mean"),
        "yrate": row.get("eval_yaw_rate_abs_mean_mean"),
        "roll": row.get("eval_roll_rms_mean"),
        "pitch": row.get("eval_pitch_rms_mean"),
        "tilt": row.get("eval_tilt_rms_mean"),
        "bounce": row.get("eval_vertical_bounce_rms_mean"),
        "cont": row.get("eval_contact_continuity_score_mean"),
        "asym": row.get("eval_diag_support_asym_score_mean"),
        "scuff": row.get("eval_swing_scuff_count_mean"),
        "ret": row.get("eval_return_total_mean"),
    }


def _select_core_episode_row(row: Dict[str, float | int | str]) -> Dict[str, float | int | str]:
    return {
        "series": row.get("series_label"),
        "ckpt_i": row.get("checkpoint_index"),
        "env": row.get("env_id"),
        "succ": row.get("success"),
        "reason": row.get("done_reason"),
        "T": row.get("duration_sec"),
        "vcmd": row.get("v_cmd_mean"),
        "vfwd": row.get("v_fwd_mean"),
        "vrmse": row.get("v_fwd_rmse"),
        "yaw": row.get("yaw_drift_abs_final"),
        "yrate": row.get("yaw_rate_abs_mean"),
        "tilt": row.get("tilt_rms"),
        "bounce": row.get("vertical_bounce_rms"),
        "cont": row.get("contact_continuity_score"),
        "asym": row.get("diag_support_asym_score"),
        "scuff": row.get("swing_scuff_count"),
        "ret": row.get("return_total"),
    }


def _refresh_outputs(out_dir: Path, rows: list[dict[str, float | int | str]], raw_episode_rows: list[dict[str, float | int | str]], include_baseline: bool) -> None:
    rows_sorted = sorted(rows, key=lambda r: (int(r["checkpoint_index"]), str(r["checkpoint_name"])))
    rows_compact = [_select_core_mean_row(r) for r in rows_sorted]
    raw_compact = [_select_core_episode_row(r) for r in raw_episode_rows]
    _save_rows_csv(out_dir / "checkpoint_convergence.csv", rows_compact)
    _save_rows_csv(out_dir / "checkpoint_episode_summaries.csv", raw_compact)

    plot_rows = [r for r in rows_compact if int(r["ckpt_i"]) >= 0]
    if include_baseline:
        plot_rows = rows_compact
    if plot_rows:
        _plot_metric(plot_rows, "ckpt_i", "yaw", out_dir / "yaw_convergence.png", "Yaw drift convergence", "yaw drift mean [rad]")
        _plot_metric(plot_rows, "ckpt_i", "tilt", out_dir / "tilt_convergence.png", "Tilt RMS convergence", "tilt RMS [rad]")
        _plot_metric(plot_rows, "ckpt_i", "bounce", out_dir / "bounce_convergence.png", "Vertical bounce convergence", "bounce RMS [m/s]")
        _plot_metric(plot_rows, "ckpt_i", "scuff", out_dir / "scuff_convergence.png", "Swing scuff convergence", "scuff count mean")
        _plot_metric(plot_rows, "ckpt_i", "succ", out_dir / "success_convergence.png", "Success-rate convergence", "success rate")


def main() -> None:
    parser = _build_app_parser()
    args = parser.parse_args()

    log_dir = Path(args.logdir).resolve()
    out_dir = Path(args.outdir).resolve() if args.outdir else (log_dir / "convergence_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        RLstage_SoftEnv, make_soft_env_cfg = _load_soft_env_components()
        cfg = RLstage_Soft_Config()
        cfg.train.num_envs = int(args.num_envs)
        cfg.train.headless = bool(getattr(args, "headless", False))
        cfg.soft_ground.compliant_contact_stiffness = float(args.ground_k)
        cfg.soft_ground.compliant_contact_damping = float(args.ground_d)

        env_cfg = make_soft_env_cfg(cfg)
        env = RLstage_SoftEnv(env_cfg)
        print("[eval] environment created", flush=True)
        wrapped_env = _make_rsl_wrapper(env)
        print("[eval] RSL-RL wrapper created", flush=True)

        from rsl_rl.runners import OnPolicyRunner

        runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
        checkpoints = _list_checkpoints(log_dir)
        stride = max(1, int(args.checkpoint_stride))
        if stride > 1:
            checkpoints = checkpoints[::stride]
        if args.max_checkpoints and args.max_checkpoints > 0:
            checkpoints = checkpoints[: int(args.max_checkpoints)]
        if not checkpoints and not args.include_baseline:
            raise FileNotFoundError(f"No checkpoints found under: {log_dir}")

        jobs = len(checkpoints) + (1 if args.include_baseline else 0)
        print(f"[eval] jobs={jobs} | checkpoints={len(checkpoints)} | include_baseline={int(bool(args.include_baseline))}", flush=True)
        if checkpoints:
            print(f"[eval] first_checkpoint={checkpoints[0]}", flush=True)
            print(f"[eval] last_checkpoint={checkpoints[-1]}", flush=True)
            preview = checkpoints[: min(8, len(checkpoints))]
            print(f"[eval] checkpoint_preview_count={len(preview)}", flush=True)
            for p in preview:
                print(f"[eval] checkpoint_preview={p}", flush=True)
        scan_path = _save_scan_file(out_dir, log_dir, checkpoints, bool(args.include_baseline))
        print(f"[eval] saved scan: {scan_path}", flush=True)

        rows: list[dict[str, float | int | str]] = []
        raw_episode_rows: list[dict[str, float | int | str]] = []

        if args.include_baseline:
            print(f"[eval] (1/{jobs}) baseline policy (zero residual)", flush=True)
            baseline_summaries = _evaluate_policy(
                env,
                None,
                label="baseline",
                episodes=int(args.episodes),
                max_total_steps=int(args.episodes) * int(args.steps_per_episode) * 2,
                progress_every_steps=int(args.progress_every_steps),
                progress_every_sec=float(args.progress_every_sec),
            )
            for r in baseline_summaries:
                r["series_label"] = "baseline"
                r["checkpoint_index"] = -1
            raw_episode_rows.extend(baseline_summaries)
            agg = _aggregate_episode_summaries(baseline_summaries, prefix="eval")
            row = {"checkpoint_index": -1, "checkpoint_name": "baseline_zero_residual"}
            row.update(agg)
            rows.append(row)
            _refresh_outputs(out_dir, rows, raw_episode_rows, bool(args.include_baseline))
            print(
                f"[eval] baseline saved | success_rate={row.get('eval_success_rate', float('nan')):.3f} | "
                f"yaw_mean={row.get('eval_yaw_drift_abs_final_mean', float('nan')):.4f} | "
                f"tilt_mean={row.get('eval_tilt_rms_mean', float('nan')):.4f}",
                flush=True,
            )

        # Construct the runner exactly once, then only swap checkpoint weights via runner.load(...).
        # This avoids repeated OnPolicyRunner construction on the same wrapped env, which is the
        # most likely source of the observed stall after model_0 in the user's logs.
        runner = None
        if checkpoints:
            print("[eval] creating reusable OnPolicyRunner once", flush=True)
            runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(out_dir), device=env.device)
            print("[eval] reusable OnPolicyRunner created", flush=True)

        start_job_idx = 2 if args.include_baseline else 1
        for i, ckpt in enumerate(checkpoints, start=start_job_idx):
            idx, _ = _checkpoint_sort_key(ckpt)
            print(f"[eval] ({i}/{jobs}) checkpoint={ckpt.name} | index={idx}", flush=True)
            assert runner is not None
            runner.load(str(ckpt))
            policy = runner.get_inference_policy(device=env.device)
            summaries = _evaluate_policy(
                env,
                policy,
                label=ckpt.name,
                episodes=int(args.episodes),
                max_total_steps=int(args.episodes) * int(args.steps_per_episode) * 2,
                progress_every_steps=int(args.progress_every_steps),
                progress_every_sec=float(args.progress_every_sec),
            )
            for r in summaries:
                r["series_label"] = ckpt.name
                r["checkpoint_index"] = idx
            raw_episode_rows.extend(summaries)
            agg = _aggregate_episode_summaries(summaries, prefix="eval")
            row = {"checkpoint_index": idx, "checkpoint_name": ckpt.name}
            row.update(agg)
            rows.append(row)
            _refresh_outputs(out_dir, rows, raw_episode_rows, bool(args.include_baseline))
            print(
                f"[eval] checkpoint saved | name={ckpt.name} | success_rate={row.get('eval_success_rate', float('nan')):.3f} | "
                f"yaw_mean={row.get('eval_yaw_drift_abs_final_mean', float('nan')):.4f} | "
                f"tilt_mean={row.get('eval_tilt_rms_mean', float('nan')):.4f} | "
                f"scuff_mean={row.get('eval_swing_scuff_count_mean', float('nan')):.2f}",
                flush=True,
            )

        print(f"saved: {out_dir / 'checkpoint_convergence.csv'}", flush=True)
        print(f"saved: {out_dir / 'checkpoint_episode_summaries.csv'}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
