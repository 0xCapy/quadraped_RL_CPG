from __future__ import annotations

"""Logging helpers for Stage-A training, evaluation, and convergence tracking.

Why this file was upgraded
--------------------------
The earlier summary CSV was useful as a smoke-test artifact, but it was not yet
scientific enough for tracking *how* the policy improved during training.

This upgraded version keeps the original per-episode summaries and adds:
- explicit vertical-bounce tracking;
- aggregate helpers for checkpoint-by-checkpoint convergence analysis;
- append-style CSV helpers so a long training run can accumulate a timeline of
  metrics without rewriting the whole file every time.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


DONE_REASON_STR = {
    0: "running",
    1: "timeout",
    2: "low_base",
    3: "roll",
    4: "pitch",
    5: "nan",
    6: "exception",
    7: "heading",
}


def make_episode_stats(num_envs: int, device: torch.device | str) -> Dict[str, torch.Tensor]:
    d = str(device)
    return {
        "steps": torch.zeros(num_envs, device=d),
        "sum_v_cmd": torch.zeros(num_envs, device=d),
        "sum_v_fwd": torch.zeros(num_envs, device=d),
        "sum_v_err_sq": torch.zeros(num_envs, device=d),
        "sum_yaw_rate_abs": torch.zeros(num_envs, device=d),
        "final_yaw": torch.zeros(num_envs, device=d),
        "sum_roll_sq": torch.zeros(num_envs, device=d),
        "sum_pitch_sq": torch.zeros(num_envs, device=d),
        # Vertical bounce proxy: body-frame z velocity RMS.
        "sum_vertical_vel_sq": torch.zeros(num_envs, device=d),
        "min_base_z": torch.full((num_envs,), float("inf"), device=d),
        "sum_contact_continuity": torch.zeros(num_envs, device=d),
        "sum_diag_asym": torch.zeros(num_envs, device=d),
        "sum_scuff": torch.zeros(num_envs, device=d),
        "sum_residual_sq": torch.zeros(num_envs, device=d),
        "sum_residual_rate_sq": torch.zeros(num_envs, device=d),
        "sum_joint_vel_sq": torch.zeros(num_envs, device=d),
        "sum_return": torch.zeros(num_envs, device=d),
    }


def reset_episode_stats(stats: Dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
    for v in stats.values():
        v[env_ids] = 0.0
    stats["min_base_z"][env_ids] = float("inf")


def update_episode_stats(
    stats: Dict[str, torch.Tensor],
    *,
    env_ids: torch.Tensor | None,
    v_cmd: torch.Tensor,
    v_fwd: torch.Tensor,
    yaw: torch.Tensor,
    yaw_rate: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    vertical_vel: torch.Tensor,
    base_z: torch.Tensor,
    contact_continuity: torch.Tensor,
    diag_asym: torch.Tensor,
    scuff: torch.Tensor,
    action_residual: torch.Tensor,
    action_rate: torch.Tensor,
    joint_vel: torch.Tensor,
    reward_total: torch.Tensor,
) -> None:
    ids = env_ids if env_ids is not None else slice(None)
    stats["steps"][ids] += 1.0
    stats["sum_v_cmd"][ids] += v_cmd.squeeze(-1)
    stats["sum_v_fwd"][ids] += v_fwd
    stats["sum_v_err_sq"][ids] += (v_fwd - v_cmd.squeeze(-1)).square()
    stats["sum_yaw_rate_abs"][ids] += yaw_rate.abs()
    stats["final_yaw"][ids] = yaw
    stats["sum_roll_sq"][ids] += roll.square()
    stats["sum_pitch_sq"][ids] += pitch.square()
    stats["sum_vertical_vel_sq"][ids] += vertical_vel.square()
    stats["min_base_z"][ids] = torch.minimum(stats["min_base_z"][ids], base_z)
    stats["sum_contact_continuity"][ids] += contact_continuity
    stats["sum_diag_asym"][ids] += diag_asym
    stats["sum_scuff"][ids] += scuff
    stats["sum_residual_sq"][ids] += torch.mean(action_residual.square(), dim=-1)
    stats["sum_residual_rate_sq"][ids] += torch.mean(action_rate.square(), dim=-1)
    stats["sum_joint_vel_sq"][ids] += torch.mean(joint_vel.square(), dim=-1)
    stats["sum_return"][ids] += reward_total


def build_episode_summary(
    stats: Dict[str, torch.Tensor],
    env_ids: torch.Tensor,
    done_reasons: torch.Tensor,
    episode_length_s: float,
    current_time_s: torch.Tensor | None = None,
    success_heading_thresh_rad: float | None = None,
    success_tilt_rms_thresh_rad: float | None = None,
    success_vertical_bounce_rms_thresh_mps: float | None = None,
) -> List[Dict[str, float | int | str]]:
    steps = torch.clamp(stats["steps"][env_ids], min=1.0)
    out: List[Dict[str, float | int | str]] = []
    for i, env_id in enumerate(env_ids.tolist()):
        reason_code = int(done_reasons[i].item())
        if current_time_s is None:
            duration_sec = float(episode_length_s)
        else:
            duration_sec = float(current_time_s[i].item())
            if reason_code == 1:
                duration_sec = float(episode_length_s)
        duration_sec = float(min(duration_sec, episode_length_s))

        roll_rms = float(torch.sqrt(stats["sum_roll_sq"][env_id] / steps[i]).item())
        pitch_rms = float(torch.sqrt(stats["sum_pitch_sq"][env_id] / steps[i]).item())
        vertical_bounce_rms = float(torch.sqrt(stats["sum_vertical_vel_sq"][env_id] / steps[i]).item())
        tilt_rms = float(np.sqrt(roll_rms * roll_rms + pitch_rms * pitch_rms))
        heading_abs = float(abs(stats["final_yaw"][env_id].item()))

        success = int(reason_code == 1)
        if success and success_heading_thresh_rad is not None:
            success &= int(heading_abs <= float(success_heading_thresh_rad))
        if success and success_tilt_rms_thresh_rad is not None:
            success &= int(tilt_rms <= float(success_tilt_rms_thresh_rad))
        if success and success_vertical_bounce_rms_thresh_mps is not None:
            success &= int(vertical_bounce_rms <= float(success_vertical_bounce_rms_thresh_mps))

        summary = {
            "env_id": env_id,
            "success": int(success),
            "done_reason_code": reason_code,
            "done_reason": DONE_REASON_STR.get(reason_code, f"code_{reason_code}"),
            "duration_sec": duration_sec,
            "survival_ratio": float(duration_sec / max(1e-6, episode_length_s)),
            "v_cmd_mean": float((stats["sum_v_cmd"][env_id] / steps[i]).item()),
            "v_fwd_mean": float((stats["sum_v_fwd"][env_id] / steps[i]).item()),
            "v_fwd_rmse": float(torch.sqrt(stats["sum_v_err_sq"][env_id] / steps[i]).item()),
            "yaw_drift_abs_final": heading_abs,
            "yaw_rate_abs_mean": float((stats["sum_yaw_rate_abs"][env_id] / steps[i]).item()),
            "roll_rms": roll_rms,
            "pitch_rms": pitch_rms,
            "tilt_rms": tilt_rms,
            "vertical_bounce_rms": vertical_bounce_rms,
            "base_z_min": float(stats["min_base_z"][env_id].item()),
            "contact_continuity_score": float((stats["sum_contact_continuity"][env_id] / steps[i]).item()),
            "diag_support_asym_score": float((stats["sum_diag_asym"][env_id] / steps[i]).item()),
            "swing_scuff_count": float(stats["sum_scuff"][env_id].item()),
            "residual_rms": float(torch.sqrt(stats["sum_residual_sq"][env_id] / steps[i]).item()),
            "residual_rate_rms": float(torch.sqrt(stats["sum_residual_rate_sq"][env_id] / steps[i]).item()),
            "joint_vel_rms": float(torch.sqrt(stats["sum_joint_vel_sq"][env_id] / steps[i]).item()),
            "return_total": float(stats["sum_return"][env_id].item()),
        }
        out.append(summary)
    return out


def format_episode_summary(summary: Dict[str, float | int | str], prefix: str = "eval", episode_idx: int | None = None) -> str:
    tag = f"[{prefix} ep {episode_idx:04d}]" if episode_idx is not None else f"[{prefix}]"
    return "\n".join(
        [
            f"{tag} success={summary['success']} | reason={summary['done_reason']} | T={summary['duration_sec']:.2f}s",
            f"cmd: v={summary['v_cmd_mean']:.2f} m/s",
            (
                "track: "
                f"v_mean={summary['v_fwd_mean']:.2f} | "
                f"v_rmse={summary['v_fwd_rmse']:.3f} | "
                f"yaw_final={summary['yaw_drift_abs_final']:.3f} | "
                f"yaw_rate_abs={summary['yaw_rate_abs_mean']:.3f}"
            ),
            (
                "stability: "
                f"roll_rms={summary['roll_rms']:.3f} | "
                f"pitch_rms={summary['pitch_rms']:.3f} | "
                f"tilt_rms={summary['tilt_rms']:.3f} | "
                f"bounce_rms={summary['vertical_bounce_rms']:.3f} | "
                f"z_min={summary['base_z_min']:.3f}"
            ),
            (
                "contact: "
                f"continuity={summary['contact_continuity_score']:.3f} | "
                f"diag_asym={summary['diag_support_asym_score']:.3f} | "
                f"scuff={summary['swing_scuff_count']:.1f}"
            ),
            (
                "control: "
                f"residual_rms={summary['residual_rms']:.3f} | "
                f"residual_rate_rms={summary['residual_rate_rms']:.3f} | "
                f"joint_vel_rms={summary['joint_vel_rms']:.3f}"
            ),
            f"return: {summary['return_total']:.3f}",
        ]
    )


def save_summaries_csv(path: str | Path, summaries: List[Dict[str, float | int | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def append_rows_csv(path: str | Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
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


def append_row_csv(path: str | Path, row: Dict[str, float | int | str]) -> None:
    append_rows_csv(path, [row])


def save_human_summaries(path: str | Path, summaries: List[Dict[str, float | int | str]], prefix: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, summary in enumerate(summaries):
        lines.append(format_episode_summary(summary, prefix=prefix, episode_idx=i))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def save_trace_npz(path: str | Path, trace: Dict[str, torch.Tensor | np.ndarray | list]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for k, v in trace.items():
        if isinstance(v, torch.Tensor):
            data[k] = tensor_to_numpy(v)
        else:
            data[k] = np.array(v)
    np.savez_compressed(path, **data)


# -----------------------------------------------------------------------------
# Aggregate helpers for scientific evaluation / convergence curves
# -----------------------------------------------------------------------------

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


def aggregate_episode_summaries(
    summaries: Sequence[Dict[str, float | int | str]],
    *,
    prefix: str = "agg",
) -> Dict[str, float | int | str]:
    summaries = list(summaries)
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


def make_copy_paste_eval_bundle(trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keep = [
        "time",
        "v_cmd",
        "v_fwd_body",
        "yaw",
        "yaw_rate",
        "roll",
        "pitch",
        "base_z",
        "contact_4",
        "toe_pos_body",
        "toe_vel_body",
        "q_baseline",
        "action_residual",
        "q_final",
        "reward_total",
        "done_reason",
    ]
    return {k: trace[k] for k in keep if k in trace}
