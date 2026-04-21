from __future__ import annotations

"""Unified soft-ground visual runner with live debug output.

This file is intentionally closer to the user's current debugging workflow than
RLstage_A_eval.py. The goal is not only to run baseline / policy visually, but
also to print useful live signals and save a per-step CSV so the user can judge
whether the robot is really moving better on soft ground.

Important implementation note:
- Isaac-Lab-dependent imports are deferred until AFTER AppLauncher is created.
  Some Windows installs are sensitive to pxr / USD DLL initialization order.
"""

import argparse
import csv
from pathlib import Path

import torch

from RLstage_A_log import save_trace_npz
from RLstage_soft_cfg import RLstage_Soft_Config


_DEFAULT_OUTDIR = (Path(__file__).resolve().parent / "tem_doc" / "soft").as_posix()


def _build_app(headless: bool):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    cli_args, _ = parser.parse_known_args([])
    cli_args.headless = headless
    app_launcher = AppLauncher(cli_args)
    return app_launcher, app_launcher.app


def _make_rsl_wrapper(env):
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        return RslRlVecEnvWrapper(env)
    except Exception:
        pass
    try:
        from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
        return RslRlVecEnvWrapper(env)
    except Exception as exc:
        raise ImportError(
            "Could not import RslRlVecEnvWrapper. Check your Isaac Lab / IsaacLab RL installation."
        ) from exc


def _save_rows_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run baseline or trained residual policy on compliant soft ground.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--outdir", type=str, default=_DEFAULT_OUTDIR)
    parser.add_argument("--ground_k", type=float, default=3500.0)
    parser.add_argument("--ground_d", type=float, default=180.0)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()

    app_launcher, simulation_app = _build_app(headless=args.headless)
    try:
        # Defer Isaac-Lab-dependent imports until after AppLauncher exists.
        from RLstage_A_obs import contact_binary_and_force_mag
        from RLstage_soft_env import RLstage_SoftEnv, make_soft_env_cfg

        cfg = RLstage_Soft_Config()
        cfg.train.num_envs = args.num_envs
        cfg.train.headless = args.headless
        cfg.soft_ground.compliant_contact_stiffness = args.ground_k
        cfg.soft_ground.compliant_contact_damping = args.ground_d

        env_cfg = make_soft_env_cfg(cfg)
        env = RLstage_SoftEnv(env_cfg)

        policy = None
        policy_mode = "baseline"
        if args.checkpoint:
            wrapped_env = _make_rsl_wrapper(env)
            from rsl_rl.runners import OnPolicyRunner

            runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
            runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=args.outdir, device=env.device)
            runner.load(args.checkpoint)
            policy = runner.get_inference_policy(device=env.device)
            policy_mode = "policy"

        outdir = Path(args.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset()
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

        for step in range(args.steps):
            with torch.no_grad():
                if policy is None:
                    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                else:
                    action = policy(obs["policy"])

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
            contact_count = int(contact_binary[0].sum().item())
            toe_z_min = float(torch.min(toe_pos_b0[:, 2]).item())
            toe_speed_mean = float(torch.mean(torch.linalg.norm(toe_vel_b0, dim=-1)).item())
            residual_norm = float(torch.linalg.norm(env.action_residual[0]).item())
            q_delta_norm = float(torch.linalg.norm(q_delta).item())
            action_abs_max = float(torch.max(torch.abs(env.action_residual[0])).item())
            force_mean = float(torch.mean(contact_force_mag[0]).item())
            force_max = float(torch.max(contact_force_mag[0]).item())

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
                "contact_count": contact_count,
                "contact_force_mean": force_mean,
                "contact_force_max": force_max,
                "toe_z_min_b": toe_z_min,
                "toe_speed_mean_b": toe_speed_mean,
                "residual_norm": residual_norm,
                "q_delta_norm": q_delta_norm,
                "action_abs_max": action_abs_max,
                "done_reason": str(dbg["done_reason"]),
                "terminated": int(bool(terminated[0].item())),
                "timeout": int(bool(time_outs[0].item())),
            }
            rows.append(row)

            if step % max(1, args.print_every) == 0:
                print(
                    f"[step {step:04d}] "
                    f"t={row['t']:.2f}s | "
                    f"cmd={row['v_cmd']:.2f} | vfwd={row['v_fwd_body']:.3f} | "
                    f"yaw={row['yaw']:+.3f} | yr={row['yaw_rate']:+.3f} | "
                    f"r/p=({row['roll']:+.3f},{row['pitch']:+.3f}) | "
                    f"z={row['base_z']:.3f} | "
                    f"contacts={row['contact_count']} | fmax={row['contact_force_max']:.2f} | "
                    f"toe_zmin={row['toe_z_min_b']:+.3f} | "
                    f"res={row['residual_norm']:.3f} | "
                    f"R={row['reward_total']:+.3f}"
                )

            if bool(terminated[0].item()) or bool(time_outs[0].item()):
                first_done_printed = True
                reason = row["done_reason"]
                status = "timeout" if row["timeout"] else "terminated"
                print(
                    f"[episode_end env0] status={status} | reason={reason} | "
                    f"t={row['t']:.2f}s | z={row['base_z']:.3f} | yaw={row['yaw']:+.3f} | vfwd={row['v_fwd_body']:.3f}"
                )
                print("[note] episode_end values can reflect the post-step auto-reset snapshot on some Isaac Lab versions.")

        prefix = "soft_policy" if policy is not None else "soft_baseline"
        csv_path = outdir / f"{prefix}_debug_env0.csv"
        _save_rows_csv(csv_path, rows)

        env.save_episode_trace(outdir / "trace_env0.npz", env_id=0)
        bundle = env.make_copy_paste_eval_bundle(env_id=0)
        save_trace_npz(outdir / "bundle_env0.npz", bundle)

        print("\nSaved files:")
        print(f"  debug csv : {csv_path}")
        print(f"  trace npz : {outdir / 'trace_env0.npz'}")
        print(f"  bundle npz: {outdir / 'bundle_env0.npz'}")

        if not first_done_printed:
            print("\nNo episode end was reached during the requested step window.")
            print("Use the live debug lines / CSV above to judge motion quality.")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
