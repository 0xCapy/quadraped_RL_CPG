
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from RLstage_A_cfg import RLstage_A_Config
from RLstage_A_env import RLstage_AEnv, make_isaac_env_cfg


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RLstage_A checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--outdir", type=str, default="RLstage_A_outputs/eval")
    args = parser.parse_args()

    app_launcher, simulation_app = _build_app(headless=args.headless)
    try:
        cfg = RLstage_A_Config()
        cfg.train.num_envs = args.num_envs
        cfg.train.headless = args.headless

        env_cfg = make_isaac_env_cfg(cfg)
        env = RLstage_AEnv(env_cfg)
        wrapped_env = _make_rsl_wrapper(env)

        from rsl_rl.runners import OnPolicyRunner

        runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=args.outdir, device=env.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)

        obs, _ = env.reset()
        for step in range(args.steps):
            with torch.no_grad():
                # Important RSL-RL API note for this Isaac Lab version:
                # get_inference_policy() returns an actor that expects the full observation
                # dictionary keyed by observation-group name(s), e.g. {"policy": tensor}.
                # Passing only obs["policy"] would give a raw tensor, and the actor would then
                # try to index that tensor with the string key "policy", causing:
                #   IndexError: too many indices for tensor of dimension 2
                action = policy(obs)
            obs, rew, terminated, time_outs, extras = env.step(action)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        env.save_episode_summaries(outdir, prefix="eval")
        env.save_episode_trace(outdir / "trace_env0.npz", env_id=0)
        bundle = env.make_copy_paste_eval_bundle(env_id=0)
        from RLstage_A_log import save_trace_npz
        save_trace_npz(outdir / "bundle_env0.npz", bundle)
        print(env.format_episode_summary(env_ids=[0], prefix="eval"))

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
