
from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train RLstage_A straight-line residual PPO.")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default="RLstage_A_outputs/train")
    args = parser.parse_args()

    app_launcher, simulation_app = _build_app(headless=args.headless)
    try:
        cfg = RLstage_A_Config()
        cfg.train.num_envs = args.num_envs
        cfg.train.max_iterations = args.max_iterations
        cfg.train.headless = args.headless

        env_cfg = make_isaac_env_cfg(cfg)
        env = RLstage_AEnv(env_cfg)

        wrapped_env = _make_rsl_wrapper(env)

        from rsl_rl.runners import OnPolicyRunner

        runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
        log_dir = Path(args.logdir)
        log_dir.mkdir(parents=True, exist_ok=True)

        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(log_dir), device=env.device)
        runner.learn(num_learning_iterations=cfg.train.max_iterations, init_at_random_ep_len=True)

        env.save_episode_summaries(log_dir / "final_eval", prefix="train")
        env.save_episode_trace(log_dir / "final_eval" / "trace_env0.npz", env_id=0)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
