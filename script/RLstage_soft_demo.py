from __future__ import annotations

"""Single-env deformable-pad visual demo.

Purpose
-------
Presentation/demo only. Training should stay on the lighter compliant-contact
soft-ground environment.

Modes
-----
1) Baseline only: no checkpoint
2) Baseline + trained residual policy: --checkpoint <path>

Design notes
------------
This version is deliberately more conservative than the previous one:
- it parses AppLauncher CLI args directly (so Windows users can pass --vulkan,
  and the physics device can be selected explicitly),
- it forces GPU simulation for deformables,
- it adds a dome light and explicit camera,
- it performs an explicit simulation reset AFTER spawning the deformable pad,
- it initializes the deformable object's nodal state / free-target buffer,
- it prints milestone messages so stalls can be localized quickly.
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Parse CLI first, then launch app (official Isaac Lab pattern)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Single-env deformable-pad visual demo for baseline or trained residual policy.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--outdir", type=str, default=(Path(__file__).resolve().parent / "tem_doc" / "soft_demo").as_posix())
parser.add_argument("--camera_eye", type=float, nargs=3, default=(1.8, 1.3, 0.9))
parser.add_argument("--camera_target", type=float, nargs=3, default=(0.0, 0.0, 0.15))

# Demo pad parameters
parser.add_argument("--pad_length", type=float, default=30)
parser.add_argument("--pad_width", type=float, default=30)
parser.add_argument("--pad_thickness", type=float, default=0.1)
parser.add_argument("--pad_youngs", type=float, default=3.0e3)
parser.add_argument("--pad_poisson", type=float, default=0.25)
parser.add_argument("--pad_contact_offset", type=float, default=0.002)
parser.add_argument("--pad_root_lift", type=float, default=0.06)
args_cli = parser.parse_args()

# Deformables require GPU simulation in Isaac Lab docs.
# If the launcher exposed a CPU device, override here to avoid silent misbehavior.
if hasattr(args_cli, "device") and isinstance(args_cli.device, str) and not args_cli.device.startswith("cuda"):
    print(f"[warn] deformables require GPU simulation; overriding device {args_cli.device!r} -> 'cuda:0'")
    args_cli.device = "cuda:0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports that depend on the app being alive
# -----------------------------------------------------------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from pxr import UsdLux

from RLstage_A_cfg import RLstage_A_Config
from RLstage_A_env import RLstage_AEnv, make_isaac_env_cfg
from RLstage_A_log import save_trace_npz
from RLstage_A_obs import contact_binary_and_force_mag


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


def _ensure_light_and_camera(env, eye, target) -> None:
    stage = env.sim.stage
    if not stage.GetPrimAtPath("/World/DemoDomeLight"):
        light = UsdLux.DomeLight.Define(stage, "/World/DemoDomeLight")
        light.CreateIntensityAttr(2500.0)
        light.CreateColorAttr((0.85, 0.85, 0.85))
    env.sim.set_camera_view(list(eye), list(target))


def _initialize_deformable_pad(pad: DeformableObject) -> None:
    """Initialize nodal state and explicitly release all nodes.

    Official deformable tutorial resets nodal state + writes a free-target buffer
    after the first simulation reset. We mimic that pattern here.
    """
    nodal_state = pad.data.default_nodal_state_w.clone()
    nodal_target = torch.zeros(pad.num_instances, nodal_state.shape[1], 4, device=pad.device)
    nodal_target[..., :3] = nodal_state[..., :3]
    nodal_target[..., 3] = 1.0  # 1 = free
    pad.write_nodal_state_to_sim(nodal_state)
    pad.write_nodal_kinematic_target_to_sim(nodal_target)
    pad.reset()


def main() -> None:
    outdir = Path(args_cli.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[milestone] building Stage-A env")
    cfg = RLstage_A_Config()
    cfg.train.num_envs = 1
    cfg.train.headless = bool(getattr(args_cli, "headless", False))
    cfg.reset.root_height_m = cfg.reset.root_height_m + args_cli.pad_root_lift

    env_cfg = make_isaac_env_cfg(cfg, num_envs=1)
    env_cfg.sim.device = args_cli.device
    env = RLstage_AEnv(env_cfg)

    _ensure_light_and_camera(env, args_cli.camera_eye, args_cli.camera_target)

    print("[milestone] spawning deformable pad")
    pad_cfg = DeformableObjectCfg(
        prim_path="/World/envs/env_0/DemoSoftPad",
        spawn=sim_utils.MeshCuboidCfg(
            size=(args_cli.pad_length, args_cli.pad_width, args_cli.pad_thickness),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=args_cli.pad_contact_offset,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.23, 0.16)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=args_cli.pad_poisson,
                youngs_modulus=args_cli.pad_youngs,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5 * args_cli.pad_thickness),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )
    demo_pad = DeformableObject(cfg=pad_cfg)

    print("[milestone] sim.reset after deformable spawn")
    env.sim.reset()
    demo_pad.update(float(env.cfg.sim.dt))
    _initialize_deformable_pad(demo_pad)

    print("[milestone] env.reset after deformable initialization")
    obs, _ = env.reset()

    # Small warm-up to let contacts settle.
    print("[milestone] warm-up steps")
    for _ in range(6):
        zero_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, *_ = env.step(zero_action)
        demo_pad.write_data_to_sim()
        demo_pad.update(float(env.cfg.sim.dt) * int(env.cfg.decimation))

    policy = None
    policy_mode = "baseline"
    if args_cli.checkpoint:
        print("[milestone] loading policy checkpoint")
        wrapped_env = _make_rsl_wrapper(env)
        from rsl_rl.runners import OnPolicyRunner

        runner_cfg = cfg.train.to_rsl_rl_dict(num_obs=env.num_obs, num_actions=env.num_actions)
        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(outdir), device=env.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        policy_mode = "policy"

    rows: list[dict[str, float | int | str]] = []
    ctrl_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)

    print("=" * 100)
    print("deformable demo mode :", policy_mode)
    print("outdir               :", outdir)
    print(f"device               : {args_cli.device}")
    print(f"pad size             : ({args_cli.pad_length:.3f}, {args_cli.pad_width:.3f}, {args_cli.pad_thickness:.3f}) m")
    print(f"pad youngs           : {args_cli.pad_youngs:.3e}")
    print(f"pad poisson          : {args_cli.pad_poisson:.3f}")
    print(f"root lift            : {args_cli.pad_root_lift:.3f}")
    print("note                 : if the viewport stays black on Windows, try re-running with --vulkan")
    print("=" * 100)

    for step in range(args_cli.steps):
        with torch.no_grad():
            if policy is None:
                action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            else:
                action = policy(obs)

        obs, rew, terminated, time_outs, extras = env.step(action)
        demo_pad.write_data_to_sim()
        demo_pad.update(ctrl_dt)

        dbg = env.get_debug_step_dict(env_ids=[0])
        contact_xyz = env._contact_force_xyz()[0:1]
        contact_binary, contact_force_mag = contact_binary_and_force_mag(contact_xyz, cfg.reward.contact_force_threshold_n)
        toe_pos_b, _ = env._toe_pos_vel_body()
        toe_pos_b0 = toe_pos_b[0]

        q_delta = env.q_final[0] - env.q_baseline[0]
        contact_count = int(contact_binary[0].sum().item())
        toe_z_min = float(torch.min(toe_pos_b0[:, 2]).item())
        residual_norm = float(torch.linalg.norm(env.action_residual[0]).item())
        force_max = float(torch.max(contact_force_mag[0]).item())

        nodal_pos = demo_pad.data.nodal_pos_w[0]
        nodal_def = demo_pad.data.default_nodal_state_w[0, :, :3]
        sink_all = torch.clamp(nodal_def[:, 2] - nodal_pos[:, 2], min=0.0)
        pad_sink_max = float(torch.max(sink_all).item())
        pad_sink_mean = float(torch.mean(sink_all).item())
        pad_top_z = float(torch.max(nodal_pos[:, 2]).item())

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
            "contact_count": contact_count,
            "contact_force_max": force_max,
            "toe_z_min_b": toe_z_min,
            "residual_norm": residual_norm,
            "q_delta_norm": float(torch.linalg.norm(q_delta).item()),
            "pad_sink_max": pad_sink_max,
            "pad_sink_mean": pad_sink_mean,
            "pad_top_z": pad_top_z,
            "done_reason": str(dbg["done_reason"]),
            "terminated": int(bool(terminated[0].item())),
            "timeout": int(bool(time_outs[0].item())),
        }
        rows.append(row)

        if step % max(1, args_cli.print_every) == 0:
            print(
                f"[step {step:04d}] "
                f"t={row['t']:.2f}s | vfwd={row['v_fwd_body']:.3f} | yaw={row['yaw']:+.3f} | "
                f"z={row['base_z']:.3f} | contacts={row['contact_count']} | "
                f"sink_max={row['pad_sink_max']:.4f} | sink_mean={row['pad_sink_mean']:.4f} | "
                f"pad_top={row['pad_top_z']:.4f} | res={row['residual_norm']:.3f}"
            )

        if bool(terminated[0].item()) or bool(time_outs[0].item()):
            status = "timeout" if bool(time_outs[0].item()) else "terminated"
            print(
                f"[episode_end env0] status={status} | reason={row['done_reason']} | "
                f"t={row['t']:.2f}s | z={row['base_z']:.3f} | yaw={row['yaw']:+.3f} | "
                f"sink_max={row['pad_sink_max']:.4f}"
            )
            break

    prefix = "deform_demo_policy" if policy is not None else "deform_demo_baseline"
    csv_path = outdir / f"{prefix}_env0.csv"
    _save_rows_csv(csv_path, rows)
    env.save_episode_trace(outdir / "trace_env0.npz", env_id=0)
    bundle = env.make_copy_paste_eval_bundle(env_id=0)
    save_trace_npz(outdir / "bundle_env0.npz", bundle)

    print("\nSaved files:")
    print(f"  debug csv : {csv_path}")
    print(f"  trace npz : {outdir / 'trace_env0.npz'}")
    print(f"  bundle npz: {outdir / 'bundle_env0.npz'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
