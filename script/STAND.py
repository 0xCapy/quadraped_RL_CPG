# STAND_CALC.py
from __future__ import annotations

import argparse
import inspect
import math

from isaacsim import SimulationApp


def _spawn_ground_plane(stage) -> None:
    """Version-safe world ground plane spawn."""
    import omni.physx.scripts.physicsUtils as physx_utils
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World")

    fn = physx_utils.add_ground_plane
    sig = inspect.signature(fn)

    PATH = "/World/GroundPlane"
    AXIS = "Z"
    POS = Gf.Vec3f(0.0, 0.0, 0.0)
    SIZE = 1000.0
    COLOR = Gf.Vec3f(0.15, 0.15, 0.15)
    HEIGHT = 0.0

    def resolve_value(param_name: str):
        n = param_name.lower()
        if "stage" in n:
            return stage
        if "path" in n or "prim" in n:
            return PATH
        if "axis" in n:
            return AXIS
        if "height" in n:
            return HEIGHT
        if "pos" in n or "trans" in n or "origin" in n or "position" in n:
            return POS
        if "size" in n or "scale" in n or "extent" in n:
            return SIZE
        if "color" in n or "colour" in n:
            return COLOR
        raise RuntimeError(f"Unsupported add_ground_plane() parameter: '{param_name}'")

    args = []
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        try:
            v = resolve_value(p.name)
        except RuntimeError:
            if p.default is not inspect._empty:
                continue
            raise
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(v)
        else:
            kwargs[p.name] = v

    fn(*args, **kwargs)


def _spawn_light(stage) -> None:
    from pxr import UsdLux
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--dt", type=float, default=1.0 / 120.0)
    ap.add_argument("--hold_time", type=float, default=8.0)
    ap.add_argument("--ramp_time", type=float, default=2.0)
    ap.add_argument("--print_hz", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.02, help="desired clearance above ground for lowest body (m)")
    args, _ = ap.parse_known_args()

    simulation_app = SimulationApp({"headless": args.headless})

    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import Articulation

    from bittle_cfg import BITTLE_CFG

    # -----------------------------
    # Sim
    # -----------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=float(args.dt))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    stage = omni.usd.get_context().get_stage()
    _spawn_ground_plane(stage)
    _spawn_light(stage)

    # -----------------------------
    # Spawn robot
    # -----------------------------
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"

    init_pos = getattr(cfg.init_state, "pos", None)
    init_z = float(init_pos[2]) if init_pos is not None else 0.0

    print("\n========== STAND CALC ==========")
    print(f"[cfg] prim_path = {cfg.prim_path}")
    try:
        print(f"[cfg] usd_path  = {cfg.spawn.usd_path}")
    except Exception:
        print("[cfg] usd_path  = <unknown>")
    print(f"[cfg] init_state.pos = {init_pos}")

    robot = Articulation(cfg=cfg)

    # -----------------------------
    # Reset + initial state write
    # -----------------------------
    sim.reset()
    sim.step()

    dt = sim.get_physics_dt()
    robot.update(dt)

    # Write default state once (based on cfg.init_state)
    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.write_data_to_sim()

    # Warmup
    for _ in range(2):
        sim.step()
        robot.update(dt)
    robot.reset()

    # -----------------------------
    # Measure min_body_z and compute dz
    # -----------------------------
    body_pos = robot.data.body_pos_w[0]  # (num_bodies, 3)
    min_body_z = float(body_pos[:, 2].min().item())

    root = robot.data.root_state_w[0]
    root_z = float(root[2].item())

    margin = float(args.margin)
    dz = (margin - min_body_z) if (min_body_z < margin) else 0.0

    print(f"[measure] root_z      = {root_z:.4f}")
    print(f"[measure] min_body_z  = {min_body_z:.4f}")
    print(f"[measure] margin      = {margin:.4f}")
    print(f"[measure] dz_to_lift  = {dz:.4f}")

    if dz > 0.0:
        suggested_init_z = init_z + dz
        print(f"[suggest] cfg.init_state.pos.z should be ~ {suggested_init_z:.4f}  (current {init_z:.4f} + dz {dz:.4f})")
    else:
        print("[suggest] no lift needed; cfg.init_state.pos.z is already high enough.")

    # Apply lift in this run (so you can see stand immediately)
    if dz > 0.0:
        root_all = robot.data.root_state_w.clone()
        root_all[:, 2] += dz
        robot.write_root_pose_to_sim(root_all[:, :7])
        robot.write_root_velocity_to_sim(root_all[:, 7:])
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        body_pos2 = robot.data.body_pos_w[0]
        min_body_z2 = float(body_pos2[:, 2].min().item())
        root2 = robot.data.root_state_w[0]
        print(f"[after] root_z     = {float(root2[2].item()):.4f}")
        print(f"[after] min_body_z = {min_body_z2:.4f}")

    # -----------------------------
    # Hold stand
    # -----------------------------
    hold_time = float(args.hold_time)
    ramp_time = float(args.ramp_time)
    steps = int(max(1, math.ceil(hold_time / dt)))
    print_every = max(1, int(round((1.0 / max(1e-6, args.print_hz)) / dt)))

    q0 = robot.data.joint_pos.clone()
    for k in range(steps):
        t = k * dt
        alpha = min(1.0, t / ramp_time) if ramp_time > 1e-6 else 1.0
        q_tgt = (1.0 - alpha) * q0 + alpha * default_q

        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if (k % print_every) == 0:
            root = robot.data.root_state_w[0]
            root_z = float(root[2].item())
            vz = float(root[9].item())
            min_body_z = float(robot.data.body_pos_w[0][:, 2].min().item())
            print(f"[t={t:5.2f}s] root_z={root_z:.4f}  min_body_z={min_body_z:.4f}  vz={vz:.4f}")

    print("[done] closing SimulationApp.")
    simulation_app.close()


if __name__ == "__main__":
    main()