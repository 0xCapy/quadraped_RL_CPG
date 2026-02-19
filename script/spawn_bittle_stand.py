# spawn_bittle_stand.py
from __future__ import annotations

import argparse

# Must create SimulationApp before importing any Omniverse/pxr modules.
from isaacsim import SimulationApp


def _spawn_ground_plane(stage) -> None:
    """
    Spawn a PhysX ground plane without relying on Isaac content paths.
    We inspect the runtime signature of add_ground_plane and pass what THIS version expects.
    """
    import inspect
    import omni.physx.scripts.physicsUtils as physx_utils
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World")

    fn = physx_utils.add_ground_plane
    sig = inspect.signature(fn)

    PATH = "/World/GroundPlane"
    AXIS = "Z"
    POS = Gf.Vec3f(0.0, 0.0, 0.0)
    SIZE = 1000.0
    COLOR = Gf.Vec3f(0.1, 0.1, 0.1)
    HEIGHT = 0.0
# Im not sure which var is needed in this version so i just try all of them (anyway it works)
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
        if "pos" in n or "trans" in n or "origin" in n:
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
            # If it's optional and unknown, skip it.
            if p.default is not inspect._empty:
                continue
            raise

        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(v)
        else:
            kwargs[p.name] = v

    fn(*args, **kwargs)


def _spawn_light(stage) -> None:
    """Create a simple distant light using UsdLux (not UsdGeom)."""
    from pxr import UsdLux, Gf

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    # Minimal attributes; safe defaults
    light.CreateIntensityAttr(3000.0)
    # Optional: point the light a bit
    xform = UsdLux.ShadowAPI(light)  # harmless; keeps compatibility with Kit
    _ = xform
    # If you want a direction, use an Xform; not required for Phase A.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run headless")
    args, _ = parser.parse_known_args()

    simulation_app = SimulationApp({"headless": args.headless})

    # Safe to import after SimulationApp is up.
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import Articulation
    from bittle_cfg import BITTLE_CFG

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)  # no 'substeps' in your version
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    stage = omni.usd.get_context().get_stage()

    _spawn_ground_plane(stage)
    _spawn_light(stage)

    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"
    robot = Articulation(cfg=cfg)

    sim.reset()

    sim.step()
    robot.update(sim.get_physics_dt())

    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.reset()
# Let the robot settle onto the ground before holding targets
    # --- Timing ---
    dt = sim.get_physics_dt()

    # Read current joint state as ramp start (no kick)
    q0 = robot.data.joint_pos.clone()
    ramp_time = 2.5  # seconds

    # --- Settle on the ground before the main hold loop ---
    settle_time = 0.6  # seconds
    settle_steps = int(settle_time / dt)

    # During settle, already track the stand pose (with a gentle ramp)
    for k in range(settle_steps):
        t_set = k * dt
        alpha = min(1.0, t_set / ramp_time)
        q_tgt = (1.0 - alpha) * q0 + alpha * default_q
        robot.set_joint_position_target(q_tgt)

        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # --- Main hold loop (10 seconds) ---
    t = 0.0
    T_END = 10.0
    while simulation_app.is_running() and t < T_END:
        alpha = min(1.0, (t + settle_time) / ramp_time)
        q_tgt = (1.0 - alpha) * q0 + alpha * default_q
        robot.set_joint_position_target(q_tgt)

        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        t += dt


    simulation_app.close()


if __name__ == "__main__":
    main()
