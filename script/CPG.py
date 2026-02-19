
from __future__ import annotations

import argparse
import math
import torch

# Must create SimulationApp before importing any Omniverse/pxr modules.
from isaacsim import SimulationApp

class EveryN:
    def __init__(self, n: int):
        self.n = n
        self.i = 0
    def hit(self) -> bool:
        self.i += 1
        return (self.i % self.n) == 0

def _fmt(v):
    # v: tensor [4] or list
    return "[" + ", ".join([f"{float(x):+0.3f}" for x in v]) + "]"

def debug_cpg_step(
    step: int,
    dt: float,
    root_state_w: torch.Tensor,   # [E,13] typical
    phase_base: torch.Tensor,     # [E] or scalar
    phi_leg: torch.Tensor,        # [4]
    hip_tgt: torch.Tensor,        # [E,4]
    knee_tgt: torch.Tensor,       # [E,4]
    hip_act: torch.Tensor,        # [E,4] actual joint pos
    knee_act: torch.Tensor,       # [E,4]
    hip_tgt_clamped: torch.Tensor = None,   # [E,4] optional
    knee_tgt_clamped: torch.Tensor = None,  # [E,4] optional
):
    e = 0  # env0
    quat = root_state_w[e, 3:7]  # xyzw
    x,y,z,w = quat
    # yaw
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # phases per leg
    pb = phase_base[e] if phase_base.ndim > 0 else phase_base
    phase_leg = (pb + phi_leg) % (2*math.pi)
    s = torch.sin(phase_leg)
    c = torch.cos(phase_leg)

    print(f"\n=== step={step} dt={dt:.5f} yaw(rad)={float(yaw):+0.3f} ===")
    print(f"phase_base={float(pb):+0.3f}  sin(phase_leg)={_fmt(s)}  cos={_fmt(c)}")
    # symmetry checks (expect FL~RR, FR~RL, and FL~ -FR for trot)
    FL, FR, RL, RR = 0, 1, 2, 3
    print(f"trot-check: sinFL-sinRR={float(s[FL]-s[RR]):+0.3f}, sinFR-sinRL={float(s[FR]-s[RL]):+0.3f}, sinFL+sinFR={float(s[FL]+s[FR]):+0.3f}")

    print(f"hip_tgt ={_fmt(hip_tgt[e])}  knee_tgt ={_fmt(knee_tgt[e])}")
    print(f"hip_act ={_fmt(hip_act[e])}  knee_act ={_fmt(knee_act[e])}")
    print(f"hip_err ={_fmt(hip_tgt[e]-hip_act[e])}  knee_err ={_fmt(knee_tgt[e]-knee_act[e])}")

    if hip_tgt_clamped is not None:
        hc = (hip_tgt[e] != hip_tgt_clamped[e]).float().mean()
        kc = (knee_tgt[e] != knee_tgt_clamped[e]).float().mean()
        print(f"clamp-hit: hip={float(hc)*100:.1f}% knee={float(kc)*100:.1f}%")
def _spawn_ground_plane(stage) -> None:
    """Spawn a PhysX ground plane without relying on Isaac content paths."""
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
    COLOR = Gf.Vec3f(0.2, 0.2, 0.2)
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
            if p.default is not inspect._empty:
                continue
            raise
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(v)
        else:
            kwargs[p.name] = v

    fn(*args, **kwargs)


def _spawn_light(stage) -> None:
    """Create a basic distant light (your version uses UsdLux)."""
    from pxr import UsdLux

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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

    from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS

    # ------------------------
    # CPG parameters (trot)
    # ------------------------
    freq_hz = 2.0              # 1.5~3.0 typically
    w = 2.0 * math.pi * freq_hz

    # Amplitudes (radians): start small, then increase
    A_sh = 0.25                # shoulder swing amplitude
    A_kn = 0.35                # knee flex amplitude

    # Knee-shoulder phase bias inside one leg (helps foot clearance)
    knee_phase_bias = math.pi / 2.0

    # Safety joint limits (rough clamp). Adjust if your USD has different limits.
    SHOULDER_LIM = (-1.2, 1.2)
    KNEE_LIM = (-1.6, 1.6)

    # Soft ramp into gait
    gait_ramp_time = 2.0       # seconds

    # Settle time before gait starts
    settle_time = 0.6          # seconds

    # Run time
    T_END = 20.0               # seconds

    # ------------------------
    # Simulation setup
    # ------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    stage = omni.usd.get_context().get_stage()
    _spawn_ground_plane(stage)
    _spawn_light(stage)

    # Spawn robot
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"
    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    robot.update(sim.get_physics_dt())

    dt = sim.get_physics_dt()

    # Reset to default (stand) pose
    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.write_data_to_sim()

    # Warm-up a couple frames
    for _ in range(2):
        sim.step()
        robot.update(dt)

    robot.reset()

    # Build a mapping from joint name -> joint index
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}

    # Indices (match your printed list)
    idx = {
        "LB_sh": name_to_idx["left_back_shoulder_joint"],
        "LF_sh": name_to_idx["left_front_shoulder_joint"],
        "RB_sh": name_to_idx["right_back_shoulder_joint"],
        "RF_sh": name_to_idx["right_front_shoulder_joint"],
        "LB_kn": name_to_idx["left_back_knee_joint"],
        "LF_kn": name_to_idx["left_front_knee_joint"],
        "RB_kn": name_to_idx["right_back_knee_joint"],
        "RF_kn": name_to_idx["right_front_knee_joint"],
    }

    # Stand offsets from your config (make sure bittle_cfg.py uses the sign-fixed knee values)
    q_stand = default_q.clone()

    # ------------------------
    # Settle on ground (hold stand)
    # ------------------------
    settle_steps = int(settle_time / dt)
    for _ in range(settle_steps):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ------------------------
    # CPG trot loop
    # Trot: (LF + RB) in phase, (RF + LB) out of phase
    # ------------------------
    t = 0.0
    while simulation_app.is_running() and t < T_END:
# Trot phases: (LF + RB) in-phase, (RF + LB) anti-phase
        # --- Gait phase ---
        phi = w * t
        ramp = _clamp(t / gait_ramp_time, 0.0, 1.0)

        # Start from stand targets every step
        q_tgt = q_stand.clone()

        # Trot phases: (LF + RB) in-phase, (RF + LB) anti-phase
        sA = math.sin(phi)
        sB = math.sin(phi + math.pi)

        # Left/right mirror sign (your model's right joints are mirrored)
        L = +1.0
        R = -1.0

        # --- Shoulders (fore-aft swing) ---
        # Group A: LF, RB
        q_tgt[0, idx["LF_sh"]] += ramp * (L * A_sh * sA)
        q_tgt[0, idx["RB_sh"]] += ramp * (R * A_sh * sA)
        # Group B: RF, LB
        q_tgt[0, idx["RF_sh"]] += ramp * (R * A_sh * sB)
        q_tgt[0, idx["LB_sh"]] += ramp * (L * A_sh * sB)

        # --- Knees (lift during swing): half-wave rectified ---
        liftA = max(0.0, math.sin(phi + knee_phase_bias))
        liftB = max(0.0, math.sin(phi + math.pi + knee_phase_bias))

        q_tgt[0, idx["LF_kn"]] += ramp * (L * A_kn * liftA)
        q_tgt[0, idx["RB_kn"]] += ramp * (R * A_kn * liftA)
        q_tgt[0, idx["RF_kn"]] += ramp * (R * A_kn * liftB)
        q_tgt[0, idx["LB_kn"]] += ramp * (L * A_kn * liftB)



        # Optional: clamp to conservative limits
        for key, jlim in [
            ("LF_sh", SHOULDER_LIM), ("RF_sh", SHOULDER_LIM), ("LB_sh", SHOULDER_LIM), ("RB_sh", SHOULDER_LIM),
            ("LF_kn", KNEE_LIM), ("RF_kn", KNEE_LIM), ("LB_kn", KNEE_LIM), ("RB_kn", KNEE_LIM),
        ]:
            j = idx[key]
            q_tgt[0, j] = _clamp(float(q_tgt[0, j]), jlim[0], jlim[1])

        # Apply
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        t += dt

    simulation_app.close()


if __name__ == "__main__":
    main()
