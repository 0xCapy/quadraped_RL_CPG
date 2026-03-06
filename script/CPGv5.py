# CPG_baseline_forward.py
# Forward baseline gait using duty-shaped (stance slow, swing fast) phase gait.
# - Not an online "optimizer": purely feedforward baseline with explicit stance/swing.
# - Intended to actually produce net forward motion (not sideways shuffling).

from omni.isaac.kit import SimulationApp  # MUST be first import
simulation_app = SimulationApp({"headless": False})

import math
import numpy as np

import omni.usd
from pxr import UsdLux

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS


def _quat_to_yaw(qw, qx, qy, qz) -> float:
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


def _smoothstep(x: float) -> float:
    # x in [0,1] -> smooth 0..1
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def _spawn_ground_plane(stage):
    # kwargs-only, signature adaptive (fits your planePath/color variant)
    import inspect
    import omni.physx.scripts.physicsUtils as physicsUtils

    fn = physicsUtils.add_ground_plane
    sig = inspect.signature(fn)

    plane_path = "/World/GroundPlane"
    color = (0.5, 0.5, 0.5)

    kwargs = {}
    for name in sig.parameters.keys():
        if name in ("stage", "usdStage"):
            kwargs[name] = stage
        elif name in ("planePath", "plane_path", "path"):
            kwargs[name] = plane_path
        elif name in ("color", "colour"):
            kwargs[name] = color
        elif name == "axis":
            kwargs[name] = "Z"
        elif name in ("size", "halfSize"):
            kwargs[name] = 50.0
        elif name == "position":
            kwargs[name] = (0.0, 0.0, 0.0)
        elif name == "normal":
            kwargs[name] = (0.0, 0.0, 1.0)
        elif name in ("staticFriction", "static_friction"):
            kwargs[name] = 1.0
        elif name in ("dynamicFriction", "dynamic_friction"):
            kwargs[name] = 1.0
        elif name == "restitution":
            kwargs[name] = 0.0

    if "planePath" in sig.parameters and "planePath" not in kwargs:
        kwargs["planePath"] = plane_path
    if "color" in sig.parameters and "color" not in kwargs:
        kwargs["color"] = color

    fn(**kwargs)


def _spawn_light(stage):
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(1500.0)
    light.CreateAngleAttr(0.5)



def main():
    # ---------------------------
    # Simulation
    # ---------------------------
    dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0")  # change to "cpu" if needed
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

    _spawn_ground_plane(stage)
    _spawn_light(stage)

    # ---------------------------
    # Robot
    # ---------------------------
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/bittle"  # match your USD hierarchy

    robot = Articulation(cfg)
    sim.reset()

    # warmup
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
    sim.step()
    robot.update(dt)

    joint_names = list(robot.data.joint_names)
    jmap = {name: i for i, name in enumerate(joint_names)}

    needed = [
        "left_front_shoulder_joint",
        "right_front_shoulder_joint",
        "left_back_shoulder_joint",
        "right_back_shoulder_joint",
        "left_front_knee_joint",
        "right_front_knee_joint",
        "left_back_knee_joint",
        "right_back_knee_joint",
    ]
    missing = [n for n in needed if n not in jmap]

    print("\n=== SANITY CHECK ===")
    print(f"[cfg.prim_path] {cfg.prim_path}")
    print(f"[joint_count] {len(joint_names)}")
    if missing:
        print(f"[ERROR] missing joints: {missing}")
        simulation_app.close()
        return
    print("[OK] joints found.\n")

    idx = {
        "LF_sh": jmap["left_front_shoulder_joint"],
        "RF_sh": jmap["right_front_shoulder_joint"],
        "LB_sh": jmap["left_back_shoulder_joint"],
        "RB_sh": jmap["right_back_shoulder_joint"],
        "LF_kn": jmap["left_front_knee_joint"],
        "RF_kn": jmap["right_front_knee_joint"],
        "LB_kn": jmap["left_back_knee_joint"],
        "RB_kn": jmap["right_back_knee_joint"],
    }

    # stand pose
    q_stand = robot.data.default_joint_pos.clone()
    for name, val in STAND_JOINT_POS.items():
        q_stand[0, jmap[name]] = float(val)

    # ---------------------------
    # Baseline gait parameters (tune these 4 first)
    freq_hz = 2.0
    beta = 0.72
    hip_amp = 0.22
    knee_lift_amp = 0.20
    dphi = -0.05 
    ramp_time = 1.2        # ramp in to avoid impulse at start

    # Trot phases [LF, RF, LB, RB]

    phase_bias = [
        0.0,              # LF
        math.pi + dphi,   # RF
        math.pi,          # LB
        0.0 + dphi        # RB
    ]

    # Mirror compensation consistent with your convention
    sh_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}
    kn_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}

    omega = 2.0 * math.pi * freq_hz

    # ---------------------------
    # Settle in stand
    # ---------------------------
    for _ in range(int(0.6 / dt)):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ---------------------------
    # Duty-shaped waveform:
    # stance: hip goes +hip_amp -> -hip_amp slowly over beta of cycle (propulsion)
    # swing : hip goes -hip_amp -> +hip_amp quickly over (1-beta) (recovery)
    # knee  : lift only during swing (half-sine bump)
    # ---------------------------
    def leg_commands(u01: float):
        """
        u01 in [0,1): phase within cycle.
        returns (hip_delta, knee_delta)
        """
        if u01 < beta:
            # stance
            s = _smoothstep(u01 / beta)
            hip = _lerp(-hip_amp, +hip_amp, s)       # slow backward sweep -> pushes body forward
            knee = 0.0                               # keep near stand in stance
        else:
            # swing
            s = _smoothstep((u01 - beta) / (1.0 - beta))
            hip = _lerp(+hip_amp, -hip_amp, s)       # quick forward recovery
            # knee lift bump (0->1->0)
            knee = knee_lift_amp * math.sin(math.pi * s)
        return hip, knee

    # ---------------------------
    # Run
    # ---------------------------
    run_s = 20.0
    steps = int(run_s / dt)
    print_every = int(0.5 / dt)

    print("=== RUN FORWARD BASELINE CPG (duty-shaped) ===")
    print("Columns: step | t | yaw | pos(x,y,z) | vel(vx,vy) | q_rms")
    print(f"[params] freq={freq_hz:.2f} beta={beta:.2f} hip_amp={hip_amp:.3f} knee_lift_amp={knee_lift_amp:.3f}\n")

    for k in range(steps):
        t = k * dt
        ramp = min(1.0, t / ramp_time)

        phi = omega * t

        # phases per leg
        u = []
        for b in phase_bias:
            th = (phi + b) % (2.0 * math.pi)
            u.append(th / (2.0 * math.pi))  # 0..1

        # command deltas
        hip = []
        knee = []
        for ui in u:
            h, kn = leg_commands(ui)
            hip.append(h)
            knee.append(kn)

        q_tgt = q_stand.clone()

        # shoulders (apply mirror sign)
        q_tgt[0, idx["LF_sh"]] = q_stand[0, idx["LF_sh"]] + ramp * (hip[0] * sh_sign["LF"])
        q_tgt[0, idx["RF_sh"]] = q_stand[0, idx["RF_sh"]] + ramp * (hip[1] * sh_sign["RF"])
        q_tgt[0, idx["LB_sh"]] = q_stand[0, idx["LB_sh"]] + ramp * (hip[2] * sh_sign["LB"])
        q_tgt[0, idx["RB_sh"]] = q_stand[0, idx["RB_sh"]] + ramp * (hip[3] * sh_sign["RB"])

        # knees (lift only during swing, apply sign)
        q_tgt[0, idx["LF_kn"]] = q_stand[0, idx["LF_kn"]] + ramp * (knee[0] * kn_sign["LF"])
        q_tgt[0, idx["RF_kn"]] = q_stand[0, idx["RF_kn"]] + ramp * (knee[1] * kn_sign["RF"])
        q_tgt[0, idx["LB_kn"]] = q_stand[0, idx["LB_kn"]] + ramp * (knee[2] * kn_sign["LB"])
        q_tgt[0, idx["RB_kn"]] = q_stand[0, idx["RB_kn"]] + ramp * (knee[3] * kn_sign["RB"])

        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if (k % print_every) == 0 or k == steps - 1:
            root_p = robot.data.root_pos_w[0].cpu().numpy()
            root_q = robot.data.root_quat_w[0].cpu().numpy()
            root_v = robot.data.root_lin_vel_w[0].cpu().numpy()
            yaw = _quat_to_yaw(root_q[0], root_q[1], root_q[2], root_q[3])

            q_act = robot.data.joint_pos[0].cpu().numpy()
            q_tar = q_tgt[0].cpu().numpy()
            q_rms = float(np.sqrt(np.mean((q_act - q_tar) ** 2)))

            print(
                f"{k:5d} | {t:6.2f} | yaw={yaw:+.3f} | "
                f"p=({root_p[0]:+.3f},{root_p[1]:+.3f},{root_p[2]:+.3f}) | "
                f"v=({root_v[0]:+.3f},{root_v[1]:+.3f}) | q_rms={q_rms:.4f}"
            )

    print("\n=== DONE ===")
    simulation_app.close()


if __name__ == "__main__":
    main()