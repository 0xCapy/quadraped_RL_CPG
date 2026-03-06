# CPGv5_fixed_lr_trim.py
# Clean baseline trot CPG for Petoi Bittle in Isaac Lab
#
# Baseline design:
# - analytic phase trot (no coupling, no online PD/PI)
# - smooth knee lift
# - small stance press
# - fixed LEFT/RIGHT calibration term lr_trim
#
# Purpose:
# - get a clean, reproducible baseline CPG
# - remove systematic yaw / lateral drift as a calibration issue
# - leave higher-level adaptation to later residual RL

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
    """Yaw from quaternion (w, x, y, z)."""
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


def _spawn_ground_plane(stage):
    """Version-adaptive ground plane spawn using kwargs only."""
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
    # Simulation / scene
    # ---------------------------
    dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0")  # change to "cpu" if needed
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

    _spawn_ground_plane(stage)
    _spawn_light(stage)

    # ---------------------------
    # Robot spawn
    # ---------------------------
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/bittle"  # match your USD hierarchy screenshot

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

    # ---------------------------
    # Joint mapping
    # ---------------------------
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
    print(f"[joint_names] {joint_names}")
    if missing:
        print(f"[ERROR] missing joints: {missing}")
        simulation_app.close()
        return
    print("[OK] required joints found.\n")

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

    # ---------------------------
    # Stand pose
    # ---------------------------
    q_stand = robot.data.default_joint_pos.clone()
    for name, val in STAND_JOINT_POS.items():
        if name not in jmap:
            print(f"[ERROR] STAND_JOINT_POS key not found in joint_names: {name}")
            simulation_app.close()
            return
        q_stand[0, jmap[name]] = float(val)

    # ---------------------------
    # Baseline parameters
    # ---------------------------
    freq_hz = 2.2
    A_sh = 0.22
    A_kn = 0.18
    ramp_time = 1.5
    knee_stance_press = 0.02

    # fixed left-right calibration term
    # try -0.08 first based on your current drift direction
    lr_trim = -0.2

    # leg phase order: [LF, RF, LB, RB]
    phase_bias = [0.0, math.pi, math.pi, 0.0]

    # mirror compensation from your existing convention
    sh_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}
    kn_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}

    omega = 2.0 * math.pi * freq_hz

    # fixed left/right amplitudes
    A_left = A_sh * (1.0 - lr_trim)
    A_right = A_sh * (1.0 + lr_trim)

    # ---------------------------
    # Settle in stand pose
    # ---------------------------
    settle_s = 0.6
    settle_steps = int(settle_s / dt)
    for _ in range(settle_steps):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ---------------------------
    # Main run
    # ---------------------------
    run_s = 20.0
    steps = int(run_s / dt)
    print_every = int(0.5 / dt)

    print("=== RUN BASELINE CPG (fixed LR trim) ===")
    print(
        "Columns: step | t | yaw | pos(x,y,z) | vel(vx,vy) | q_rms | hip[LF RF LB RB] | lift[LF RF LB RB]"
    )
    print(
        f"[params] freq={freq_hz:.2f} A_sh={A_sh:.3f} A_kn={A_kn:.3f} "
        f"lr_trim={lr_trim:+.3f} press={knee_stance_press:.3f} "
        f"A_left={A_left:.3f} A_right={A_right:.3f}\n"
    )

    for k in range(steps):
        t = k * dt
        ramp = min(1.0, t / ramp_time)

        # current yaw for debug print
        root_q = robot.data.root_quat_w[0].cpu().numpy()
        yaw = _quat_to_yaw(root_q[0], root_q[1], root_q[2], root_q[3])

        # analytic phases
        phi = omega * t
        theta = [(phi + b) % (2.0 * math.pi) for b in phase_bias]
        hip_sig = [math.sin(th) for th in theta]

        # smooth swing-lift bump in [0, 1]
        lift = []
        for th in theta:
            s = math.sin(th)
            swing = 1.0 if s > 0.0 else 0.0
            lift.append(swing * 0.5 * (1.0 - math.cos(th)))

        q_tgt = q_stand.clone()

        # shoulders: LEFT / RIGHT trim
        q_tgt[0, idx["LF_sh"]] = q_stand[0, idx["LF_sh"]] + ramp * (A_left * hip_sig[0] * sh_sign["LF"])
        q_tgt[0, idx["RF_sh"]] = q_stand[0, idx["RF_sh"]] + ramp * (A_right * hip_sig[1] * sh_sign["RF"])
        q_tgt[0, idx["LB_sh"]] = q_stand[0, idx["LB_sh"]] + ramp * (A_left * hip_sig[2] * sh_sign["LB"])
        q_tgt[0, idx["RB_sh"]] = q_stand[0, idx["RB_sh"]] + ramp * (A_right * hip_sig[3] * sh_sign["RB"])

        # knees: smooth lift + slight stance press
        def knee_target(leg: str, key: str, lift_val: float) -> float:
            stance = 1.0 if lift_val <= 1e-9 else 0.0
            press = -knee_stance_press * stance * kn_sign[leg]
            return float(
                q_stand[0, idx[key]] + ramp * (A_kn * lift_val * kn_sign[leg] + press)
            )

        q_tgt[0, idx["LF_kn"]] = knee_target("LF", "LF_kn", lift[0])
        q_tgt[0, idx["RF_kn"]] = knee_target("RF", "RF_kn", lift[1])
        q_tgt[0, idx["LB_kn"]] = knee_target("LB", "LB_kn", lift[2])
        q_tgt[0, idx["RB_kn"]] = knee_target("RB", "RB_kn", lift[3])

        # step
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        # debug print
        if (k % print_every) == 0 or k == steps - 1:
            root_p = robot.data.root_pos_w[0].cpu().numpy()
            root_v = robot.data.root_lin_vel_w[0].cpu().numpy()

            q_act = robot.data.joint_pos[0].cpu().numpy()
            q_tar = q_tgt[0].cpu().numpy()
            q_err = q_act - q_tar
            q_rms = float(np.sqrt(np.mean(q_err * q_err)))

            print(
                f"{k:5d} | {t:6.2f} | yaw={yaw:+.3f} | "
                f"p=({root_p[0]:+.3f},{root_p[1]:+.3f},{root_p[2]:+.3f}) | "
                f"v=({root_v[0]:+.3f},{root_v[1]:+.3f}) | "
                f"q_rms={q_rms:.4f} | "
                f"hip=[{hip_sig[0]:+.2f},{hip_sig[1]:+.2f},{hip_sig[2]:+.2f},{hip_sig[3]:+.2f}] | "
                f"lift=[{lift[0]:+.2f},{lift[1]:+.2f},{lift[2]:+.2f},{lift[3]:+.2f}]"
            )

    print("\n=== DONE ===")
    simulation_app.close()


if __name__ == "__main__":
    main()