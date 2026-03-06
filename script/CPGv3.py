# CPGv4_yawtrim.py
# Baseline trot CPG + automatic yaw-trim controller (very light PI)
#
# Goal: reduce yaw drift & lateral drift without changing USD/contact.
# Still a deterministic baseline (no learning), suitable for residual RL later.

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


def _wrap_pi(a: float) -> float:
    # wrap to (-pi, pi]
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def _spawn_ground_plane(stage):
    import inspect
    import omni.physx.scripts.physicsUtils as physicsUtils

    fn = physicsUtils.add_ground_plane
    sig = inspect.signature(fn)

    plane_path = "/World/GroundPlane"
    color = (0.5, 0.5, 0.5)
    axis = "Z"
    size = 50.0
    position = (0.0, 0.0, 0.0)
    normal = (0.0, 0.0, 1.0)

    kwargs = {}
    for name in sig.parameters.keys():
        if name in ("stage", "usdStage"):
            kwargs[name] = stage
        elif name in ("planePath", "plane_path", "path"):
            kwargs[name] = plane_path
        elif name in ("color", "colour"):
            kwargs[name] = color
        elif name == "axis":
            kwargs[name] = axis
        elif name in ("size", "halfSize"):
            kwargs[name] = size
        elif name == "position":
            kwargs[name] = position
        elif name == "normal":
            kwargs[name] = normal
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
    # Simulation / Robot spawn
    # ---------------------------
    dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0")  # if needed: device="cpu"
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

    _spawn_ground_plane(stage)
    _spawn_light(stage)

    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/bittle"  # match your USD

    robot = Articulation(cfg)
    sim.reset()

    # Warmup
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

    # Stand pose
    q_stand = robot.data.default_joint_pos.clone()
    for name, val in STAND_JOINT_POS.items():
        if name not in jmap:
            print(f"[ERROR] STAND_JOINT_POS key not found: {name}")
            return
        q_stand[0, jmap[name]] = float(val)

    # ---------------------------
    # Baseline params
    # ---------------------------
    freq_hz = 2.2        # IMPORTANT: lowered slightly to reduce slip; try 2.2 first
    A_sh = 0.22
    A_kn = 0.18
    ramp_time = 1.5

    knee_stance_press = 0.02

    phase_bias = [0.0, math.pi, math.pi, 0.0]  # [LF, RF, LB, RB]
    sh_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}
    kn_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}

    omega = 2.0 * math.pi * freq_hz

    # ---------------------------
    # Yaw-trim controller (very light PI)
    # trim modifies front/back shoulder amplitude:
    # A_front = A*(1 - trim), A_back = A*(1 + trim)
    # ---------------------------
    yaw_ref = 0.0
    trim = 0.08          # initial guess (from v3)
    trim_limit = 0.20    # safety bound
    Kp = 0.35            # proportional on yaw error
    Ki = 0.08            # integral on yaw error
    integ = 0.0

    # Use yaw at t=0 as reference offset to avoid initial heading bias
    root_q0 = robot.data.root_quat_w[0].cpu().numpy()
    yaw0 = _quat_to_yaw(root_q0[0], root_q0[1], root_q0[2], root_q0[3])

    # ---------------------------
    # Settle
    # ---------------------------
    settle_s = 0.6
    for _ in range(int(settle_s / dt)):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ---------------------------
    # Run
    # ---------------------------
    run_s = 20.0
    steps = int(run_s / dt)
    print_every = int(0.5 / dt)

    print("=== RUN BASELINE CPG v4 (auto yaw-trim) ===")
    print(
        "Columns: step | t | yaw | trim | pos(x,y,z) | vel(vx,vy) | q_rms | hip[LF RF LB RB] | lift[LF RF LB RB]"
    )
    print(f"[params] freq={freq_hz:.2f} A_sh={A_sh:.3f} A_kn={A_kn:.3f} press={knee_stance_press:.3f} Kp={Kp:.2f} Ki={Ki:.2f}\n")

    for k in range(steps):
        t = k * dt
        ramp = min(1.0, t / ramp_time)

        # Current yaw
        root_q = robot.data.root_quat_w[0].cpu().numpy()
        yaw = _quat_to_yaw(root_q[0], root_q[1], root_q[2], root_q[3])
        yaw_rel = _wrap_pi(yaw - yaw0)  # relative to initial

        # PI trim update (slow, stable)
        err = _wrap_pi(yaw_ref - yaw_rel)
        integ += err * dt
        trim_cmd = trim + (Kp * err + Ki * integ) * dt  # dt makes it very gentle
        trim = max(-trim_limit, min(trim_limit, trim_cmd))

        A_sh_front = A_sh * (1.0 - trim)
        A_sh_back = A_sh * (1.0 + trim)

        # Analytic phases
        phi = omega * t
        theta = [(phi + b) % (2.0 * math.pi) for b in phase_bias]
        hip_sig = [math.sin(th) for th in theta]

        # Smooth lift
        lift = []
        for th in theta:
            s = math.sin(th)
            swing = 1.0 if s > 0.0 else 0.0
            lift.append(swing * 0.5 * (1.0 - math.cos(th)))

        # Build target
        q_tgt = q_stand.clone()

        # shoulders
        q_tgt[0, idx["LF_sh"]] = q_stand[0, idx["LF_sh"]] + ramp * (A_sh_front * hip_sig[0] * sh_sign["LF"])
        q_tgt[0, idx["RF_sh"]] = q_stand[0, idx["RF_sh"]] + ramp * (A_sh_front * hip_sig[1] * sh_sign["RF"])
        q_tgt[0, idx["LB_sh"]] = q_stand[0, idx["LB_sh"]] + ramp * (A_sh_back * hip_sig[2] * sh_sign["LB"])
        q_tgt[0, idx["RB_sh"]] = q_stand[0, idx["RB_sh"]] + ramp * (A_sh_back * hip_sig[3] * sh_sign["RB"])

        # knees
        def knee_target(leg: str, key: str, lift_val: float) -> float:
            stance = 1.0 if lift_val <= 1e-9 else 0.0
            press = -knee_stance_press * stance * kn_sign[leg]
            return float(q_stand[0, idx[key]] + ramp * (A_kn * lift_val * kn_sign[leg] + press))

        q_tgt[0, idx["LF_kn"]] = knee_target("LF", "LF_kn", lift[0])
        q_tgt[0, idx["RF_kn"]] = knee_target("RF", "RF_kn", lift[1])
        q_tgt[0, idx["LB_kn"]] = knee_target("LB", "LB_kn", lift[2])
        q_tgt[0, idx["RB_kn"]] = knee_target("RB", "RB_kn", lift[3])

        # step
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        # diagnostics print
        if (k % print_every) == 0 or k == steps - 1:
            root_p = robot.data.root_pos_w[0].cpu().numpy()
            root_v = robot.data.root_lin_vel_w[0].cpu().numpy()

            q_act = robot.data.joint_pos[0].cpu().numpy()
            q_t = q_tgt[0].cpu().numpy()
            q_err = q_act - q_t
            q_rms = float(np.sqrt(np.mean(q_err * q_err)))

            print(
                f"{k:5d} | {t:6.2f} | yaw={yaw_rel:+.3f} | trim={trim:+.3f} | "
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