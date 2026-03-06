from omni.isaac.kit import SimulationApp  # MUST be first import

import os
HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))
simulation_app = SimulationApp({"headless": HEADLESS})

import math
import inspect
from dataclasses import dataclass

import numpy as np

import omni.usd
from pxr import UsdLux

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS


# ----------------------------
# Helpers
# ----------------------------
def clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def smoothstep(t: float) -> float:
    t = clampf(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bezier_cubic(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t = clampf(t, 0.0, 1.0)
    u = 1.0 - t
    return (u**3) * p0 + 3.0 * (u**2) * t * p1 + 3.0 * u * (t**2) * p2 + (t**3) * p3


def quat_to_yaw_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


def quat_rotate_wxyz(qw: float, qx: float, qy: float, qz: float, vx: float, vy: float, vz: float):
    # Rotate vector v by quaternion q (wxyz). Returns tuple (x,y,z) in world.
    # q * v * q_conj
    # v as pure quaternion (0,v)
    # Using optimized formula.
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + w*t + cross(q_vec, t)
    vpx = vx + qw * tx + (qy * tz - qz * ty)
    vpy = vy + qw * ty + (qz * tx - qx * tz)
    vpz = vz + qw * tz + (qx * ty - qy * tx)
    return vpx, vpy, vpz


# ----------------------------
# Ground + Light (version-safe)
# ----------------------------
def spawn_ground_plane(stage):
    import omni.physx.scripts.physicsUtils as physicsUtils

    fn = physicsUtils.add_ground_plane
    sig = inspect.signature(fn)

    plane_path = "/World/GroundPlane"
    color = (0.2, 0.2, 0.2)

    kwargs = {}
    for name in sig.parameters.keys():
        n = name.lower()
        if "stage" in n:
            kwargs[name] = stage
        elif "planepath" in n or (("path" in n or "prim" in n) and "plane" in n):
            kwargs[name] = plane_path
        elif "color" in n or "colour" in n:
            kwargs[name] = color
        elif "axis" in n:
            kwargs[name] = "Z"
        elif "size" in n or "halfsize" in n:
            kwargs[name] = 50.0
        elif "position" in n:
            kwargs[name] = (0.0, 0.0, 0.0)
        elif "normal" in n:
            kwargs[name] = (0.0, 0.0, 1.0)
        elif ("static" in n and "friction" in n) or n == "staticfriction":
            kwargs[name] = 1.0
        elif ("dynamic" in n and "friction" in n) or n == "dynamicfriction":
            kwargs[name] = 1.0
        elif "restitution" in n:
            kwargs[name] = 0.0

    # Some versions need exact kw names.
    if "planePath" in sig.parameters and "planePath" not in kwargs:
        kwargs["planePath"] = plane_path
    if "color" in sig.parameters and "color" not in kwargs:
        kwargs["color"] = color

    fn(**kwargs)



def spawn_debug_grid(stage, extent: float = 2.0, spacing: float = 0.1, z: float = 0.002):
    """
    Visual-only ground grid to make world axes obvious.
    - extent: half-size in meters (grid covers [-extent, +extent])
    - spacing: grid line spacing in meters
    - z: small lift to avoid z-fighting with the PhysX ground plane
    """
    from pxr import UsdGeom, Gf, Vt

    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, "/World/Debug")

    # --- grid lines (single prim with multiple linear curves) ---
    n = int(round((2.0 * extent) / spacing))
    n = max(n, 2)

    points = []
    counts = []

    # lines parallel to X (varying Y)
    y = -extent
    while y <= extent + 1e-6:
        points.append(Gf.Vec3f(-extent, y, z))
        points.append(Gf.Vec3f(+extent, y, z))
        counts.append(2)
        y += spacing

    # lines parallel to Y (varying X)
    x = -extent
    while x <= extent + 1e-6:
        points.append(Gf.Vec3f(x, -extent, z))
        points.append(Gf.Vec3f(x, +extent, z))
        counts.append(2)
        x += spacing

    grid = UsdGeom.BasisCurves.Define(stage, "/World/Debug/Grid")
    grid.CreateTypeAttr("linear")
    grid.CreateWrapAttr("nonperiodic")
    grid.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    grid.CreatePointsAttr(Vt.Vec3fArray(points))
    grid.CreateWidthsAttr(Vt.FloatArray([0.002] * len(points)))
    # subtle grey
    grid.GetDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.35, 0.35)])

    # --- axes (thicker, colored) ---
    ax = UsdGeom.BasisCurves.Define(stage, "/World/Debug/AxisX")
    ax.CreateTypeAttr("linear")
    ax.CreateWrapAttr("nonperiodic")
    ax.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
    ax.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-extent, 0.0, z * 1.5), Gf.Vec3f(+extent, 0.0, z * 1.5)]))
    ax.CreateWidthsAttr(Vt.FloatArray([0.01, 0.01]))
    ax.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.20, 0.20)])  # X axis

    ay = UsdGeom.BasisCurves.Define(stage, "/World/Debug/AxisY")
    ay.CreateTypeAttr("linear")
    ay.CreateWrapAttr("nonperiodic")
    ay.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
    ay.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, -extent, z * 1.5), Gf.Vec3f(0.0, +extent, z * 1.5)]))
    ay.CreateWidthsAttr(Vt.FloatArray([0.01, 0.01]))
    ay.GetDisplayColorAttr().Set([Gf.Vec3f(0.20, 0.85, 0.20)])  # Y axis

    # origin marker (small cross)
    o = UsdGeom.BasisCurves.Define(stage, "/World/Debug/Origin")
    o.CreateTypeAttr("linear")
    o.CreateWrapAttr("nonperiodic")
    o.CreateCurveVertexCountsAttr(Vt.IntArray([2, 2]))
    o.CreatePointsAttr(
        Vt.Vec3fArray([
            Gf.Vec3f(-0.05, 0.0, z * 2.0), Gf.Vec3f(+0.05, 0.0, z * 2.0),
            Gf.Vec3f(0.0, -0.05, z * 2.0), Gf.Vec3f(0.0, +0.05, z * 2.0),
        ])
    )
    o.CreateWidthsAttr(Vt.FloatArray([0.01, 0.01, 0.01, 0.01]))
    o.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.95, 0.95)])


def spawn_light(stage):
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(0.5)


# ----------------------------
# Petoi-feasible joint-space gait generator
# ----------------------------
@dataclass
class GaitParams:
    # timing
    freq_hz: float = 2.4
    beta: float = 0.62  # stance ratio (duty factor)
    back_phase_lead: float = 0.03  # fraction of cycle (0..1), back legs lead slightly

    # amplitudes (radians)
    hip_amp_front: float = 0.20
    hip_amp_back: float = 0.22
    knee_lift_front: float = 0.18
    knee_lift_back: float = 0.20

    # stance shaping
    knee_stance_comp: float = 0.04   # small extra knee bend in stance for traction
    toe_off: float = 0.05            # small knee extension near end-stance for push-off

    # waveform shaping (dimensionless)
    stance_shape: float = 0.60       # p1/p2 scaling for stance hip Bezier
    swing_sharp: float = 0.25        # p2 scaling for swing knee Bezier (smaller -> sharper)


def petoi_leg_wave(u: float, p: GaitParams, hip_amp: float, knee_lift: float):
    """Return (hip_delta, knee_delta) around stand for one leg.

    - Uses adjustable stance/swing durations via beta (duty factor).
    - Hip: slow stance sweep + fast swing return.
    - Knee: stance compression (traction) + toe-off extension + swing clearance bump.

    All deltas are in the *leg's local sagittal DOF*; final sign mapping happens outside.
    """
    u = u % 1.0

    if u < p.beta:
        s = smoothstep(u / p.beta)

        # stance hip sweep: +A -> -A (with controllable mid-stance velocity)
        a = float(p.stance_shape)
        hip = bezier_cubic(+hip_amp, +a * hip_amp, -a * hip_amp, -hip_amp, s)

        # stance knee: small compression hump + toe-off extension near end stance
        comp = p.knee_stance_comp * bezier_cubic(0.0, 1.0, 1.0, 0.0, s)
        toe = p.toe_off * smoothstep((s - 0.75) / 0.25)  # ramps in late stance
        knee = (+comp) - toe

    else:
        s = smoothstep((u - p.beta) / (1.0 - p.beta))

        # swing hip return: -A -> +A
        hip = bezier_cubic(-hip_amp, -hip_amp, +hip_amp, +hip_amp, s)

        # swing knee clearance: sharp bump (0 -> K -> 0)
        k2 = float(p.swing_sharp)
        knee = bezier_cubic(0.0, +knee_lift, +k2 * knee_lift, 0.0, s)

    return hip, knee


# ----------------------------
# Main
# ----------------------------
def main():
    # ---- simulation ----
    dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0")  # set to "cpu" if needed
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    sim.set_camera_view([1.8, 1.8, 1.2], [0.0, 0.0, 0.15])

    spawn_ground_plane(stage)
    spawn_debug_grid(stage, extent=2.0, spacing=0.1)
    spawn_light(stage)

    # ---- robot ----
    cfg = BITTLE_CFG.copy()
    # Petoi USDs commonly use /World/bittle; keep your cfg if already correct.
    # cfg.prim_path = "/World/bittle"

    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    robot.update(dt)

    # robust reset
    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.write_data_to_sim()
    for _ in range(2):
        sim.step()
        robot.update(dt)
    robot.reset()

    # ---- joint mapping ----
    joint_names = list(robot.data.joint_names)
    jmap = {n: i for i, n in enumerate(joint_names)}

    needed = [
        "left_front_shoulder_joint", "right_front_shoulder_joint",
        "left_back_shoulder_joint", "right_back_shoulder_joint",
        "left_front_knee_joint", "right_front_knee_joint",
        "left_back_knee_joint", "right_back_knee_joint",
    ]
    missing = [n for n in needed if n not in jmap]

    print("\n=== SANITY ===")
    print(f"[prim_path] {cfg.prim_path}")
    print(f"[joint_count] {len(joint_names)}")
    if missing:
        print("[ERROR] Missing joints:", missing)
        simulation_app.close()
        return

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

    # ---- stand pose ----
    q_stand = robot.data.default_joint_pos.clone()
    for jn, val in STAND_JOINT_POS.items():
        if jn in jmap:
            q_stand[0, jmap[jn]] = float(val)

    # settle
    for _ in range(int(0.6 / dt)):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ---- gait params ----
    p = GaitParams()
    omega = 2.0 * math.pi * p.freq_hz
    phi = 0.0

    # trot phases [LF, RF, LB, RB]
    phase_off = [0.0, math.pi, math.pi, 0.0]
    back_lead_rad = 2.0 * math.pi * p.back_phase_lead

    # mirror sign convention (validated): left +, right -
    sh_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}
    kn_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}
    side_sign = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}  # for trim in *joint space*

    legs = ["LF", "RF", "LB", "RB"]

    # safety clamps (conservative)
    SHOULDER_LIM = (-1.2, 1.2)
    KNEE_LIM = (-1.6, 1.6)

    # ---- optional tiny IMU-style trim (helps drift; keep small for RL residual)
    ENABLE_TRIM = True
    k_yaw = 0.25
    k_vy = 0.30
    trim_max = 0.08

    # run
    ramp_time = 1.2
    T_END = 20.0
    steps = int(T_END / dt)
    print_every = int(0.5 / dt)

    print("\n=== RUN: Petoi-feasible Bezier CPG (TROT baseline) ===")
    print(f"[params] freq={p.freq_hz:.2f}Hz beta={p.beta:.2f} back_lead={p.back_phase_lead:.3f}cyc")
    print(f"         hip(front/back)=({p.hip_amp_front:.3f},{p.hip_amp_back:.3f})  kneeLift(front/back)=({p.knee_lift_front:.3f},{p.knee_lift_back:.3f})")
    print(f"         stance_comp={p.knee_stance_comp:.3f} toe_off={p.toe_off:.3f}  trim={'ON' if ENABLE_TRIM else 'OFF'}")
    print("Columns: step | t | yaw | pos(x,y,z) | vel(vx,vy) | trim | q_rms")

    for k in range(steps):
        t = k * dt
        ramp = min(1.0, t / ramp_time)

        phi = (phi + omega * dt) % (2.0 * math.pi)

        # small trim in *joint space* (mimics Petoi gyro feedback conceptually)
        trim = 0.0
        if ENABLE_TRIM:
            root_q = robot.data.root_quat_w[0].cpu().numpy()  # (w,x,y,z)
            root_v = robot.data.root_lin_vel_w[0].cpu().numpy()
            yaw = quat_to_yaw_wxyz(float(root_q[0]), float(root_q[1]), float(root_q[2]), float(root_q[3]))
            fwdx, fwdy, fwdz = quat_rotate_wxyz(float(root_q[0]), float(root_q[1]), float(root_q[2]), float(root_q[3]), 1.0, 0.0, 0.0)
            vy = float(root_v[1])
            trim = clampf((-k_yaw * yaw) + (-k_vy * vy), -trim_max, +trim_max)

        q_tgt = q_stand.clone()

        for i, leg in enumerate(legs):
            th = (phi + phase_off[i]) % (2.0 * math.pi)
            if leg in ("LB", "RB"):
                th = (th + back_lead_rad) % (2.0 * math.pi)

            u = th / (2.0 * math.pi)

            if leg in ("LF", "RF"):
                hip_amp = p.hip_amp_front
                knee_lift = p.knee_lift_front
            else:
                hip_amp = p.hip_amp_back
                knee_lift = p.knee_lift_back

            hip_d, knee_d = petoi_leg_wave(u, p, hip_amp, knee_lift)

            # apply mirror mapping + small trim in joint space
            sh = float(q_stand[0, idx[f"{leg}_sh"]]) + ramp * (hip_d * sh_sign[leg] + trim * side_sign[leg])
            kn = float(q_stand[0, idx[f"{leg}_kn"]]) + ramp * (knee_d * kn_sign[leg])

            q_tgt[0, idx[f"{leg}_sh"]] = clampf(sh, SHOULDER_LIM[0], SHOULDER_LIM[1])
            q_tgt[0, idx[f"{leg}_kn"]] = clampf(kn, KNEE_LIM[0], KNEE_LIM[1])

        # apply (critical order)
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if (k % print_every) == 0 or k == steps - 1:
            root_p = robot.data.root_pos_w[0].cpu().numpy()
            root_q = robot.data.root_quat_w[0].cpu().numpy()
            root_v = robot.data.root_lin_vel_w[0].cpu().numpy()
            yaw = quat_to_yaw_wxyz(float(root_q[0]), float(root_q[1]), float(root_q[2]), float(root_q[3]))

            q_act = robot.data.joint_pos[0].cpu().numpy()
            q_tar = q_tgt[0].cpu().numpy()
            q_rms = float(np.sqrt(np.mean((q_act - q_tar) ** 2)))

            print(
                f"{k:5d} | {t:6.2f} | yaw={yaw:+.3f} | fwd=({fwdx:+.2f},{fwdy:+.2f}) | "
                f"p=({root_p[0]:+.3f},{root_p[1]:+.3f},{root_p[2]:+.3f}) | "
                f"v=({root_v[0]:+.3f},{root_v[1]:+.3f}) | trim={trim:+.3f} | q_rms={q_rms:.4f}"
            )

    print("\n=== DONE ===")
    simulation_app.close()


if __name__ == "__main__":
    main()
