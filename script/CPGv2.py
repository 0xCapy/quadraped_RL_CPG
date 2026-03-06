# CPGv2.py
# Full fixed version:
# - DOES NOT override cfg.spawn.usd_path (uses bittle_cfg.py)
# - prim_path points to "/World/Bittle/bittle"
# - handles Isaac Lab joint tensors as (num_envs, num_joints): ALWAYS index [0, idx]
# - safe stand-pose mapping using name_to_idx (no list.index)
# - conservative, stable default CPG amplitudes (A_sh=0.22, A_kn=0.18)
# - keeps your debug prints style (root_xy, z, yaw, phase, tgt/act)

from __future__ import annotations

import os
import math
import traceback
from dataclasses import dataclass

# ============================================================
# MUST create SimulationApp BEFORE any isaaclab/pxr imports.
# ============================================================
from omni.isaac.kit import SimulationApp  # noqa: E402

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class CPGParams:
    freq_hz: float = 2.4          # base frequency
    K: float = 6.0                # coupling strength
    duty: float = 0.50            # stance fraction (0.5 symmetric)
    ramp_time: float = 1.5        # amplitude ramp (s)

    # joint-space mapping gains
    A_sh: float = 0.22            # shoulder amplitude
    A_kn: float = 0.18            # knee lift amplitude

    # conservative joint clamps
    SHOULDER_LIM: tuple[float, float] = (-1.2, 1.2)
    KNEE_LIM: tuple[float, float] = (-1.6, 1.6)


class PhaseCPG:
    """
    4-oscillator network with fixed desired phase offsets for trot.

    Leg order: [LF, RF, LB, RB]
    Desired phases: LF=0, RF=pi, LB=pi, RB=0
    """
    def __init__(self, params: CPGParams):
        self.p = params
        self.omega = 2.0 * math.pi * self.p.freq_hz
        self.theta = [0.0, math.pi, math.pi, 0.0]

        des_phase = [0.0, math.pi, math.pi, 0.0]
        self.des = [[0.0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                self.des[i][j] = des_phase[j] - des_phase[i]

    def step(self, dt: float):
        dtheta = [self.omega for _ in range(4)]
        for i in range(4):
            coup = 0.0
            for j in range(4):
                if i == j:
                    continue
                coup += math.sin(self.theta[j] - self.theta[i] - self.des[i][j])
            dtheta[i] += self.p.K * coup

        for i in range(4):
            self.theta[i] = (self.theta[i] + dtheta[i] * dt) % (2.0 * math.pi)

    def get_theta(self):
        return self.theta[:]


def duty_warp(theta: float, duty: float) -> float:
    """
    Warp phase to get duty-shaped stance vs swing.

    - stance fraction = duty in (0,1)
    - returns warped phase psi in [0, 2pi)
    """
    duty = clamp(duty, 1e-3, 1.0 - 1e-3)
    twopi = 2.0 * math.pi
    u = (theta % twopi) / twopi  # [0,1)

    if u < duty:
        v = (u / duty) * 0.5
    else:
        v = 0.5 + ((u - duty) / (1.0 - duty)) * 0.5
    return v * twopi


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw from quat (w,x,y,z)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def setup_ground_and_lighting():
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdLux, Sdf
    from omni.physx.scripts import physicsUtils

    stage = sim_utils.get_current_stage()

    # -----------------------
    # 1) Visible ground mesh (for rendering)
    # -----------------------
    # Create a large plane mesh as a thin cube (always visible, receives shadows)
    ground_path = Sdf.Path("/World/Ground")
    cube = UsdGeom.Cube.Define(stage, ground_path)
    cube.GetSizeAttr().Set(1.0)

    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set((0.0, 0.0, -0.001))   # slightly below z=0
    xf.AddScaleOp().Set((50.0, 50.0, 0.001))      # 100m x 100m, very thin

    # -----------------------
    # 2) Physics ground plane (for collision)
    # -----------------------
    # This ensures physics contact even if render mesh is changed later.
    try:
        physicsUtils.add_ground_plane(
            stage=stage,
            planePath="/World/PhysicsGroundPlane",
            axis="Z",
            size=2000.0,
            position=(0.0, 0.0, 0.0),
        )
    except Exception:
        # some versions have different signature; ignore if already exists
        pass

    # -----------------------
    # 3) Lighting + shadows
    # -----------------------
    dome_prim = stage.DefinePrim(Sdf.Path("/World/DomeLight"), "DomeLight")
    dome = UsdLux.DomeLight(dome_prim)
    dome.CreateIntensityAttr(1500.0)

    sun_prim = stage.DefinePrim(Sdf.Path("/World/SunLight"), "DistantLight")
    sun = UsdLux.DistantLight(sun_prim)
    sun.CreateIntensityAttr(3000.0)
    # rotate sun a bit so shadows appear clearly
    sun_xf = UsdGeom.Xformable(sun_prim)
    sun_xf.ClearXformOpOrder()
    sun_xf.AddRotateXYZOp().Set((-45.0, 0.0, 45.0))

def main():
    # IsaacLab imports AFTER SimulationApp
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import Articulation

    from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS

    # ------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------
    sim_cfg = SimulationCfg(dt=1 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.reset()
    setup_ground_and_lighting()
    from pxr import UsdGeom
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()

    # Enable PhysX debug visualization (collision shapes/contact)
    dbg = stage.DefinePrim("/World/PhysxDebug", "Scope")

    try:
        import omni.physx as physx
        iface = physx.get_physx_interface()
        # Turn on debug visualization flags (works across many versions)
        iface.set_visualization_parameter("COLLISION_SHAPES", 1)
        iface.set_visualization_parameter("CONTACT_POINTS", 1)
        iface.set_visualization_parameter("CONTACT_NORMALS", 1)
    except Exception as e:
        print("[warn] physx debug vis not available:", e, flush=True)
    # ------------------------------------------------------------
    # Spawn robot (IMPORTANT: do NOT override usd_path here)
    # ------------------------------------------------------------
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle/bittle"  # <<<<<< FIX: articulation root at /bittle

    robot = Articulation(cfg=cfg)
    sim.reset()

    # reset to defaults (written to sim)
    robot.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)

    # warm-up
    for _ in range(2):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

    # ------------------------------------------------------------
    # Build name->idx mapping
    # ------------------------------------------------------------
    joint_names = list(robot.joint_names)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    # ------------------------------------------------------------
    # Settle phase: hold stand pose
    # ------------------------------------------------------------
    settle_time = env_float("SETTLE_TIME", 0.6)
    n_settle = int(settle_time / sim_cfg.dt)

    # IMPORTANT: joint tensors are (num_envs, num_joints)
    q_stand = robot.data.default_joint_pos.clone()
    for name, val in STAND_JOINT_POS.items():
        if name in name_to_idx:
            q_stand[0, name_to_idx[name]] = float(val)

    for _ in range(n_settle):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

    # ------------------------------------------------------------
    # CPG params (env overrides)
    # ------------------------------------------------------------
    p = CPGParams(
        freq_hz=env_float("CPG_FREQ", 2.4),
        K=env_float("CPG_K", 6.0),
        duty=env_float("CPG_DUTY", 0.50),
        ramp_time=env_float("CPG_RAMP", 1.5),
        A_sh=env_float("CPG_A_SH", 0.22),
        A_kn=env_float("CPG_A_KN", 0.18),
    )
    cpg = PhaseCPG(p)

    # joint index map
    jmap = {
        "LF_sh": name_to_idx["left_front_shoulder_joint"],
        "RF_sh": name_to_idx["right_front_shoulder_joint"],
        "LB_sh": name_to_idx["left_back_shoulder_joint"],
        "RB_sh": name_to_idx["right_back_shoulder_joint"],
        "LF_kn": name_to_idx["left_front_knee_joint"],
        "RF_kn": name_to_idx["right_front_knee_joint"],
        "LB_kn": name_to_idx["left_back_knee_joint"],
        "RB_kn": name_to_idx["right_back_knee_joint"],
    }

    # left/right mirror sign handling (keep your previous convention)
    side = {"LF": +1.0, "LB": +1.0, "RF": -1.0, "RB": -1.0}

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------
    T = env_float("RUN_TIME", 20.0)
    steps = int(T / sim_cfg.dt)
    dbg_stride = max(1, int(0.25 / sim_cfg.dt))

    for step in range(steps):
        t = step * sim_cfg.dt
        ramp = clamp(t / p.ramp_time, 0.0, 1.0)

        # update CPG
        cpg.step(sim_cfg.dt)
        theta = cpg.get_theta()  # [LF, RF, LB, RB]

        # duty warp + signals
        psi = [duty_warp(th, p.duty) for th in theta]
        hip_sig = [math.sin(x) for x in psi]
        lift_sig = [max(0.0, s) for s in hip_sig]  # lift only on positive half

        # targets (clone 2D tensor)
        q_tgt = q_stand.clone()

        # shoulders (2D indexing!)
        q_tgt[0, jmap["LF_sh"]] = q_stand[0, jmap["LF_sh"]] + ramp * (p.A_sh * hip_sig[0] * side["LF"])
        q_tgt[0, jmap["RF_sh"]] = q_stand[0, jmap["RF_sh"]] + ramp * (p.A_sh * hip_sig[1] * side["RF"])
        q_tgt[0, jmap["LB_sh"]] = q_stand[0, jmap["LB_sh"]] + ramp * (p.A_sh * hip_sig[2] * side["LB"])
        q_tgt[0, jmap["RB_sh"]] = q_stand[0, jmap["RB_sh"]] + ramp * (p.A_sh * hip_sig[3] * side["RB"])

        # knees (lift)
        q_tgt[0, jmap["LF_kn"]] = q_stand[0, jmap["LF_kn"]] + ramp * (p.A_kn * lift_sig[0] * side["LF"])
        q_tgt[0, jmap["RF_kn"]] = q_stand[0, jmap["RF_kn"]] + ramp * (p.A_kn * lift_sig[1] * side["RF"])
        q_tgt[0, jmap["LB_kn"]] = q_stand[0, jmap["LB_kn"]] + ramp * (p.A_kn * lift_sig[2] * side["LB"])
        q_tgt[0, jmap["RB_kn"]] = q_stand[0, jmap["RB_kn"]] + ramp * (p.A_kn * lift_sig[3] * side["RB"])

        # clamp (2D indexing!)
        for k in ("LF_sh", "RF_sh", "LB_sh", "RB_sh"):
            i = jmap[k]
            q_tgt[0, i] = clamp(float(q_tgt[0, i]), p.SHOULDER_LIM[0], p.SHOULDER_LIM[1])
        for k in ("LF_kn", "RF_kn", "LB_kn", "RB_kn"):
            i = jmap[k]
            q_tgt[0, i] = clamp(float(q_tgt[0, i]), p.KNEE_LIM[0], p.KNEE_LIM[1])

        # send targets
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

        # debug print (matches your style)
        if step > 0 and step % dbg_stride == 0:
            rs = robot.data.root_state_w[0]
            x, y, z = rs[0].item(), rs[1].item(), rs[2].item()
            qw, qx, qy, qz = rs[3].item(), rs[4].item(), rs[5].item(), rs[6].item()
            yaw = quat_to_yaw(qw, qx, qy, qz)

            # phase check
            phase_err_lf_rb = ((theta[0] - theta[3] + math.pi) % (2 * math.pi)) - math.pi
            phase_err_rf_lb = ((theta[1] - theta[2] + math.pi) % (2 * math.pi)) - math.pi
            phase_err_lf_rf = ((theta[0] - theta[1] + math.pi) % (2 * math.pi)) - math.pi
            phase_err_lf_lb = ((theta[0] - theta[2] + math.pi) % (2 * math.pi)) - math.pi

            # deltas for debug (target - stand), using 2D indexing
            sh_dlt = [
                float(q_tgt[0, jmap["LF_sh"]] - q_stand[0, jmap["LF_sh"]]),
                float(q_tgt[0, jmap["RF_sh"]] - q_stand[0, jmap["RF_sh"]]),
                float(q_tgt[0, jmap["LB_sh"]] - q_stand[0, jmap["LB_sh"]]),
                float(q_tgt[0, jmap["RB_sh"]] - q_stand[0, jmap["RB_sh"]]),
            ]
            kn_dlt = [
                float(q_tgt[0, jmap["LF_kn"]] - q_stand[0, jmap["LF_kn"]]),
                float(q_tgt[0, jmap["RF_kn"]] - q_stand[0, jmap["RF_kn"]]),
                float(q_tgt[0, jmap["LB_kn"]] - q_stand[0, jmap["LB_kn"]]),
                float(q_tgt[0, jmap["RB_kn"]] - q_stand[0, jmap["RB_kn"]]),
            ]

            # actual joint pos (2D)
            q_act = robot.data.joint_pos[0]
            sh_act = [
                float(q_act[jmap["LF_sh"]]),
                float(q_act[jmap["RF_sh"]]),
                float(q_act[jmap["LB_sh"]]),
                float(q_act[jmap["RB_sh"]]),
            ]
            kn_act = [
                float(q_act[jmap["LF_kn"]]),
                float(q_act[jmap["RF_kn"]]),
                float(q_act[jmap["LB_kn"]]),
                float(q_act[jmap["RB_kn"]]),
            ]

            print(
                f"--- step={step} t={t:.3f}s ramp={ramp:.2f} yaw={yaw:+.3f} root_xy=({x:+.3f},{y:+.3f}) z={z:+.3f} ---",
                flush=True,
            )
            print(
                f"theta[LF,RF,LB,RB]=[{theta[0]:+.3f}, {theta[1]:+.3f}, {theta[2]:+.3f}, {theta[3]:+.3f}]  "
                f"theta_dot(rad/s)=[{2*math.pi*p.freq_hz:+.3f}, {2*math.pi*p.freq_hz:+.3f}, {2*math.pi*p.freq_hz:+.3f}, {2*math.pi*p.freq_hz:+.3f}]",
                flush=True,
            )
            print(
                f"phase_err: LF-RB={phase_err_lf_rb:+.3f}  RF-LB={phase_err_rf_lb:+.3f}  LF-RF={phase_err_lf_rf:+.3f}  LF-LB={phase_err_lf_lb:+.3f}",
                flush=True,
            )
            print(
                f"psi_warp[LF,RF,LB,RB]=[{psi[0]:+.3f}, {psi[1]:+.3f}, {psi[2]:+.3f}, {psi[3]:+.3f}]",
                flush=True,
            )
            print(
                f"hip_sig[LF,RF,LB,RB]=[{hip_sig[0]:+.3f}, {hip_sig[1]:+.3f}, {hip_sig[2]:+.3f}, {hip_sig[3]:+.3f}]   "
                f"lift_sig[LF,RF,LB,RB]=[{lift_sig[0]:+.3f}, {lift_sig[1]:+.3f}, {lift_sig[2]:+.3f}, {lift_sig[3]:+.3f}]",
                flush=True,
            )
            print(
                f"sh_dlt_tgt[LF,RF,LB,RB]=[{sh_dlt[0]:+.3f}, {sh_dlt[1]:+.3f}, {sh_dlt[2]:+.3f}, {sh_dlt[3]:+.3f}]  "
                f"sh_act=[{sh_act[0]:+.3f}, {sh_act[1]:+.3f}, {sh_act[2]:+.3f}, {sh_act[3]:+.3f}]",
                flush=True,
            )
            print(
                f"kn_dlt_tgt[LF,RF,LB,RB]=[{kn_dlt[0]:+.3f}, {kn_dlt[1]:+.3f}, {kn_dlt[2]:+.3f}, {kn_dlt[3]:+.3f}]  "
                f"kn_act=[{kn_act[0]:+.3f}, {kn_act[1]:+.3f}, {kn_act[2]:+.3f}, {kn_act[3]:+.3f}]",
                flush=True,
            )

    print("[done] finished.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()