# spawn_bittle_cpg_trot_v2.py
# Upgraded "true" CPG (phase-oscillator network + duty-shaped waveform)
# Keeps your Route-1 / version-safe patterns and adds structured debug prints.

from __future__ import annotations

import os
import sys
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# MUST create SimulationApp before pxr/omni/isaaclab imports
# ============================================================
from omni.isaac.kit import SimulationApp  # noqa: E402

HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))
simulation_app = SimulationApp({"headless": HEADLESS})

# Isaac Lab source path (Route 1)
ISAACLAB_SOURCE = r"D:\IsaacLab\source"
if ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)

import torch  # noqa: E402
import omni.usd  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


# ============================================================
# Small utilities
# ============================================================
class EveryN:
    def __init__(self, n: int):
        self.n = max(int(n), 1)
        self.i = 0

    def hit(self) -> bool:
        self.i += 1
        return (self.i % self.n) == 0


def clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_0_2pi(x: float) -> float:
    twopi = 2.0 * math.pi
    x = x % twopi
    if x < 0.0:
        x += twopi
    return x


def wrap_mpi_pi(x: float) -> float:
    # wrap to (-pi, pi]
    twopi = 2.0 * math.pi
    x = (x + math.pi) % twopi - math.pi
    return x


def _fmt4(v) -> str:
    return "[" + ", ".join([f"{float(x):+0.3f}" for x in v]) + "]"


def quat_to_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    # q: [4] xyzw
    x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def smooth_relu(x: float, eps: float = 1e-6) -> float:
    # smooth max(0,x); C1-ish
    return 0.5 * (x + math.sqrt(x * x + eps))


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ============================================================
# Version-safe ground + light
# ============================================================
def spawn_ground_and_light():
    import inspect
    import omni.physx.scripts.physicsUtils as physx_utils
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
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

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)


def auto_usd_path() -> str:
    # script is ...\RLCPG\script\xxx.py
    script_dir = Path(__file__).resolve().parent
    usd_dir = (script_dir.parent / "bittle_fixed").resolve()
    if not usd_dir.exists():
        raise FileNotFoundError(f"USD_DIR does not exist: {usd_dir}")
    cands = list(usd_dir.glob("*.usd"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No .usd files in: {usd_dir}")

    def score(p: Path) -> int:
        n = p.name.lower()
        s = 0
        if "bittle" in n:
            s += 20
        if "fixed" in n:
            s += 10
        return s

    cands.sort(key=score, reverse=True)
    print(f"[info] USD_PATH={str(cands[0])}", flush=True)
    return str(cands[0])


# ============================================================
# CPG: Phase Oscillator Network (Kuramoto-like with desired offsets)
# ============================================================
LEG_ORDER = ["LF", "RF", "LB", "RB"]  # fixed order in this script
LEG_IDX = {n: i for i, n in enumerate(LEG_ORDER)}


@dataclass
class CPGParams:
    freq_hz: float = 2.4          # base frequency
    K: float = 6.0                # coupling strength
    duty: float = 0.60            # stance fraction (0.5 = symmetric)
    ramp_time: float = 1.5        # amplitude ramp (s)

    # joint-space mapping gains
    A_sh: float = 0.22            # shoulder amplitude
    A_kn: float = 0.18            # knee lift amplitude
    knee_phase_bias: float = 0.60 * math.pi  # bias between hip swing and knee lift

    # limits (keep conservative)
    SHOULDER_LIM: tuple = (-1.2, 1.2)
    KNEE_LIM: tuple = (-1.6, 1.6)


class PhaseCPG:
    """
    4-oscillator network with fixed desired phase offsets for trot.

    theta_i dot = omega + Σ_j K_ij * sin(theta_j - theta_i - des_ij)
    """

    def __init__(self, params: CPGParams):
        self.p = params
        self.omega = 2.0 * math.pi * self.p.freq_hz

        # desired offsets: des[i,j] = desired (theta_j - theta_i)
        self.des = [[0.0 for _ in range(4)] for _ in range(4)]
        self._build_trot_offsets()

        # coupling weights
        self.K = [[0.0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.K[i][j] = self.p.K

        # state
        self.theta = [0.0, 0.0, 0.0, 0.0]

    def _build_trot_offsets(self):
        # Trot:
        # Group A: LF & RB in-phase
        # Group B: RF & LB in-phase
        # Group B is pi out of phase with Group A
        LF, RF, LB, RB = (LEG_IDX["LF"], LEG_IDX["RF"], LEG_IDX["LB"], LEG_IDX["RB"])

        # set absolute target phase (for init convenience)
        target = [0.0, math.pi, math.pi, 0.0]  # [LF, RF, LB, RB]
        for i in range(4):
            for j in range(4):
                self.des[i][j] = wrap_mpi_pi(target[j] - target[i])

    def reset(self, theta_init=None):
        if theta_init is None:
            # default stable trot init
            self.theta = [0.0, math.pi, math.pi, 0.0]  # [LF, RF, LB, RB]
        else:
            assert len(theta_init) == 4
            self.theta = [wrap_0_2pi(float(x)) for x in theta_init]

    def step(self, dt: float):
        # one Euler step
        th = self.theta[:]  # snapshot
        th_dot = [self.omega, self.omega, self.omega, self.omega]

        for i in range(4):
            csum = 0.0
            for j in range(4):
                if i == j:
                    continue
                # coupling drives (theta_j - theta_i) -> des[i][j]
                e = th[j] - th[i] - self.des[i][j]
                csum += self.K[i][j] * math.sin(e)
            th_dot[i] = self.omega + csum

        for i in range(4):
            self.theta[i] = wrap_0_2pi(self.theta[i] + dt * th_dot[i])

        return th_dot

    def phase_errors(self):
        # return a few key errors in (-pi,pi]
        th = self.theta
        LF, RF, LB, RB = (LEG_IDX["LF"], LEG_IDX["RF"], LEG_IDX["LB"], LEG_IDX["RB"])

        def err(i, j):
            return wrap_mpi_pi((th[j] - th[i]) - self.des[i][j])

        return {
            "LF-RB(0)": err(LF, RB),
            "RF-LB(0)": err(RF, LB),
            "LF-RF(pi)": err(LF, RF),
            "LF-LB(pi)": err(LF, LB),
        }


# ============================================================
# Waveform shaping + joint mapping
# ============================================================
def duty_warp(theta: float, duty: float) -> float:
    """
    Warp oscillator phase theta (uniform in time) into psi so that:
      - swing (sin(psi) > 0, psi in [0, pi)) occupies (1-duty) time
      - stance (sin(psi) < 0, psi in [pi, 2pi)) occupies duty time
    This keeps your "knee lift when sin>0" logic but allows duty != 0.5.
    """
    duty = clampf(duty, 0.05, 0.95)
    twopi = 2.0 * math.pi
    theta = wrap_0_2pi(theta)

    T_swing = twopi * (1.0 - duty)  # time fraction for psi in [0, pi)
    T_stance = twopi * duty         # time fraction for psi in [pi, 2pi)

    if theta < T_swing:
        # map [0, T_swing) -> [0, pi)
        psi = (theta / T_swing) * math.pi
    else:
        # map [T_swing, 2pi) -> [pi, 2pi)
        psi = math.pi + ((theta - T_swing) / T_stance) * math.pi

    return wrap_0_2pi(psi)


@dataclass
class JointMap:
    # indices in robot joint array
    LF_sh: int
    RF_sh: int
    LB_sh: int
    RB_sh: int
    LF_kn: int
    RF_kn: int
    LB_kn: int
    RB_kn: int


def build_joint_map(robot: Articulation) -> JointMap:
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    return JointMap(
        LB_sh=name_to_idx["left_back_shoulder_joint"],
        LF_sh=name_to_idx["left_front_shoulder_joint"],
        RB_sh=name_to_idx["right_back_shoulder_joint"],
        RF_sh=name_to_idx["right_front_shoulder_joint"],
        LB_kn=name_to_idx["left_back_knee_joint"],
        LF_kn=name_to_idx["left_front_knee_joint"],
        RB_kn=name_to_idx["right_back_knee_joint"],
        RF_kn=name_to_idx["right_front_knee_joint"],
    )


# ============================================================
# Main
# ============================================================
def main():
    # ---- IMPORTANT: reuse your working cfg ----
    from bittle_cfg import BITTLE_CFG

    # ----------------------------
    # Params (allow env overrides)
    # ----------------------------
    p = CPGParams(
        freq_hz=env_float("CPG_FREQ_HZ", 2.4),
        K=env_float("CPG_K", 6.0),
        duty=env_float("CPG_DUTY", 0.50),
        ramp_time=env_float("CPG_RAMP", 1.5),
        A_sh=env_float("CPG_A_SH", 0.6),
        A_kn=env_float("CPG_A_KN", 0.9),
        knee_phase_bias=env_float("CPG_KNEE_BIAS", 0.60 * math.pi),
    )

    settle_time = env_float("CPG_SETTLE", 0.6)
    T_END = env_float("CPG_T_END", 20.0)

    # simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    spawn_ground_and_light()

    # spawn robot
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"
    try:
        cfg.spawn.usd_path = auto_usd_path()
    except Exception:
        pass

    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    dt = float(sim.get_physics_dt())
    robot.update(dt)

    # reset to default
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

    # joint indices
    jm = build_joint_map(robot)

    # stand target
    q_stand = default_q.clone()

    # settle
    settle_steps = int(settle_time / dt)
    for _ in range(settle_steps):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ----------------------------
    # CPG init
    # ----------------------------
    cpg = PhaseCPG(p)
    cpg.reset()  # [LF, RF, LB, RB] = [0, pi, pi, 0]

    # side mirror (your proven convention)
    # left legs +1, right legs -1
    side_sign = {
        "LF": +1.0,
        "LB": +1.0,
        "RF": -1.0,
        "RB": -1.0,
    }

    # knee sign (from your stable version)
    knee_sign = {
        "LF": +1.0,
        "LB": +1.0,
        "RF": -1.0,
        "RB": -1.0,
    }

    # debug cadence
    dbg_fast = EveryN(1)     # first 1s: every step
    dbg_slow = EveryN(30)    # afterwards: ~0.25s at 120Hz

    t = 0.0
    step = 0

    # Pre-pack index access for speed/clarity
    sh_idx = {"LF": jm.LF_sh, "RF": jm.RF_sh, "LB": jm.LB_sh, "RB": jm.RB_sh}
    kn_idx = {"LF": jm.LF_kn, "RF": jm.RF_kn, "LB": jm.LB_kn, "RB": jm.RB_kn}

    print(
        f"[CPGv2] dt={dt:.6f}  freq_hz={p.freq_hz:.3f}  K={p.K:.3f}  duty={p.duty:.2f}  "
        f"A_sh={p.A_sh:.3f}  A_kn={p.A_kn:.3f}  knee_bias={p.knee_phase_bias/math.pi:.2f}π",
        flush=True,
    )

    while simulation_app.is_running() and t < T_END:
        # amplitude ramp
        ramp = clampf(t / p.ramp_time, 0.0, 1.0)

        # CPG update
        th_dot = cpg.step(dt)
        th = cpg.theta[:]  # [LF, RF, LB, RB]

        # build targets
        q_tgt = q_stand.clone()

        # per-leg signals
        hip_sig = {}
        lift_sig = {}
        psi_leg = {}

        for leg in LEG_ORDER:
            i = LEG_IDX[leg]
            psi = duty_warp(th[i], p.duty)
            psi_leg[leg] = psi

            # hip swing signal (centered)
            s = math.sin(psi)
            hip_sig[leg] = s

            # knee lift only during swing (sin>0) but smooth
            lift = smooth_relu(hip_sig[leg])
            lift_sig[leg] = lift

        # shoulders
        for leg in LEG_ORDER:
            q_tgt[0, sh_idx[leg]] += ramp * (side_sign[leg] * p.A_sh * hip_sig[leg])

        # knees (lift)
        for leg in LEG_ORDER:
            q_tgt[0, kn_idx[leg]] += ramp * (knee_sign[leg] * p.A_kn * lift_sig[leg])

        # clamp
        for leg in LEG_ORDER:
            j = sh_idx[leg]
            q_tgt[0, j] = clampf(float(q_tgt[0, j]), p.SHOULDER_LIM[0], p.SHOULDER_LIM[1])
        for leg in LEG_ORDER:
            j = kn_idx[leg]
            q_tgt[0, j] = clampf(float(q_tgt[0, j]), p.KNEE_LIM[0], p.KNEE_LIM[1])

        # ----- APPLY (critical order) -----
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        # ----- DEBUG PRINT -----
        do_print = dbg_fast.hit() if t < 1.0 else dbg_slow.hit()
        if do_print:
            rs = robot.data.root_state_w[0]
            yaw = quat_to_yaw_xyzw(rs[3:7])
            pos = robot.data.root_state_w[0, 0:3]

            # phase errors (how well coupling holds the trot)
            pe = cpg.phase_errors()

            # targets vs stand
            q_act = robot.data.joint_pos[0]

            sh_stand = torch.tensor(
                [q_stand[0, sh_idx["LF"]], q_stand[0, sh_idx["RF"]], q_stand[0, sh_idx["LB"]], q_stand[0, sh_idx["RB"]]]
            )
            kn_stand = torch.tensor(
                [q_stand[0, kn_idx["LF"]], q_stand[0, kn_idx["RF"]], q_stand[0, kn_idx["LB"]], q_stand[0, kn_idx["RB"]]]
            )

            sh_tgt = torch.tensor(
                [q_tgt[0, sh_idx["LF"]], q_tgt[0, sh_idx["RF"]], q_tgt[0, sh_idx["LB"]], q_tgt[0, sh_idx["RB"]]]
            )
            kn_tgt = torch.tensor(
                [q_tgt[0, kn_idx["LF"]], q_tgt[0, kn_idx["RF"]], q_tgt[0, kn_idx["LB"]], q_tgt[0, kn_idx["RB"]]]
            )

            sh_act = torch.tensor(
                [q_act[sh_idx["LF"]], q_act[sh_idx["RF"]], q_act[sh_idx["LB"]], q_act[sh_idx["RB"]]]
            )
            kn_act = torch.tensor(
                [q_act[kn_idx["LF"]], q_act[kn_idx["RF"]], q_act[kn_idx["LB"]], q_act[kn_idx["RB"]]]
            )

            sh_dlt = sh_tgt - sh_stand
            kn_dlt = kn_tgt - kn_stand

            theta_vec = torch.tensor(th, dtype=torch.float32)
            thdot_vec = torch.tensor(th_dot, dtype=torch.float32)

            print(f"\n--- step={step} t={t:.3f}s ramp={ramp:.2f} yaw={float(yaw):+0.3f} "
                  f"root_xy=({float(pos[0]):+0.3f},{float(pos[1]):+0.3f}) z={float(pos[2]):+0.3f} ---", flush=True)

            print(f"theta[LF,RF,LB,RB]={_fmt4(theta_vec)}  theta_dot(rad/s)={_fmt4(thdot_vec)}", flush=True)
            print(
                f"phase_err: LF-RB={pe['LF-RB(0)']:+0.3f}  RF-LB={pe['RF-LB(0)']:+0.3f}  "
                f"LF-RF={pe['LF-RF(pi)']:+0.3f}  LF-LB={pe['LF-LB(pi)']:+0.3f}",
                flush=True,
            )

            psi_vec = torch.tensor([psi_leg["LF"], psi_leg["RF"], psi_leg["LB"], psi_leg["RB"]], dtype=torch.float32)
            hip_vec = torch.tensor([hip_sig["LF"], hip_sig["RF"], hip_sig["LB"], hip_sig["RB"]], dtype=torch.float32)
            lift_vec = torch.tensor([lift_sig["LF"], lift_sig["RF"], lift_sig["LB"], lift_sig["RB"]], dtype=torch.float32)
            print(f"psi_warp[LF,RF,LB,RB]={_fmt4(psi_vec)}", flush=True)
            print(f"hip_sig[LF,RF,LB,RB]={_fmt4(hip_vec)}   lift_sig[LF,RF,LB,RB]={_fmt4(lift_vec)}", flush=True)

            print(f"sh_dlt_tgt[LF,RF,LB,RB]={_fmt4(sh_dlt)}  sh_act={_fmt4(sh_act)}", flush=True)
            print(f"kn_dlt_tgt[LF,RF,LB,RB]={_fmt4(kn_dlt)}  kn_act={_fmt4(kn_act)}", flush=True)

        t += dt
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("!!! EXCEPTION !!!", repr(e), flush=True)
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
