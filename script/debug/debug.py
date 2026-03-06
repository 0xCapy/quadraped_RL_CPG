from __future__ import annotations

import os
import sys
import math
import traceback
from pathlib import Path

# MUST create SimulationApp before pxr/omni imports
from omni.isaac.kit import SimulationApp
HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))
simulation_app = SimulationApp({"headless": HEADLESS})

# Isaac Lab source path (Route 1)
ISAACLAB_SOURCE = r"D:\IsaacLab\source"
if ISAACLAB_SOURCE not in sys.path:
    sys.path.append(ISAACLAB_SOURCE)

import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

# -----------------------------
# Debug helpers
# -----------------------------
class EveryN:
    def __init__(self, n: int):
        self.n = max(int(n), 1)
        self.i = 0
    def hit(self) -> bool:
        self.i += 1
        return (self.i % self.n) == 0

def _fmt4(v):
    return "[" + ", ".join([f"{float(x):+0.3f}" for x in v]) + "]"

def quat_to_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    # q: [4] xyzw
    x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

def clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def spawn_ground_and_light():
    # reuse your robust ground/light (from original)
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
    # script is ...\quadraped_RL_CPG\script\xxx.py
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
        if "bittle" in n: s += 20
        if "fixed" in n: s += 10
        return s
    cands.sort(key=score, reverse=True)
    print(f"[info] USD_PATH={str(cands[0])}", flush=True)
    return str(cands[0])

# -----------------------------
# Main
# -----------------------------
def main():
    # ---- IMPORTANT: use the SAME cfg file you used before ----
    # It contains actuator settings and stand pose. Do NOT change it here.
    from bittle_cfg import BITTLE_CFG  # keep consistent with your working script

    # CPG params (same structure as your working script)
    freq_hz = 2.0
    w = 2.0 * math.pi * freq_hz
    A_sh = 0.32
    A_kn = 0.25
    knee_phase_bias = math.pi / 2.0

    SHOULDER_LIM = (-1.2, 1.2)
    KNEE_LIM = (-1.6, 1.6)

    gait_ramp_time = 2.0
    settle_time = 0.6
    T_END = 10.0

    # simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.8], [0.0, 0.0, 0.2])

    spawn_ground_and_light()

    # spawn robot (force usd path if your cfg doesn't hardcode it)
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"

    # if your BITTLE_CFG.spawn.usd_path exists, you can override to be safe:
    try:
        cfg.spawn.usd_path = auto_usd_path()
    except Exception:
        # if cfg.spawn doesn't exist in your version, ignore (your cfg already works)
        pass

    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    dt = sim.get_physics_dt()
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

    # joint map (from your printed list)
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
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

    # stand target = default pose (if your bittle_cfg sets default_joint_pos as stand)
    q_stand = default_q.clone()

    # settle on ground
    settle_steps = int(settle_time / dt)
    for _ in range(settle_steps):
        robot.set_joint_position_target(q_stand)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # debug cadence
    dbg_fast = EveryN(1)      # first second print every step
    dbg_slow = EveryN(37)     # afterwards ~0.5s
    t = 0.0
    step = 0

    while simulation_app.is_running() and t < T_END:
        phi = w * t
        ramp = clampf(t / gait_ramp_time, 0.0, 1.0)

        q_tgt = q_stand.clone()

        # diagonal trot: (LF + RB) in phase, (RF + LB) anti-phase
        sA = math.sin(phi)
        sB = math.sin(phi + math.pi)

        # right side mirrored in your working script
        L = +1.0
        R = -1.0

        # shoulders
        q_tgt[0, idx["LF_sh"]] += ramp * (L * A_sh * sA)
        q_tgt[0, idx["RB_sh"]] += ramp * (R * A_sh * sA)
        q_tgt[0, idx["RF_sh"]] += ramp * (R * A_sh * sB)
        q_tgt[0, idx["LB_sh"]] += ramp * (L * A_sh * sB)

        # knees: half-wave lift
# same-phase lift with shoulder swing groups
        liftA = max(0.0, sA)   # sA = sin(phi)
        liftB = max(0.0, sB)   # sB = sin(phi + pi)


        knee_sign = {
            "LF_kn": +1.0,
            "RF_kn": -1.0,
            "LB_kn": +1.0,
            "RB_kn": -1.0,
        }

        q_tgt[0, idx["LF_kn"]] += ramp * (knee_sign["LF_kn"] * A_kn * liftA)  # LF (A)
        q_tgt[0, idx["RB_kn"]] += ramp * (knee_sign["RB_kn"] * A_kn * liftA)  # RB (A)
        q_tgt[0, idx["RF_kn"]] += ramp * (knee_sign["RF_kn"] * A_kn * liftB)  # RF (B)
        q_tgt[0, idx["LB_kn"]] += ramp * (knee_sign["LB_kn"] * A_kn * liftB)  # LB (B)

        # clamp
        for key, jlim in [
            ("LF_sh", SHOULDER_LIM), ("RF_sh", SHOULDER_LIM), ("LB_sh", SHOULDER_LIM), ("RB_sh", SHOULDER_LIM),
            ("LF_kn", KNEE_LIM), ("RF_kn", KNEE_LIM), ("LB_kn", KNEE_LIM), ("RB_kn", KNEE_LIM),
        ]:
            j = idx[key]
            q_tgt[0, j] = clampf(float(q_tgt[0, j]), jlim[0], jlim[1])

        # ----- APPLY (this is the critical order) -----
        robot.set_joint_position_target(q_tgt)
        robot.write_data_to_sim()     # <-- THIS is why your earlier debug didn't move
        sim.step()
        robot.update(dt)

        # ----- DEBUG PRINT -----
        if t < 1.0:
            do_print = dbg_fast.hit()
        else:
            do_print = dbg_slow.hit()

        if do_print:
            rs = robot.data.root_state_w[0]
            yaw = quat_to_yaw_xyzw(rs[3:7])

            # build phase check in the same leg order used by your old debug (FL/FR/RL/RR style)
            # Here we map to: [LF, RF, LB, RB]
            phase_leg = torch.tensor([phi, phi + math.pi, phi + math.pi, phi], dtype=torch.float32)
            s = torch.sin(phase_leg)

            # actuals (read back)
            q_act = robot.data.joint_pos[0]

            hip_tgt = torch.tensor([q_tgt[0, idx["LF_sh"]], q_tgt[0, idx["RF_sh"]], q_tgt[0, idx["LB_sh"]], q_tgt[0, idx["RB_sh"]]])
            knee_tgt = torch.tensor([q_tgt[0, idx["LF_kn"]], q_tgt[0, idx["RF_kn"]], q_tgt[0, idx["LB_kn"]], q_tgt[0, idx["RB_kn"]]])
            hip_act = torch.tensor([q_act[idx["LF_sh"]], q_act[idx["RF_sh"]], q_act[idx["LB_sh"]], q_act[idx["RB_sh"]]])
            knee_act = torch.tensor([q_act[idx["LF_kn"]], q_act[idx["RF_kn"]], q_act[idx["LB_kn"]], q_act[idx["RB_kn"]]])
            pos = robot.data.root_state_w[0, 0:3]  # x,y,z
            phi_mod = (phi % (2*math.pi))
            print(f"phi_mod={phi_mod:+0.3f}  sA={math.sin(phi):+0.3f}  sB={math.sin(phi+math.pi):+0.3f}  liftA={liftA:+0.3f}  liftB={liftB:+0.3f}", flush=True)
            print(f"root_xy=({float(pos[0]):+0.3f}, {float(pos[1]):+0.3f}) z={float(pos[2]):+0.3f}", flush=True)


            print(f"\n--- step={step} t={t:.3f}s yaw={float(yaw):+0.3f} ramp={ramp:.2f} ---", flush=True)
            print(f"sin(phase_leg)[LF,RF,LB,RB]={_fmt4(s)}", flush=True)
            print(f"trot-check: sinLF-sinRB={float(s[0]-s[3]):+0.3f}, sinRF-sinLB={float(s[1]-s[2]):+0.3f}, sinLF+sinRF={float(s[0]+s[1]):+0.3f}", flush=True)
            sh_stand = torch.tensor([q_stand[0, idx["LF_sh"]], q_stand[0, idx["RF_sh"]], q_stand[0, idx["LB_sh"]], q_stand[0, idx["RB_sh"]]])
            kn_stand = torch.tensor([q_stand[0, idx["LF_kn"]], q_stand[0, idx["RF_kn"]], q_stand[0, idx["LB_kn"]], q_stand[0, idx["RB_kn"]]])

            sh_dlt_tgt = hip_tgt - sh_stand
            kn_dlt_tgt = knee_tgt - kn_stand

            print(f"sh_stand={_fmt4(sh_stand)}", flush=True)
            print(f"sh_dlt_tgt={_fmt4(sh_dlt_tgt)}", flush=True)
            print(f"kn_stand={_fmt4(kn_stand)}", flush=True)
            print(f"kn_dlt_tgt={_fmt4(kn_dlt_tgt)}", flush=True)

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
