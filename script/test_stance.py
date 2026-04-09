#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bittle stance / alternating diagonal support test for Isaac Lab.

What this version changes
-------------------------
1) It does not trust the USD path in bittle_cfg.py blindly.
   If bittle_cfg.py points to a tiny top-level "bittle.usd" and a sibling
   configuration/bittle_physics.usd exists, this script automatically prefers
   bittle_physics.usd.
2) It audits articulation-root prims in the chosen USD before spawning.
   If the asset still contains nested articulation roots, the script stops with
   a clean error instead of crashing later with a vague PhysX message.
3) It keeps the support/lift test logic simple and self-contained. It does not
   query old visual prim paths, so stale path assumptions are removed.

How to run
----------
isaaclab.bat -p D:\Project\RLCPG\quadraped_RL_CPG\script\test_stance_rewrite.py

Optional override
-----------------
You can force a specific USD with an environment variable before launch:
set BITTLE_USD_PATH=D:\Project\RLCPG\quadraped_RL_CPG\bittle\configuration\bittle_physics.usd
"""

from __future__ import annotations

import csv
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Isaac Sim rule: create SimulationApp before omni / isaaclab imports.
from omni.isaac.kit import SimulationApp

HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))
simulation_app = SimulationApp({"headless": HEADLESS})

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdLux, UsdPhysics  # noqa: E402

try:  # noqa: E402
    from pxr import PhysxSchema  # type: ignore
except Exception:  # noqa: E402
    PhysxSchema = None

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from bittle_cfg import BITTLE_CFG, STAND_JOINT_POS  # noqa: E402


# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
@dataclass
class ManualParams:
    # Mild default support offsets. Start conservative.
    front_sh_y: float = -0.45
    back_sh_y: float = -0.45
    front_kn_crouch: float = -0.4
    back_kn_crouch: float = -0.4

    # Extra offsets applied only to the currently lifted diagonal pair.
    lift_front_sh_y: float = 0.00
    lift_back_sh_y: float = 0.00
    lift_front_kn_flex: float = 0.35
    lift_back_kn_flex: float = 0.39

    start_pair: str = "LF_RB"  # "LF_RB" or "RF_LB"
    alternate_pairs: bool = True

    # Timing (seconds)
    stand_settle_time: float = 0.8
    support_settle_time: float = 1.0
    pre_hold_time: float = 0.25
    lift_time: float = 0.20
    hold_time: float = 0.30
    down_time: float = 0.20
    post_hold_time: float = 1
    cycles: int = 20

    # Spawn / sim
    root_z0: float = 1.0
    dt: float = 1.0 / 120.0
    device: str = os.environ.get("BITTLE_DEVICE", "cuda:0")
    save_dir_name: str = "support_test_outputs_rewrite"


P = ManualParams()

LEGS = ["LF", "RF", "LB", "RB"]
FRONT_LEGS = ["LF", "RF"]
BACK_LEGS = ["LB", "RB"]
PAIR_ORDER = ["LF_RB", "RF_LB"]

# Interface sign conventions already validated in your Bittle route.
SH_SIGN = {"LF": -1.0, "LB": -1.0, "RF": +1.0, "RB": +1.0}
KN_SIGN = {"LF": -1.0, "LB": -1.0, "RF": +1.0, "RB": +1.0}

SHOULDER_LIM = (-1.15, 1.15)
KNEE_LIM = (-1.55, 1.55)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def smoothstep(s: float) -> float:
    s = clampf(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def quat_to_euler_xyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def get_lift_legs(pair: str) -> List[str]:
    if pair == "LF_RB":
        return ["LF", "RB"]
    if pair == "RF_LB":
        return ["RF", "LB"]
    raise ValueError(f"Unsupported pair={pair!r}")


def get_support_legs(pair: str) -> List[str]:
    lifted = set(get_lift_legs(pair))
    return [leg for leg in LEGS if leg not in lifted]


def get_pair_for_cycle(cycle_idx: int) -> str:
    start_idx = PAIR_ORDER.index(P.start_pair)
    if not P.alternate_pairs:
        return P.start_pair
    return PAIR_ORDER[(start_idx + cycle_idx) % 2]


# -----------------------------------------------------------------------------
# USD preflight
# -----------------------------------------------------------------------------
def resolve_usd_path(cfg_usd_path: str) -> str:
    """
    Prefer a physics-only layer automatically when the cfg points to a tiny
    top-level bittle.usd entry layer.
    Priority:
      1) BITTLE_USD_PATH env override
      2) configuration/bittle_physics.usd beside cfg_usd_path when cfg ends in bittle.usd
      3) original cfg_usd_path
    """
    env_override = os.environ.get("BITTLE_USD_PATH", "").strip()
    if env_override:
        return env_override.replace("\\", "/")

    p = Path(cfg_usd_path)
    if p.name.lower() == "bittle.usd":
        candidate = p.parent / "configuration" / "bittle_physics.usd"
        if candidate.exists():
            return candidate.as_posix()

    return p.as_posix()


def collect_articulation_like_prims(stage: Usd.Stage) -> List[dict]:
    rows: List[dict] = []
    for prim in stage.Traverse():
        has_usd_root = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        has_physx_art = False
        if PhysxSchema is not None:
            try:
                has_physx_art = prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
            except Exception:
                has_physx_art = False
        if has_usd_root or has_physx_art:
            rows.append(
                {
                    "path": str(prim.GetPath()),
                    "usd_root": has_usd_root,
                    "physx_art": has_physx_art,
                }
            )
    return rows


def find_nested_paths(paths: List[str]) -> List[Tuple[str, str]]:
    nested: List[Tuple[str, str]] = []
    spaths = sorted(paths)
    for i, a in enumerate(spaths):
        for b in spaths[i + 1 :]:
            if b.startswith(a + "/"):
                nested.append((a, b))
    return nested


def preflight_usd(usd_path: str) -> List[dict]:
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    art_rows = collect_articulation_like_prims(stage)
    usd_root_paths = [r["path"] for r in art_rows if r["usd_root"]]
    nested = find_nested_paths(usd_root_paths)

    print("\n=== USD PREFLIGHT ===")
    print(f"[resolved_usd_path] {usd_path}")
    print(f"[articulation_like_prim_count] {len(art_rows)}")
    for row in art_rows:
        print(
            f"  {row['path']} | "
            f"UsdPhysics.ArticulationRootAPI={row['usd_root']} | "
            f"PhysxArticulationAPI={row['physx_art']}"
        )

    if nested:
        print("\n[nested_usd_articulation_roots]")
        for parent, child in nested:
            print(f"  parent={parent} | child={child}")
        raise RuntimeError(
            "Asset still contains nested UsdPhysics articulation roots. "
            "This is an asset problem, not a stance-script problem. "
            "Point to a clean physics layer or remove the extra inner articulation root."
        )
    return art_rows


# -----------------------------------------------------------------------------
# Joint helpers
# -----------------------------------------------------------------------------
def build_joint_maps(joint_names: List[str]):
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    needed = {
        "LF_sh": "left_front_shoulder_joint",
        "RF_sh": "right_front_shoulder_joint",
        "LB_sh": "left_back_shoulder_joint",
        "RB_sh": "right_back_shoulder_joint",
        "LF_kn": "left_front_knee_joint",
        "RF_kn": "right_front_knee_joint",
        "LB_kn": "left_back_knee_joint",
        "RB_kn": "right_back_knee_joint",
    }
    missing = [name for name in needed.values() if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in asset: {missing}")

    idx = {k: name_to_idx[v] for k, v in needed.items()}
    leg_to_sh = {"LF": idx["LF_sh"], "RF": idx["RF_sh"], "LB": idx["LB_sh"], "RB": idx["RB_sh"]}
    leg_to_kn = {"LF": idx["LF_kn"], "RF": idx["RF_kn"], "LB": idx["LB_kn"], "RB": idx["RB_kn"]}
    return idx, leg_to_sh, leg_to_kn


def build_stand_pose_from_cfg(jmap: Dict[str, int], default_joint_pos: torch.Tensor) -> torch.Tensor:
    q = default_joint_pos.clone()
    for jn, val in STAND_JOINT_POS.items():
        if jn in jmap:
            q[0, jmap[jn]] = float(val)
    return q


def clamp_joint_limits(q: torch.Tensor, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int]) -> torch.Tensor:
    q = q.clone()
    for leg in LEGS:
        q[0, leg_to_sh[leg]] = clampf(float(q[0, leg_to_sh[leg]]), *SHOULDER_LIM)
        q[0, leg_to_kn[leg]] = clampf(float(q[0, leg_to_kn[leg]]), *KNEE_LIM)
    return q


def apply_support_posture(q_stand: torch.Tensor, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int]) -> torch.Tensor:
    q = q_stand.clone()

    for leg in FRONT_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * P.front_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * P.front_kn_crouch

    for leg in BACK_LEGS:
        q[0, leg_to_sh[leg]] += SH_SIGN[leg] * P.back_sh_y
        q[0, leg_to_kn[leg]] -= KN_SIGN[leg] * P.back_kn_crouch

    return clamp_joint_limits(q, leg_to_sh, leg_to_kn)


def apply_lift_pair(q_support: torch.Tensor, pair: str, leg_to_sh: Dict[str, int], leg_to_kn: Dict[str, int]) -> torch.Tensor:
    q = q_support.clone()
    for leg in get_lift_legs(pair):
        if leg in FRONT_LEGS:
            q[0, leg_to_sh[leg]] += SH_SIGN[leg] * P.lift_front_sh_y
            q[0, leg_to_kn[leg]] += KN_SIGN[leg] * P.lift_front_kn_flex
        else:
            q[0, leg_to_sh[leg]] += SH_SIGN[leg] * P.lift_back_sh_y
            q[0, leg_to_kn[leg]] += KN_SIGN[leg] * P.lift_back_kn_flex
    return clamp_joint_limits(q, leg_to_sh, leg_to_kn)


# -----------------------------------------------------------------------------
# Stage setup
# -----------------------------------------------------------------------------
def spawn_ground_plane(stage):
    import omni.physx.scripts.physicsUtils as physicsUtils

    fn = physicsUtils.add_ground_plane
    sig = inspect.signature(fn)

    plane_path = "/World/GroundPlane"
    color = (0.16, 0.16, 0.16)
    kwargs = {}
    for name in sig.parameters.keys():
        low = name.lower()
        if "stage" in low:
            kwargs[name] = stage
        elif "planepath" in low or (("path" in low or "prim" in low) and "plane" in low):
            kwargs[name] = plane_path
        elif "color" in low or "colour" in low:
            kwargs[name] = color
        elif "axis" in low:
            kwargs[name] = "Z"
        elif "size" in low or "halfsize" in low:
            kwargs[name] = 50.0
        elif "position" in low:
            kwargs[name] = (0.0, 0.0, 0.0)
        elif "normal" in low:
            kwargs[name] = (0.0, 0.0, 1.0)
        elif ("static" in low and "friction" in low) or low == "staticfriction":
            kwargs[name] = 2.0
        elif ("dynamic" in low and "friction" in low) or low == "dynamicfriction":
            kwargs[name] = 1.8
        elif "restitution" in low:
            kwargs[name] = 0.0

    if "planePath" in sig.parameters and "planePath" not in kwargs:
        kwargs["planePath"] = plane_path
    if "color" in sig.parameters and "color" not in kwargs:
        kwargs["color"] = color

    fn(**kwargs)


def spawn_light(stage):
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(0.5)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = BITTLE_CFG.copy()
    original_usd = str(cfg.spawn.usd_path)
    resolved_usd = resolve_usd_path(original_usd)
    preflight_usd(resolved_usd)

    cfg.spawn.usd_path = resolved_usd
    cfg.init_state.pos = (0.0, 0.0, P.root_z0)

    sim_cfg = sim_utils.SimulationCfg(dt=P.dt, device=P.device)
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    sim.set_camera_view([1.8, 1.8, 1.1], [0.0, 0.0, 0.12])

    spawn_ground_plane(stage)
    spawn_light(stage)

    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    robot.update(P.dt)

    default_root = robot.data.default_root_state.clone()
    default_q = robot.data.default_joint_pos.clone()
    default_qd = robot.data.default_joint_vel.clone()

    default_root[:, 0] = 0.0
    default_root[:, 1] = 0.0
    default_root[:, 2] = P.root_z0

    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(default_q, default_qd)
    robot.write_data_to_sim()
    for _ in range(2):
        sim.step()
        robot.update(P.dt)
    robot.reset()

    joint_names = list(robot.data.joint_names)
    jmap = {n: i for i, n in enumerate(joint_names)}
    _, leg_to_sh, leg_to_kn = build_joint_maps(joint_names)

    q_stand = build_stand_pose_from_cfg(jmap, robot.data.default_joint_pos)
    q_support = apply_support_posture(q_stand, leg_to_sh, leg_to_kn)

    steps_stand = max(1, int(P.stand_settle_time / P.dt))
    steps_support = max(1, int(P.support_settle_time / P.dt))
    steps_pre = max(1, int(P.pre_hold_time / P.dt))
    steps_up = max(1, int(P.lift_time / P.dt))
    steps_hold = max(1, int(P.hold_time / P.dt))
    steps_down = max(1, int(P.down_time / P.dt))
    steps_post = max(1, int(P.post_hold_time / P.dt))

    output_dir = Path(__file__).resolve().parent / P.save_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "stance_test_metrics.csv"
    cycle_csv = output_dir / "stance_test_cycle_summary.csv"
    config_txt = output_dir / "stance_test_run_info.txt"

    metrics_rows: List[Dict] = []
    cycle_summaries: List[Dict] = []
    t = 0.0

    def step_with_target(q_cmd: torch.Tensor):
        nonlocal t
        robot.set_joint_position_target(q_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(P.dt)
        t += P.dt

    def sample_metrics(phase: str, cycle_idx: int, lift_pair: str, ref_root_z: float, ref_x: float, ref_y: float):
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()

        roll, pitch, yaw = quat_to_euler_xyz(
            float(root_quat[0]), float(root_quat[1]), float(root_quat[2]), float(root_quat[3])
        )
        x = float(root_pos[0])
        y = float(root_pos[1])
        z = float(root_pos[2])
        dx = x - ref_x
        dy = y - ref_y
        z_drop = ref_root_z - z
        xy_drift = math.hypot(dx, dy)
        vz = float(root_lin_vel[2])

        metrics_rows.append(
            {
                "t": t,
                "cycle": cycle_idx,
                "phase": phase,
                "lift_pair": lift_pair,
                "lift_legs": "+".join(get_lift_legs(lift_pair)),
                "support_pair": "+".join(get_support_legs(lift_pair)),
                "root_x": x,
                "root_y": y,
                "root_z": z,
                "dx": dx,
                "dy": dy,
                "xy_drift": xy_drift,
                "z_drop": z_drop,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "vz": vz,
            }
        )

    def summarize_cycle_rows(cycle_idx: int) -> Dict[str, float]:
        cyc_rows = [r for r in metrics_rows if int(r["cycle"]) == cycle_idx]
        if not cyc_rows:
            raise RuntimeError(f"No rows collected for cycle {cycle_idx}.")

        def max_abs(key: str, phase: str | None = None) -> float:
            vals = [abs(float(r[key])) for r in cyc_rows if phase is None or r["phase"] == phase]
            return max(vals) if vals else 0.0

        def max_pos(key: str, phase: str | None = None) -> float:
            vals = [float(r[key]) for r in cyc_rows if phase is None or r["phase"] == phase]
            return max(vals) if vals else 0.0

        pair = str(cyc_rows[0]["lift_pair"])
        support_pair = str(cyc_rows[0]["support_pair"])
        return {
            "cycle": cycle_idx,
            "lift_pair": pair,
            "support_pair": support_pair,
            "lift_hold_max_z_drop": max_pos("z_drop", phase="lift_hold"),
            "lift_hold_max_abs_dx": max_abs("dx", phase="lift_hold"),
            "lift_hold_max_abs_dy": max_abs("dy", phase="lift_hold"),
            "lift_hold_max_abs_xy_drift": max_abs("xy_drift", phase="lift_hold"),
            "lift_hold_max_abs_roll": max_abs("roll", phase="lift_hold"),
            "lift_hold_max_abs_pitch": max_abs("pitch", phase="lift_hold"),
            "lift_hold_max_abs_yaw": max_abs("yaw", phase="lift_hold"),
            "lift_hold_max_abs_vz": max_abs("vz", phase="lift_hold"),
        }

    print("\n=== bittle_stance_test_rewrite ===")
    print(f"[cfg_usd_path] {original_usd}")
    print(f"[resolved_usd_path] {resolved_usd}")
    print(f"[prim_path] {cfg.prim_path}")
    print(f"[device] {P.device}")
    print(f"[root_z0] {P.root_z0:.3f}")
    print(
        f"[support posture] front_sh_y={P.front_sh_y:+.3f} back_sh_y={P.back_sh_y:+.3f} | "
        f"front_kn_crouch={P.front_kn_crouch:+.3f} back_kn_crouch={P.back_kn_crouch:+.3f}"
    )
    print(
        f"[lift offsets] lift_front_sh_y={P.lift_front_sh_y:+.3f} lift_back_sh_y={P.lift_back_sh_y:+.3f} | "
        f"lift_front_kn_flex={P.lift_front_kn_flex:+.3f} lift_back_kn_flex={P.lift_back_kn_flex:+.3f}"
    )

    # Stand then settle to support posture.
    for _ in range(steps_stand):
        step_with_target(q_stand)
    for _ in range(steps_support):
        step_with_target(q_support)

    print("\n[support targets]")
    for leg in LEGS:
        print(
            f"  {leg}: sh={float(q_support[0, leg_to_sh[leg]]):+.3f} | "
            f"kn={float(q_support[0, leg_to_kn[leg]]):+.3f}"
        )

    for cyc in range(P.cycles):
        lift_pair = get_pair_for_cycle(cyc)
        q_lift = apply_lift_pair(q_support, lift_pair, leg_to_sh, leg_to_kn)

        for _ in range(steps_pre):
            step_with_target(q_support)

        root_pos_ref = robot.data.root_pos_w[0].detach().cpu().numpy()
        ref_x = float(root_pos_ref[0])
        ref_y = float(root_pos_ref[1])
        ref_root_z = float(root_pos_ref[2])

        sample_metrics("support_ref", cyc, lift_pair, ref_root_z, ref_x, ref_y)

        for i in range(steps_up):
            s = smoothstep((i + 1) / steps_up)
            q_cmd = (1.0 - s) * q_support + s * q_lift
            q_cmd = clamp_joint_limits(q_cmd, leg_to_sh, leg_to_kn)
            step_with_target(q_cmd)
            sample_metrics("lift_up", cyc, lift_pair, ref_root_z, ref_x, ref_y)

        for _ in range(steps_hold):
            step_with_target(q_lift)
            sample_metrics("lift_hold", cyc, lift_pair, ref_root_z, ref_x, ref_y)

        for i in range(steps_down):
            s = smoothstep((i + 1) / steps_down)
            q_cmd = (1.0 - s) * q_lift + s * q_support
            q_cmd = clamp_joint_limits(q_cmd, leg_to_sh, leg_to_kn)
            step_with_target(q_cmd)
            sample_metrics("down", cyc, lift_pair, ref_root_z, ref_x, ref_y)

        for _ in range(steps_post):
            step_with_target(q_support)
            sample_metrics("post_hold", cyc, lift_pair, ref_root_z, ref_x, ref_y)

        cyc_summary = summarize_cycle_rows(cyc)
        cycle_summaries.append(cyc_summary)
        print(
            f"[cycle {cyc:02d}] lift={lift_pair} | "
            f"z_drop={cyc_summary['lift_hold_max_z_drop']:.4f} | "
            f"|dx|={cyc_summary['lift_hold_max_abs_dx']:.4f} | "
            f"|dy|={cyc_summary['lift_hold_max_abs_dy']:.4f} | "
            f"|roll|={cyc_summary['lift_hold_max_abs_roll']:.4f} | "
            f"|pitch|={cyc_summary['lift_hold_max_abs_pitch']:.4f}"
        )

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    with cycle_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cycle_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(cycle_summaries)

    with config_txt.open("w", encoding="utf-8") as f:
        f.write("stance_test_rewrite\n")
        f.write(f"cfg_usd_path = {original_usd}\n")
        f.write(f"resolved_usd_path = {resolved_usd}\n")
        f.write(f"prim_path = {cfg.prim_path}\n")
        for k, v in vars(P).items():
            f.write(f"{k} = {v}\n")

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {cycle_csv}")
    print(f"Saved: {config_txt}")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
