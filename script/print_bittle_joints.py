# print_bittle_joints.py
import argparse
from pathlib import Path

# ---- MUST: start SimulationApp BEFORE any Omniverse/pxr imports ----
from isaacsim import SimulationApp  # noqa

def main():
    parser = argparse.ArgumentParser()
    # 这里不用 AppLauncher，直接用 SimulationApp 更直观、更稳
    parser.add_argument("--headless", action="store_true", help="Run headless")
    args, _ = parser.parse_known_args()

    simulation_app = SimulationApp({"headless": args.headless})

    # ---- Now it's safe to import isaaclab / pxr related modules ----
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import Articulation

    # 载入你的 cfg（确保 bittle_cfg.py 里 USD 路径已改对）
    from bittle_cfg import BITTLE_CFG

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)

    # spawn robot
    cfg = BITTLE_CFG.copy()
    cfg.prim_path = "/World/Bittle"
    robot = Articulation(cfg=cfg)

    sim.reset()
    sim.step()
    robot.update(sim.get_physics_dt())

    print("\n========== Bittle Joint Names ==========")
    for i, n in enumerate(robot.joint_names):
        print(f"{i:02d}: {n}")
    print(f"Total joints: {len(robot.joint_names)}")

    simulation_app.close()


if __name__ == "__main__":
    main()
