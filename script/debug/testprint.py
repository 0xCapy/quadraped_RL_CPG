# usd_probe_bittle.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})  # 诊断用 headless 更快

def main():
    from pxr import Usd, UsdGeom, UsdPhysics

    USD_PATH = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys_col.usd"
    ROBOT_PRIM_PATH = "/bittle"

    print("\n==============================")
    print("USD PROBE (v2)")
    print("==============================")
    print(f"USD_PATH = {USD_PATH}")

    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        print("[error] Failed to open USD.")
        return

    print("\n[Stage Metadata]")
    print("  metersPerUnit =", stage.GetMetadata("metersPerUnit"))
    print("  upAxis        =", UsdGeom.GetStageUpAxis(stage))

    prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    print("\n[Robot Prim Check]")
    print("  prim_path  =", ROBOT_PRIM_PATH)
    print("  prim_valid =", bool(prim))
    if not prim:
        print("  top-level prims:")
        for p in stage.GetPseudoRoot().GetChildren():
            print("   -", p.GetPath())
        return

    # ---- BBox size (world) ----
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.GetRange()
    mn = rng.GetMin()
    mx = rng.GetMax()
    size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    print("\n[BBox world]")
    print(f"  min = ({mn[0]:+.6f}, {mn[1]:+.6f}, {mn[2]:+.6f})")
    print(f"  max = ({mx[0]:+.6f}, {mx[1]:+.6f}, {mx[2]:+.6f})")
    print(f"  size(dx,dy,dz) = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})")
    print("  note: Bittle should be ~0.1m scale in dz, not ~1.0m.")

    # ---- Print prim tree (2 levels) ----
    print("\n[Prim tree under /bittle (depth<=2)]")
    def print_tree(p, depth=0, max_depth=2):
        if depth > max_depth:
            return
        indent = "  " * depth
        print(f"{indent}- {p.GetPath()}  type={p.GetTypeName()}")
        for c in p.GetChildren():
            print_tree(c, depth + 1, max_depth)

    print_tree(prim, 0, 2)

    # ---- Scan stage for physics schemas ----
    joint_prims = []
    art_roots = []
    rigid_bodies = 0
    collisions = 0

    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            joint_prims.append(p)
        # articulation root is an API schema, check via HasAPI
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(p)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies += 1
        if p.HasAPI(UsdPhysics.CollisionAPI):
            collisions += 1

    print("\n[Physics summary]")
    print("  articulationRootAPI count =", len(art_roots))
    for r in art_roots[:10]:
        print("   -", r.GetPath())
    if len(art_roots) > 10:
        print("   ...")

    print("  rigidBodyAPI count        =", rigid_bodies)
    print("  collisionAPI count        =", collisions)
    print("  UsdPhysics.Joint count    =", len(joint_prims))

    # ---- If joints exist, dump axis/limits for first 30 ----
    if joint_prims:
        print("\n[Joint dump (first 30)]")
        for p in joint_prims[:30]:
            name = p.GetName()
            path = str(p.GetPath())
            axis = None
            lo = None
            hi = None
            try:
                rj = UsdPhysics.RevoluteJoint(p)
                if rj:
                    axis = rj.GetAxisAttr().Get()
                    lo = rj.GetLowerLimitAttr().Get()
                    hi = rj.GetUpperLimitAttr().Get()
            except Exception:
                pass
            print(f"  - {name:40s}  {path}")
            if axis is not None:
                print(f"      axis={axis}  limit_deg=({lo},{hi})")

    print("\nDone.\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()