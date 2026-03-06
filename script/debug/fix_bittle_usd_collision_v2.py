# fix_bittle_usd_collision_v4.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    import os, traceback
    from pxr import Usd, UsdGeom, UsdPhysics
    from pxr import PhysxSchema

    IN_USD  = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys.usd"
    OUT_USD = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys_col.usd"

    print("\n==============================")
    print("FIX COLLISION (v4) - apply to UsdGeom.Gprim")
    print("==============================")
    print("[info] IN_USD =", IN_USD)
    print("[info] OUT_USD=", OUT_USD)

    if not os.path.exists(IN_USD):
        print("[error] IN_USD not found.")
        return
    os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)

    try:
        stage = Usd.Stage.Open(IN_USD)
        if stage is None:
            print("[error] Failed to open stage.")
            return

        # 1) Find all prims named "collisions"
        collisions_roots = [p for p in stage.Traverse() if p.GetName() == "collisions"]
        print(f"[info] Found collisions roots: {len(collisions_roots)}")

        gprim_total = 0
        applied = 0

        # 2) Apply collision to ALL Gprims under each collisions root
        for croot in collisions_roots:
            for p in Usd.PrimRange(croot):
                g = UsdGeom.Gprim(p)
                if not g:
                    continue

                gprim_total += 1

                # Apply schemas
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(p)
                if not p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(p)

                # (optional but helpful) set collision approximation for meshes
                # If it's a Mesh, you can choose convexHull for stability:
                if p.GetTypeName() == "Mesh":
                    px = PhysxSchema.PhysxCollisionAPI(p)
                    # Only set if attribute exists
                    try:
                        px.GetCollisionApproximationAttr().Set("convexHull")
                    except Exception:
                        pass

                applied += 1

        print(f"[info] Gprim under collisions = {gprim_total}")
        print(f"[info] Collision applied prims = {applied}")

        # 3) Sanity count: how many prims now have CollisionAPI
        collision_count = 0
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
        print(f"[info] collisionAPI count (in-memory) = {collision_count}")

        # 4) Export
        ok = stage.GetRootLayer().Export(OUT_USD)
        print("[info] Export returned:", ok)
        print("[info] OUT exists:", os.path.exists(OUT_USD), " size:", os.path.getsize(OUT_USD) if os.path.exists(OUT_USD) else 0)
        if ok:
            print("[info] Exported:", OUT_USD)

    except Exception:
        print("[error] Exception occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()