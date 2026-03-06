# fix_bittle_usd_scale_collision_v3.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    import os, traceback
    from pxr import Usd, UsdGeom, UsdPhysics
    from pxr import PhysxSchema

    IN_USD  = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle.usd"
    OUT_USD = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys.usd"
    ROBOT_ROOT = "/bittle"
    UNIFORM_SCALE = 0.12

    print("\n==============================")
    print("FIX USD (v3) - idempotent")
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

        root = stage.GetPrimAtPath(ROBOT_ROOT)
        if not root:
            print("[error] Root prim not found:", ROBOT_ROOT)
            print("[info] top-level prims:")
            for p in stage.GetPseudoRoot().GetChildren():
                print(" -", p.GetPath())
            return

        # -------------------------
        # 1) Set/ensure uniform scale on /bittle
        # -------------------------
        xform = UsdGeom.Xformable(root)

        # Find an existing scale op if present
        scale_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op is None:
            # No scale op yet -> add one
            scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            print("[info] Added new scale op.")
        else:
            print(f"[info] Found existing scale op: {scale_op.GetName()}")

        scale_op.Set((UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE))
        print(f"[info] Set {ROBOT_ROOT} scale = {UNIFORM_SCALE}")

        # -------------------------
        # 2) Apply CollisionAPI under */collisions
        # -------------------------
        collisions_roots = [p for p in stage.Traverse() if p.GetName() == "collisions"]
        print(f"[info] Found collisions roots: {len(collisions_roots)}")

        mesh_like = 0
        applied = 0

        def apply_collision_to_geom(p):
            nonlocal mesh_like, applied
            if p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"):
                mesh_like += 1
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(p)
                if not p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(p)
                applied += 1

        for croot in collisions_roots:
            for p in Usd.PrimRange(croot):
                apply_collision_to_geom(p)

        print(f"[info] Geom candidates under collisions: {mesh_like}")
        print(f"[info] Collision applied to: {applied}")

        # -------------------------
        # 3) Export + verify
        # -------------------------
        ok = stage.GetRootLayer().Export(OUT_USD)
        print("[info] Export returned:", ok)

        exists = os.path.exists(OUT_USD)
        size = os.path.getsize(OUT_USD) if exists else 0
        print("[info] OUT exists:", exists, " size(bytes):", size)
        if exists:
            print("[info] Exported:", OUT_USD)
        else:
            print("[error] Export did not create file (permissions/path).")

    except Exception:
        print("[error] Exception occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()