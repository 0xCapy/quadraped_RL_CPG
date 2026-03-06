from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    import os
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from pxr import PhysxSchema

    IN_USD  = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys.usd"
    OUT_USD = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys_boxcol.usd"
    ROBOT_ROOT = "/bittle"
    COLL_ROOT = "/bittle/_colliders"

    os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)

    stage = Usd.Stage.Open(IN_USD)
    if stage is None:
        print("[error] cannot open", IN_USD, flush=True); return

    stage.SetEditTarget(stage.GetRootLayer())

    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root:
        print("[error] missing", ROBOT_ROOT, flush=True); return

    # create a non-instanced container
    UsdGeom.Xform.Define(stage, COLL_ROOT)
    coll_root = stage.GetPrimAtPath(COLL_ROOT)
    # ensure it's not instanceable
    coll_root.SetInstanceable(False)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

    links = [p for p in root.GetChildren() if p.GetName().endswith("_link")]
    print("[info] link count =", len(links), flush=True)

    created = 0
    for link in links:
        link_path = str(link.GetPath())
        name = link.GetName()

        bbox = bbox_cache.ComputeWorldBound(link)
        rng = bbox.GetRange()
        mn = rng.GetMin()
        mx = rng.GetMax()

        size = Gf.Vec3d(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
        center = Gf.Vec3d((mx[0]+mn[0])*0.5, (mx[1]+mn[1])*0.5, (mx[2]+mn[2])*0.5)

        if size[0] < 1e-4 or size[1] < 1e-4 or size[2] < 1e-4:
            continue

        cube_path = f"{COLL_ROOT}/{name}_box"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(center)
        xf.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())

        created += 1

    print("[info] created colliders =", created, flush=True)

    col_count = 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            col_count += 1
    print("[info] collisionAPI count =", col_count, flush=True)

    ok = stage.GetRootLayer().Export(OUT_USD)
    print("[info] exported =", ok, "->", OUT_USD, flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()