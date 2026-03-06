from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    from pxr import Usd, UsdGeom, UsdPhysics
    from pxr import PhysxSchema

    USD_PATH = r"D:\PATH\TO\YOUR\NEW_EXPORT.usd"  # 改这里
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        print("[error] failed to open:", USD_PATH, flush=True)
        return

    print("USD_PATH =", USD_PATH)

    hits = {
        "UsdPhysics.CollisionAPI": [],
        "PhysxSchema.PhysxCollisionAPI": [],
        "PhysxSchema.PhysxRigidBodyAPI": [],
        "UsdPhysics.MeshCollisionAPI": [],
        "UsdPhysics.CollisionGroup": [],
    }

    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            hits["UsdPhysics.CollisionAPI"].append(str(p.GetPath()))
        if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
            hits["PhysxSchema.PhysxCollisionAPI"].append(str(p.GetPath()))
        if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            hits["PhysxSchema.PhysxRigidBodyAPI"].append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            hits["UsdPhysics.MeshCollisionAPI"].append(str(p.GetPath()))
        # collision groups are typed prims
        if p.GetTypeName() in ("PhysicsCollisionGroup",):
            hits["UsdPhysics.CollisionGroup"].append(str(p.GetPath()))

    print("\n=== COLLIDER SCHEMA HITS ===")
    for k, v in hits.items():
        print(f"{k}: {len(v)}")
        for path in v[:20]:
            print(" -", path)
        if len(v) > 20:
            print(" ...")

    # Also check if ANY Gprim has physics:collisionEnabled attr (some pipelines use attrs)
    print("\n=== Gprim collisionEnabled attr check (first 20) ===")
    shown = 0
    for p in stage.Traverse():
        g = UsdGeom.Gprim(p)
        if not g:
            continue
        attr = p.GetAttribute("physics:collisionEnabled")
        if attr and attr.HasAuthoredValueOpinion():
            print(" -", p.GetPath(), "physics:collisionEnabled =", attr.Get())
            shown += 1
            if shown >= 20:
                break

    print("\nDone.\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()