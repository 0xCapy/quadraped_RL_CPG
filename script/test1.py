from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    from pxr import Usd, UsdGeom, UsdPhysics
    from pxr import PhysxSchema

    USD_PATH = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_hull.usd" 
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        print("[error] failed to open:", USD_PATH, flush=True)
        return

    print("\n=== USD OPENED ===")
    print("USD_PATH =", USD_PATH)

    # counts
    cnt = {
        "UsdPhysics.CollisionAPI": 0,
        "PhysxSchema.PhysxCollisionAPI": 0,
        "UsdPhysics.MeshCollisionAPI": 0,
        "UsdPhysics.RigidBodyAPI": 0,
        "UsdPhysics.ArticulationRootAPI": 0,
        "UsdPhysics.Joint": 0,
    }

    collisions_roots = []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            cnt["UsdPhysics.CollisionAPI"] += 1
        if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
            cnt["PhysxSchema.PhysxCollisionAPI"] += 1
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            cnt["UsdPhysics.MeshCollisionAPI"] += 1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            cnt["UsdPhysics.RigidBodyAPI"] += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            cnt["UsdPhysics.ArticulationRootAPI"] += 1
        if p.IsA(UsdPhysics.Joint):
            cnt["UsdPhysics.Joint"] += 1
        if p.GetName() == "collisions":
            collisions_roots.append(p)

    print("\n=== PHYSICS SCHEMA COUNTS ===")
    for k, v in cnt.items():
        print(f"{k}: {v}")

    print("\n=== COLLISIONS ROOTS (first 5) ===")
    print("collisions roots count =", len(collisions_roots))
    for i, c in enumerate(collisions_roots[:5]):
        local_mesh = 0
        local_gprim = 0
        local_total = 0
        for q in Usd.PrimRange(c):
            local_total += 1
            if q.GetTypeName() == "Mesh":
                local_mesh += 1
            if UsdGeom.Gprim(q):
                local_gprim += 1
        print(f"[{i}] {c.GetPath()} total={local_total} Gprim={local_gprim} Mesh={local_mesh}")

    print("\nDone.\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()