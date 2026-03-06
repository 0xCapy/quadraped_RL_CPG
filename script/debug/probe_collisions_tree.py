from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

def main():
    from pxr import Usd, UsdGeom, UsdPhysics

    USD_PATH = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_fixed_phys.usd"
    ROOT = "/bittle"

    stage = Usd.Stage.Open(USD_PATH)
    print("USD_PATH =", USD_PATH)
    print("metersPerUnit =", stage.GetMetadata("metersPerUnit"))

    # find all ".../collisions" prims
    col_roots = []
    for p in stage.Traverse():
        if p.GetName() == "collisions":
            col_roots.append(p)

    print("\n[collisions roots] count =", len(col_roots))
    for i, c in enumerate(col_roots[:20]):
        print(f"  {i:02d}:", c.GetPath(), "type=", c.GetTypeName(),
              " instanceProxy=", c.IsInstanceProxy(),
              " instanceable=", c.IsInstanceable())

    # dump one collisions subtree deeper (first one)
    if not col_roots:
        return

    c0 = col_roots[0]
    print("\n[Dump first collisions subtree]")
    max_print = 200
    n = 0
    gprim_n = 0
    boundable_n = 0
    mesh_n = 0

    for p in Usd.PrimRange(c0):
        n += 1
        if n <= max_print:
            is_gprim = bool(UsdGeom.Gprim(p))
            is_boundable = bool(UsdGeom.Boundable(p))
            has_col_api = p.HasAPI(UsdPhysics.CollisionAPI)
            print("-", p.GetPath(), "type=", p.GetTypeName(),
                  "Gprim=", is_gprim,
                  "Boundable=", is_boundable,
                  "HasCollisionAPI=", has_col_api,
                  "instanceProxy=", p.IsInstanceProxy())
        if UsdGeom.Gprim(p):
            gprim_n += 1
        if UsdGeom.Boundable(p):
            boundable_n += 1
        if p.GetTypeName() == "Mesh":
            mesh_n += 1

    print("\n[Subtree stats]")
    print("  total prims =", n)
    print("  Gprim       =", gprim_n)
    print("  Boundable   =", boundable_n)
    print("  Mesh        =", mesh_n)

    # global collision API count
    total_col_api = 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            total_col_api += 1
    print("\n[Global] collisionAPI count =", total_col_api)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()