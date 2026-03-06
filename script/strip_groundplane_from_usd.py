from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd

# ====== EDIT THESE PATHS ======
INPUT_USD = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_phyx.usd"
OUTPUT_USD = r"D:\Project\RLCPG\quadraped_RL_CPG\bittle\bittle_phyx_NOGROUND.usd"
# ==============================

ctx = omni.usd.get_context()
ctx.open_stage(INPUT_USD)
stage = ctx.get_stage()

print("\n========== STRIP GROUNDPLANE ==========")
print("[io] input :", INPUT_USD)
print("[io] output:", OUTPUT_USD)

# Author edits to root layer so export includes the deletions.
root_layer = stage.GetRootLayer()
stage.SetEditTarget(root_layer)

# Scan all prims named GroundPlane
targets = []
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.GetName() == "GroundPlane":
        targets.append(prim.GetPath().pathString)

print(f"[scan] found {len(targets)} prim(s) named 'GroundPlane':")
for p in targets:
    print("  -", p)

# Remove deepest first
targets_sorted = sorted(targets, key=lambda s: s.count("/"), reverse=True)

removed = []
for p in targets_sorted:
    prim = stage.GetPrimAtPath(p)
    if prim and prim.IsValid():
        stage.RemovePrim(p)
        removed.append(p)

print(f"[fix] removed {len(removed)} prim(s):")
for p in removed:
    print("  -", p)

# Verify remaining
remain = []
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.GetName() == "GroundPlane":
        remain.append(prim.GetPath().pathString)
print(f"[verify] remaining GroundPlane prim(s): {remain}")

# ---- Save to new file (version-safe) ----
saved = False

# 1) Prefer Layer.Export (acts like SaveAs)
if hasattr(root_layer, "Export"):
    try:
        root_layer.Export(OUTPUT_USD)
        print("[save] root_layer.Export ->", OUTPUT_USD)
        saved = True
    except Exception as e:
        print("[save] root_layer.Export failed:", repr(e))

# 2) Fallback: stage.Export (if available)
if (not saved) and hasattr(stage, "Export"):
    try:
        stage.Export(OUTPUT_USD)
        print("[save] stage.Export ->", OUTPUT_USD)
        saved = True
    except Exception as e:
        print("[save] stage.Export failed:", repr(e))

# 3) Last resort: overwrite original
if not saved:
    print("[save] WARNING: no Export/SaveAs available; falling back to root_layer.Save() (overwrites input).")
    root_layer.Save()
    print("[save] root_layer.Save() done. (input overwritten)")

simulation_app.close()