"""
fix_corners.py
--------------
Removes the rounded ends (top and bottom in Y) of the phone case STL
and replaces them with sharp 90-degree flat caps.

Strategy:
  1. Find where the outer X-span reaches its full width (flat section starts)
     → bottom flat starts at Y ≈ -67.5, top flat ends at Y ≈ 71.5
  2. Clip the mesh at those Y planes (removing the rounded taper)
  3. Fill the open boundaries with flat triangulated caps
  4. Save to a new STL
"""

import pyvista as pv
import numpy as np

INPUT  = 'day-10-phone-case-v7.stl'
OUTPUT = 'day-10-phone-case-v7-sharp.stl'

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
print("Loading case...")
case = pv.read(INPUT)
bounds = case.bounds
y_min, y_max = bounds[2], bounds[3]
print(f"  Original Y range: {y_min:.2f} → {y_max:.2f}  (height {y_max-y_min:.2f} mm)")

# ------------------------------------------------------------------
# 2. Find the flat-region boundaries
#    From the scan: full X-span (82.48mm) is reached at:
#      bottom: Y ≈ -62.87  (use -63.0 to be safe, keep a tiny margin)
#      top:    Y ≈  69.79  (use  70.0)
# ------------------------------------------------------------------
# Scan precisely to find the first Y (from each end) where X-span == full
full_span = bounds[1] - bounds[0]          # 82.48 mm
tolerance = 1.0                            # within 1mm of full width → "flat"

print("\nFinding clip planes...")
clip_bot = None
clip_top = None

# Scan bottom → upward
for y_frac in np.linspace(0.0, 0.25, 200):
    y = y_min + (y_max - y_min) * y_frac
    sl = case.slice(normal='y', origin=(0, y, 0))
    if sl.n_points == 0:
        continue
    xs = sl.points[:, 0]
    if (xs.max() - xs.min()) >= full_span - tolerance:
        clip_bot = y
        break

# Scan top → downward
for y_frac in np.linspace(1.0, 0.75, 200):
    y = y_min + (y_max - y_min) * y_frac
    sl = case.slice(normal='y', origin=(0, y, 0))
    if sl.n_points == 0:
        continue
    xs = sl.points[:, 0]
    if (xs.max() - xs.min()) >= full_span - tolerance:
        clip_top = y
        break

print(f"  Clip bottom plane: Y = {clip_bot:.2f} mm")
print(f"  Clip top    plane: Y = {clip_top:.2f} mm")
print(f"  New height will be: {clip_top - clip_bot:.2f} mm  (original: {y_max - y_min:.2f} mm)")

# ------------------------------------------------------------------
# 3. Clip the mesh at both planes (closed clip → keeps solid)
#    PyVista convention: normal='+y' KEEPS everything above the plane,
#                        normal='-y' KEEPS everything below the plane
# ------------------------------------------------------------------
print("\nClipping...")

# Remove bottom rounded cap → keep everything ABOVE clip_bot (normal=+y)
clipped = case.clip_closed_surface(normal='y',
                                   origin=(0, clip_bot, 0))

# Remove top rounded cap → keep everything BELOW clip_top (normal=-y)
clipped = clipped.clip_closed_surface(normal='-y',
                                      origin=(0, clip_top, 0))

print(f"  Clipped mesh: {clipped.n_cells} cells, {clipped.n_points} points")

new_bounds = clipped.bounds
print(f"  New Y range: {new_bounds[2]:.2f} → {new_bounds[3]:.2f}  (height {new_bounds[3]-new_bounds[2]:.2f} mm)")
print(f"  New X range: {new_bounds[0]:.2f} → {new_bounds[1]:.2f}  (width  {new_bounds[1]-new_bounds[0]:.2f} mm)")
print(f"  New Z range: {new_bounds[4]:.2f} → {new_bounds[5]:.2f}  (depth  {new_bounds[5]-new_bounds[4]:.2f} mm)")

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
print(f"\nSaving to {OUTPUT}...")
clipped.save(OUTPUT)
print("Done.")
print()
print("Summary:")
print(f"  Original: {y_max-y_min:.2f} mm tall | rounded ends (~10mm taper each side)")
print(f"  Fixed:    {new_bounds[3]-new_bounds[2]:.2f} mm tall | sharp 90° ends")
print(f"  Removed:  {(y_max-y_min) - (new_bounds[3]-new_bounds[2]):.2f} mm total")
