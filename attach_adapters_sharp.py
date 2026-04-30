"""
attach_adapters_sharp.py
------------------------
Attaches two acoustic adapters (mic + speaker) to the new sharp-cornered
phone case (day-10-phone-case-v7-sharp.stl).

iPhone 13 Pro Max bottom-edge layout (looking at the phone face-up):
  - The bottom edge is at the Y_min face of the case
  - Lightning port is roughly centered (center_x ≈ 2.59 mm in STL coords)
  - Microphone slot:   ~15 mm to the LEFT  of center_x  → X ≈ -12.4 mm
  - Speaker grille:    ~18 mm to the RIGHT of center_x  → X ≈  20.6 mm

Adapter orientation:
  - adapter.stl is built with its base at Z=0, opening pointing +Z
  - Rotate 90° around X-axis → base now at Y=0, opening pointing -Y (outward)
  - Translate base to flush with case Y_min face (embed 1mm for watertight join)

Output: case_with_adapters_sharp.stl
"""

import pyvista as pv
import numpy as np

INPUT_CASE    = 'day-10-phone-case-v7-sharp.stl'
INPUT_ADAPTER = 'adapter.stl'
OUTPUT        = 'case_with_adapters_sharp.stl'

# ------------------------------------------------------------------
# 1. Load meshes
# ------------------------------------------------------------------
print("Loading meshes...")
case    = pv.read(INPUT_CASE)
adapter = pv.read(INPUT_ADAPTER)

cb = case.bounds
ab = adapter.bounds

x_min, x_max = cb[0], cb[1]
y_min, y_max = cb[2], cb[3]
z_min, z_max = cb[4], cb[5]

center_x = (x_min + x_max) / 2.0   #  2.59 mm
center_z = (z_min + z_max) / 2.0   #  3.45 mm  (mid-depth of case)

print(f"  Case   bounds: X[{x_min:.1f},{x_max:.1f}] Y[{y_min:.1f},{y_max:.1f}] Z[{z_min:.1f},{z_max:.1f}]")
print(f"  Adapter bounds: X[{ab[0]:.1f},{ab[1]:.1f}] Y[{ab[2]:.1f},{ab[3]:.1f}] Z[{ab[4]:.1f},{ab[5]:.1f}]")
print(f"  Case center_x={center_x:.2f}  center_z={center_z:.2f}  y_min={y_min:.2f}")

# ------------------------------------------------------------------
# 2. iPhone 13 Pro Max mic/speaker offsets from case center
#    (measured from teardown reference; Lightning port ≈ center)
#
#    Mic:     left side of Lightning port, ~15 mm from center
#    Speaker: right side (grille side),   ~18 mm from center
# ------------------------------------------------------------------
MIC_OFFSET_X     = -15.0   # mm left of center_x
SPEAKER_OFFSET_X =  18.0   # mm right of center_x

EMBED_Y = 1.0  # mm to embed adapter base into case wall (watertight union)

mic_x     = center_x + MIC_OFFSET_X      # -12.41
speaker_x = center_x + SPEAKER_OFFSET_X  #  20.59

print(f"\n  Mic X = {mic_x:.2f} mm  (center_x {MIC_OFFSET_X:+.1f} mm)")
print(f"  Speaker X = {speaker_x:.2f} mm  (center_x {SPEAKER_OFFSET_X:+.1f} mm)")

# ------------------------------------------------------------------
# 3. Build the rotation matrix: adapter Z-axis → -Y-axis
#    Rotate +90° around world X-axis:  Z→-Y,  Y→+Z
# ------------------------------------------------------------------
angle = np.radians(90)
Rx = np.array([
    [1,           0,            0, 0],
    [0,  np.cos(angle), -np.sin(angle), 0],
    [0,  np.sin(angle),  np.cos(angle), 0],
    [0,           0,            0, 1],
], dtype=float)

# ------------------------------------------------------------------
# 4. Helper: build one adapter at the given (x, z) position
# ------------------------------------------------------------------
def make_adapter(target_x, target_z, label):
    """
    Returns a PyVista PolyData for one adapter, placed so that:
      - Its base (originally at Z=0) sits at Y = y_min + EMBED_Y
      - Its opening points outward in -Y
      - It is centered at (target_x, _, target_z)
    """
    pts = adapter.points.copy()          # shape (N,3): x,y,z in adapter space

    # Step A – Rotate: Z→-Y, Y→+Z  (apply 3×3 rotation)
    R3 = Rx[:3, :3]
    pts = pts @ R3.T                     # (N,3) in rotated space
    # After rotation: original Z is now -Y, original Y is now +Z
    # adapter base (was Z=0) is now at Y=0
    # adapter opening (was Z=+15) is now at Y=-15 (points away from case)

    # Step B – Translate to target position
    # We want the base (Y=0) to be at y_min + EMBED_Y
    pts[:, 0] += target_x               # shift in X
    pts[:, 1] += y_min + EMBED_Y        # base at case bottom wall
    pts[:, 2] += target_z               # shift in Z

    result = pv.PolyData(pts, adapter.faces)
    print(f"  {label}: base_Y={y_min+EMBED_Y:.2f}, tip_Y={pts[:,1].min():.2f}, "
          f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}], "
          f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
    return result

print("\nPositioning adapters...")
mic_adapter     = make_adapter(mic_x,     center_z, "Mic    adapter")
speaker_adapter = make_adapter(speaker_x, center_z, "Speaker adapter")

# ------------------------------------------------------------------
# 5. Combine and save
#    Using PyVista merge (concatenate geometry) - works fine for most
#    slicers which handle multi-body STLs
# ------------------------------------------------------------------
print("\nMerging meshes...")
combined = case.merge([mic_adapter, speaker_adapter])

print(f"  Combined: {combined.n_cells} cells, {combined.n_points} points")

print(f"\nSaving to {OUTPUT}...")
combined.save(OUTPUT)
print("Done!")

print(f"""
=== SUMMARY ===
  Input case:    {INPUT_CASE}
  Output:        {OUTPUT}

  Mic adapter
    X center: {mic_x:.2f} mm  (case_center_x {MIC_OFFSET_X:+.0f} mm)
    Protrudes: {ab[5]:.1f} mm outward from case bottom face

  Speaker adapter
    X center: {speaker_x:.2f} mm  (case_center_x {SPEAKER_OFFSET_X:+.0f} mm)
    Protrudes: {ab[5]:.1f} mm outward from case bottom face

  Both adapters centered at Z = {center_z:.2f} mm (mid-depth of case)
""")
