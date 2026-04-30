import pyvista as pv
import numpy as np
import os

# ============================================================
# Adapter interface (from create_adapter.py)
#   Adapter top pipe: OD = 10mm (r=5), ID = 6mm (r=3), wall = 2mm
#   Pipe section is 10mm long (Z=5 to Z=15 in adapter space)
# ============================================================
ADAPTER_OD = 10.0          # adapter top outer diameter
FIT_CLEARANCE = 0.6        # total diametral clearance for slip fit
SOCKET_ID = ADAPTER_OD + FIT_CLEARANCE   # 10.6mm – socket slides over adapter
SOCKET_OD = SOCKET_ID + 4.0             # 14.6mm – 2mm wall each side
SOCKET_DEPTH = 8.0                       # grip depth (mm)

# Mic-to-speaker adapter separation on case bottom (X axis)
ADAPTER_SEPARATION = 33.0  # mm


def create_flute_stl(pipe_od=10.0,
                     wall=2.0,
                     num_holes=3,
                     hole_dia=2.0,
                     pipe_length=40.0,
                     style="elbow",
                     cavity_dia=0.0,
                     output="flute.stl"):
    """
    Generate a modular flute pipe or Helmholtz resonator.

    'straight' = one socket + straight neck + optional bulb
    'elbow'    = U-pipe connecting adapters + optional central bulb
    """
    pipe_id = pipe_od - 2 * wall
    if pipe_id <= 0:
        print("Error: wall too thick."); return

    print(f"\nPipe OD={pipe_od}mm  ID={pipe_id}mm  style={style}")
    print(f"Socket ID={SOCKET_ID}mm (fits adapter OD {ADAPTER_OD}mm + {FIT_CLEARANCE}mm clearance)")
    print(f"Socket OD={SOCKET_OD}mm  depth={SOCKET_DEPTH}mm\n")

    r_out = pipe_od / 2
    r_in  = pipe_id / 2
    RES   = 40  # circumferential resolution (lower = faster booleans)

    if style == "elbow":
        _build_elbow(r_out, r_in, wall, num_holes, hole_dia, cavity_dia, RES, output)
    else:
        _build_straight(r_out, r_in, wall, num_holes, hole_dia, pipe_length, cavity_dia, RES, output)


def _build_straight(r_out, r_in, wall, n_holes, hole_dia, length, cavity_dia, res, output):
    """Simple straight pipe/neck with one socket and optional end bulb."""
    print(f"Building straight {'Helmholtz' if cavity_dia > 0 else 'pipe'}...")

    # Main tube
    outer = pv.Cylinder(center=(0,0,length/2), direction=(0,0,1),
                        radius=r_out, height=length, resolution=res).triangulate()
    inner = pv.Cylinder(center=(0,0,length/2), direction=(0,0,1),
                        radius=r_in, height=length+2, resolution=res).triangulate()
    pipe = outer.boolean_difference(inner).triangulate()

    # Acoustic holes
    if n_holes > 0:
        pipe = _drill_holes(pipe, n_holes, hole_dia, r_out, r_in,
                            positions=[(0, length/(n_holes+1)*(i+1)) for i in range(n_holes)],
                            direction=(1,0,0), axis='x', res=res)

    # Optional Helmholtz bulb at the end
    if cavity_dia > r_out * 2:
        print(f"Adding cavity bulb (Ø{cavity_dia}mm)...")
        b_outer = pv.Sphere(radius=cavity_dia/2, center=(0,0,length), phi_resolution=res, theta_resolution=res).triangulate()
        b_inner = pv.Sphere(radius=cavity_dia/2 - wall, center=(0,0,length), phi_resolution=res, theta_resolution=res).triangulate()
        bulb = b_outer.boolean_difference(b_inner).triangulate()
        pipe = pipe.merge([bulb])

    # Socket at Z=0
    sock = _make_socket(base_center=(0,0,0), direction=(0,0,-1), r_pipe_out=r_out, r_pipe_in=r_in, res=res)
    result = pipe.merge([sock])

    result.save(output)
    print(f"Saved → {os.path.abspath(output)}")


def _build_elbow(r_out, r_in, wall, n_holes, hole_dia, cavity_dia, res, output):
    """
    U-shaped pipe built from simple primitives:
      - Two vertical arms (cylinders)
      - One horizontal bottom (cylinder)
      - Two 90° bends (torus quarter-sections)
      - Two sockets at the top of each arm
      - Holes drilled in the horizontal section

    Layout in XY plane (Z is depth):
      Socket 1 at X=0            Socket 2 at X=SEP
         |                           |
         | Arm 1 (vertical)          | Arm 2 (vertical)
         |                           |
       Bend 1 ---- Bottom ---- Bend 2

    Arms go in -Y direction. Sockets face +Y (adapters protrude -Y from case).
    """
    SEP   = ADAPTER_SEPARATION      # 33mm
    arm_h = 10.0                    # arm height (vertical section)
    bend_r = max(r_out * 2, 10.0)   # centreline bend radius
    bottom_length = SEP - 2 * bend_r

    if bottom_length < 0:
        # If adapters too close for the bend radius, reduce bend_r
        bend_r = SEP / 2 - 0.5
        bottom_length = SEP - 2 * bend_r
        if bottom_length < 0:
            bottom_length = 0
            bend_r = SEP / 2

    bottom_y = -(arm_h + bend_r)    # Y coordinate of the horizontal section
    print(f"Arm height: {arm_h}mm, bend radius: {bend_r}mm, bottom length: {bottom_length:.1f}mm")
    print("Building segments...")

    parts = []

    # ---------- ARM 1: X=0, vertical from Y=0 to Y=-arm_h ----------
    arm1_outer = pv.Cylinder(center=(0, -arm_h/2, 0), direction=(0,1,0),
                             radius=r_out, height=arm_h, resolution=res).triangulate()
    arm1_inner = pv.Cylinder(center=(0, -arm_h/2, 0), direction=(0,1,0),
                             radius=r_in, height=arm_h+2, resolution=res).triangulate()
    arm1 = arm1_outer.boolean_difference(arm1_inner).triangulate()
    parts.append(arm1)

    # ---------- ARM 2: X=SEP, vertical from Y=0 to Y=-arm_h ----------
    arm2_outer = pv.Cylinder(center=(SEP, -arm_h/2, 0), direction=(0,1,0),
                             radius=r_out, height=arm_h, resolution=res).triangulate()
    arm2_inner = pv.Cylinder(center=(SEP, -arm_h/2, 0), direction=(0,1,0),
                             radius=r_in, height=arm_h+2, resolution=res).triangulate()
    arm2 = arm2_outer.boolean_difference(arm2_inner).triangulate()
    parts.append(arm2)

    # ---------- BEND 1: quarter-torus at (bend_r, -arm_h, 0) ----------
    # Connects arm1 bottom to horizontal section
    print("Building bends (quarter torus)...")
    bend1 = _quarter_torus(center=(bend_r, -arm_h, 0), major_r=bend_r,
                           minor_r_out=r_out, minor_r_in=r_in,
                           start_angle=np.pi, end_angle=1.5*np.pi, res=res)
    parts.append(bend1)

    # ---------- BEND 2: quarter-torus at (SEP-bend_r, -arm_h, 0) ----------
    bend2 = _quarter_torus(center=(SEP-bend_r, -arm_h, 0), major_r=bend_r,
                           minor_r_out=r_out, minor_r_in=r_in,
                           start_angle=1.5*np.pi, end_angle=2.0*np.pi, res=res)
    parts.append(bend2)

    # ---------- BOTTOM: horizontal from X=bend_r to X=SEP-bend_r at Y=bottom_y ----------
    if bottom_length > 0.1:
        bot_cx = bend_r + bottom_length / 2
        bot_outer = pv.Cylinder(center=(bot_cx, bottom_y, 0), direction=(1,0,0),
                                radius=r_out, height=bottom_length, resolution=res).triangulate()
        bot_inner = pv.Cylinder(center=(bot_cx, bottom_y, 0), direction=(1,0,0),
                                radius=r_in, height=bottom_length+2, resolution=res).triangulate()
        bottom = bot_outer.boolean_difference(bot_inner).triangulate()

        # Optional Helmholtz bulb in center of horizontal section
        if cavity_dia > r_out * 2:
            print(f"Adding central cavity bulb (Ø{cavity_dia}mm)...")
            b_outer = pv.Sphere(radius=cavity_dia/2, center=(bot_cx, bottom_y, 0), phi_resolution=res, theta_resolution=res).triangulate()
            b_inner = pv.Sphere(radius=cavity_dia/2 - wall, center=(bot_cx, bottom_y, 0), phi_resolution=res, theta_resolution=res).triangulate()
            bulb = b_outer.boolean_difference(b_inner).triangulate()
            bottom = bottom.merge([bulb])

        parts.append(bottom)

    # ---------- SOCKETS at top of each arm ----------
    print("Adding sockets...")
    sock1 = _make_socket(base_center=(0, 0, 0), direction=(0,1,0), r_pipe_out=r_out, r_pipe_in=r_in, res=res)
    sock2 = _make_socket(base_center=(SEP, 0, 0), direction=(0,1,0), r_pipe_out=r_out, r_pipe_in=r_in, res=res)
    parts.append(sock1)
    parts.append(sock2)

    # ---------- MERGE & SAVE ----------
    print("Merging all segments...")
    result = parts[0]
    for p in parts[1:]:
        result = result.merge([p])

    result.save(output)
    print(f"\nDone! → {os.path.abspath(output)}")
    print(f"  Pipe bore Ø{r_in*2:.1f}mm  |  OD Ø{r_out*2:.1f}mm")
    print(f"  Socket ID Ø{SOCKET_ID:.1f}mm → fits over adapter OD {ADAPTER_OD}mm")
    print(f"  Socket OD Ø{SOCKET_OD:.1f}mm  depth {SOCKET_DEPTH}mm")
    print(f"  U-span: {SEP}mm (arm-to-arm)")


def _quarter_torus(center, major_r, minor_r_out, minor_r_in,
                   start_angle, end_angle, res=40, n_arc=20):
    """
    Build a quarter-section of a hollow torus (no booleans, pure vertices).
    The torus lies in the XY plane, centred at `center`.
    major_r    = distance from torus centre to tube centreline
    minor_r_*  = outer/inner cross-section radii
    start/end  = arc angles in the XY plane (radians)
    """
    theta = np.linspace(0, 2 * np.pi, res, endpoint=False)  # around tube
    phi   = np.linspace(start_angle, end_angle, n_arc)       # along arc

    cx, cy, cz = center

    def make_ring_verts(minor_r):
        verts = []
        for p in phi:
            # Centre of cross-section on the torus centreline
            cx_ring = cx + major_r * np.cos(p)
            cy_ring = cy + major_r * np.sin(p)
            # Radial direction (outward from torus centre in XY plane)
            rx, ry = np.cos(p), np.sin(p)
            for t in theta:
                # Point on the ring: radial + axial (Z) components
                verts.append([
                    cx_ring + minor_r * np.cos(t) * rx,
                    cy_ring + minor_r * np.cos(t) * ry,
                    cz     + minor_r * np.sin(t),
                ])
        return np.array(verts)

    outer_v = make_ring_verts(minor_r_out)
    inner_v = make_ring_verts(minor_r_in)
    all_v   = np.vstack([outer_v, inner_v])

    N = res
    M = n_arc
    oi = lambda i, j: i * N + j % N
    ii = lambda i, j: M * N + i * N + j % N

    faces = []
    # Outer surface
    for i in range(M - 1):
        for j in range(N):
            a, b, c, d = oi(i,j), oi(i,j+1), oi(i+1,j), oi(i+1,j+1)
            faces += [[3, a, b, d], [3, a, d, c]]
    # Inner surface (reversed winding)
    for i in range(M - 1):
        for j in range(N):
            a, b, c, d = ii(i,j), ii(i,j+1), ii(i+1,j), ii(i+1,j+1)
            faces += [[3, a, d, b], [3, a, c, d]]
    # End caps (annular rings)
    for i_ring in [0, M - 1]:
        for j in range(N):
            vo0, vo1 = oi(i_ring, j), oi(i_ring, j+1)
            vi0, vi1 = ii(i_ring, j), ii(i_ring, j+1)
            if i_ring == 0:
                faces += [[3, vo0, vi0, vo1], [3, vo1, vi0, vi1]]
            else:
                faces += [[3, vo0, vo1, vi0], [3, vo1, vi1, vi0]]

    return pv.PolyData(all_v, np.hstack(faces))


def _make_socket(base_center, direction, r_pipe_out, r_pipe_in, res=40):
    """
    Hollow cylinder socket with a tapered transition (shoulder) for stability.
    - base_center: connection point to the pipe
    - direction: points from base toward the opening
    - r_pipe_out/in: dimensions of the pipe being attached to
    """
    c = np.array(base_center, dtype=float)
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    
    # 1. Main Socket Tube
    # Length of the straight socket section
    cyl_h = SOCKET_DEPTH
    cyl_c = c + d * (cyl_h / 2 + 5.0) # Offset by taper height

    outer = pv.Cylinder(center=cyl_c, direction=d,
                        radius=SOCKET_OD/2, height=cyl_h,
                        resolution=res).triangulate()
    inner = pv.Cylinder(center=cyl_c, direction=d,
                        radius=SOCKET_ID/2, height=cyl_h+2,
                        resolution=res).triangulate()
    socket_body = outer.boolean_difference(inner).triangulate()
    
    # Add a small lead-in chamfer at the opening (mouth)
    mouth_c = cyl_c + d * (cyl_h / 2)
    chamfer = pv.Cylinder(center=mouth_c, direction=d,
                          radius=SOCKET_ID/2 + 1.0, height=1.0,
                          resolution=res).triangulate()
    # Subtracting a cone/chamfer would be better but a slightly larger cylinder at the tip 
    # to soften the edge is a start. Let's just stick to the taper for now as stability is priority.

    # 2. Tapered Transition (Shoulder)
    # Connects pipe_od to socket_od over 5mm distance
    taper_h = 5.0
    taper_c = c + d * (taper_h / 2)
    
    # Create a frustum using vertex manipulation on a cylinder
    frustum_outer = pv.Cylinder(center=(0,0,0), direction=(0,0,1), 
                                radius=1.0, height=taper_h, resolution=res).triangulate()
    pts = frustum_outer.points.copy()
    # Z range is [-taper_h/2, taper_h/2]
    # base (towards pipe) is at Z = -taper_h/2
    # top (towards socket) is at Z = taper_h/2
    for i in range(len(pts)):
        t = (pts[i, 2] + taper_h/2) / taper_h
        r = r_pipe_out * (1 - t) + (SOCKET_OD/2) * t
        pts[i, 0] *= r
        pts[i, 1] *= r
    frustum_outer.points = pts
    
    # Hollow it out with the bore
    frustum_inner = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                                radius=r_pipe_in, height=taper_h+2,
                                resolution=res).triangulate()
    taper = frustum_outer.boolean_difference(frustum_inner).triangulate()
    
    # Align taper with the socket direction
    # Rotation from (0,0,1) to d
    target_d = d
    source_d = [0,0,1]
    if not np.allclose(source_d, target_d):
        axis = np.cross(source_d, target_d)
        if np.linalg.norm(axis) < 1e-6: # Antiparallel
            axis = [1,0,0]
            angle = 180
        else:
            angle = np.degrees(np.arccos(np.dot(source_d, target_d)))
        taper = taper.rotate_vector(axis, angle)
    
    taper = taper.translate(taper_c)

    return socket_body.merge([taper]).triangulate()


def _drill_holes(mesh, n, dia, r_out, r_in, positions, direction, axis, res=24):
    """Boolean-subtract cylindrical holes from a mesh."""
    r = dia / 2
    wall = r_out - r_in
    for _, coord in positions:
        center = [0, 0, 0]
        center[{'x':0,'y':1,'z':2}[axis]] = (r_in + r_out) / 2
        center[2] = coord  # Z position along pipe
        cutter = pv.Cylinder(center=center, direction=direction,
                             radius=r, height=wall*2+1, resolution=res).triangulate()
        mesh = mesh.boolean_difference(cutter).triangulate()
    return mesh


# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Flute Pipe Generator (Adapter-Compatible) ===")
    print(f"  Adapter top: OD={ADAPTER_OD}mm → Socket ID={SOCKET_ID}mm\n")

    try:
        v = input("Pipe outer diameter mm [10.0]: ").strip()
        pod = float(v) if v else 10.0

        v = input("Wall thickness mm [2.0]: ").strip()
        w = float(v) if v else 2.0

        v = input("Number of acoustic holes [3]: ").strip()
        nh = int(v) if v else 3

        v = input("Hole diameter mm [2.5]: ").strip()
        hd = float(v) if v else 2.5

        v = input("Pipe length mm [40] (straight only): ").strip()
        pl = float(v) if v else 40.0

        v = input("Style: [1] Straight  [2] U-Elbow [default 2]: ").strip()
        st = "straight" if v == "1" else "elbow"

        v = input("Helmholtz cavity diameter mm [0 for none]: ").strip()
        cd = float(v) if v else 0.0

        v = input("Output filename [flute.stl]: ").strip()
        fn = v if v else "flute.stl"
        if not fn.endswith(".stl"):
            fn += ".stl"

    except ValueError:
        print("Bad input."); exit(1)

    try:
        create_flute_stl(pod, w, nh, hd, pl, st, cd, fn)
    except Exception as e:
        import traceback; traceback.print_exc()
