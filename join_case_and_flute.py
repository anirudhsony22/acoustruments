import pyvista as pv
import numpy as np
import os

# Constants from create_flute.py
ADAPTER_OD = 10.0
FIT_CLEARANCE = 0.6
SOCKET_ID = ADAPTER_OD + FIT_CLEARANCE
SOCKET_OD = SOCKET_ID + 4.0
SOCKET_DEPTH = 8.0
ADAPTER_SEPARATION = 33.0

def _quarter_torus(center, major_r, minor_r_out, minor_r_in, start_angle, end_angle, res=40, n_arc=20):
    theta = np.linspace(0, 2 * np.pi, res, endpoint=False)
    phi   = np.linspace(start_angle, end_angle, n_arc)
    cx, cy, cz = center
    def make_ring_verts(minor_r):
        verts = []
        for p in phi:
            cx_ring = cx + major_r * np.cos(p)
            cy_ring = cy + major_r * np.sin(p)
            rx, ry = np.cos(p), np.sin(p)
            for t in theta:
                verts.append([cx_ring + minor_r * np.cos(t) * rx,
                              cy_ring + minor_r * np.cos(t) * ry,
                              cz     + minor_r * np.sin(t)])
        return np.array(verts)
    outer_v = make_ring_verts(minor_r_out)
    inner_v = make_ring_verts(minor_r_in)
    all_v   = np.vstack([outer_v, inner_v])
    N, M = res, n_arc
    oi = lambda i, j: i * N + j % N
    ii = lambda i, j: M * N + i * N + j % N
    faces = []
    for i in range(M - 1):
        for j in range(N):
            a, b, c, d = oi(i,j), oi(i,j+1), oi(i+1,j), oi(i+1,j+1)
            faces += [[3, a, b, d], [3, a, d, c]]
            a, b, c, d = ii(i,j), ii(i,j+1), ii(i+1,j), ii(i+1,j+1)
            faces += [[3, a, d, b], [3, a, c, d]]
    for i_ring in [0, M - 1]:
        for j in range(N):
            vo0, vo1 = oi(i_ring, j), oi(i_ring, j+1)
            vi0, vi1 = ii(i_ring, j), ii(i_ring, j+1)
            if i_ring == 0: faces += [[3, vo0, vi0, vo1], [3, vo1, vi0, vi1]]
            else: faces += [[3, vo0, vo1, vi0], [3, vo1, vi1, vi0]]
    return pv.PolyData(all_v, np.hstack(faces))


def _make_tapered_arm(x, arm_h, r_out_pipe, r_in_pipe, res=40):
    """Transition from elliptical base (15x8mm) to circular pipe."""
    # Base dimensions matching standard phone ports
    a_in_base, b_in_base = 7.5, 4.0
    a_out_base, b_out_base = 9.5, 6.0
    
    z_steps = np.linspace(0, arm_h, 15)
    theta = np.linspace(0, 2*np.pi, res, endpoint=False)
    
    def get_verts(a_base, b_base, r_pipe):
        verts = []
        for y_rel in z_steps:
            t = y_rel / arm_h
            # Transition from ellipse to circle
            a = (1-t)*a_base + t*r_pipe
            b = (1-t)*b_base + t*r_pipe
            for th in theta:
                # Arm is along Y axis. Base at Y=0, extending to Y=-arm_h
                verts.append([x + a*np.cos(th), -y_rel, b*np.sin(th)])
        return np.array(verts)

    outer_v = get_verts(a_out_base, b_out_base, r_out_pipe)
    inner_v = get_verts(a_in_base, b_in_base, r_in_pipe)
    all_v = np.vstack([outer_v, inner_v])
    
    N, M = res, len(z_steps)
    oi = lambda i, j: i * N + j % N
    ii = lambda i, j: M * N + i * N + j % N
    faces = []
    for i in range(M - 1):
        for j in range(N):
            # Outer
            v0, v1, v2, v3 = oi(i,j), oi(i,j+1), oi(i+1,j), oi(i+1,j+1)
            faces += [[3, v0, v1, v3], [3, v0, v3, v2]]
            # Inner (flipped)
            v0, v1, v2, v3 = ii(i,j), ii(i,j+1), ii(i+1,j), ii(i+1,j+1)
            faces += [[3, v0, v3, v1], [3, v0, v2, v3]]
    
    # Top/Bottom Rings
    for th in range(N):
        # Base ring (Y=0)
        vo0, vo1 = oi(0, th), oi(0, th+1)
        vi0, vi1 = ii(0, th), ii(0, th+1)
        faces += [[3, vo0, vi0, vo1], [3, vo1, vi0, vi1]]
        # End ring (Y=-arm_h)
        vo0, vo1 = oi(M-1, th), oi(M-1, th+1)
        vi0, vi1 = ii(M-1, th), ii(M-1, th+1)
        faces += [[3, vo0, vo1, vi0], [3, vo1, vi1, vi0]]
        
    return pv.PolyData(all_v, np.hstack(faces))

def _drill_holes(mesh, dia, r_out, r_in, centers, direction, res=24):
    r, wall = dia / 2, r_out - r_in
    for center in centers:
        cutter = pv.Cylinder(center=center, direction=direction, radius=r, height=wall*2+2, resolution=res).triangulate()
        mesh = mesh.boolean_difference(cutter).triangulate()
    return mesh

def build_elbow_mesh(r_out, r_in, wall, n_holes, hole_dia, res=40):
    SEP, arm_h = ADAPTER_SEPARATION, 10.0
    bend_r = max(r_out + 2.0, 7.0)
    bottom_length = SEP - 2 * bend_r
    if bottom_length < (n_holes * hole_dia + 2.0) and n_holes > 0:
        bend_r = max(r_out + 1.0, 6.0)
        bottom_length = SEP - 2 * bend_r
    bottom_y = -(arm_h + bend_r)
    parts = []
    # Arms (Cylindrical to fit existing adapters)
    for x in [0, SEP]:
        o = pv.Cylinder(center=(x, -arm_h/2, 0), direction=(0,1,0), radius=r_out, height=arm_h, resolution=res).triangulate()
        i = pv.Cylinder(center=(x, -arm_h/2, 0), direction=(0,1,0), radius=r_in, height=arm_h+2, resolution=res).triangulate()
        parts.append(o.boolean_difference(i).triangulate())
    # Bends
    parts.append(_quarter_torus(center=(bend_r, -arm_h, 0), major_r=bend_r, minor_r_out=r_out, minor_r_in=r_in, start_angle=np.pi, end_angle=1.5*np.pi, res=res))
    parts.append(_quarter_torus(center=(SEP-bend_r, -arm_h, 0), major_r=bend_r, minor_r_out=r_out, minor_r_in=r_in, start_angle=1.5*np.pi, end_angle=2.0*np.pi, res=res))
    # Bottom
    bot_cx = bend_r + bottom_length / 2
    o = pv.Cylinder(center=(bot_cx, bottom_y, 0), direction=(1,0,0), radius=r_out, height=bottom_length, resolution=res).triangulate()
    i = pv.Cylinder(center=(bot_cx, bottom_y, 0), direction=(1,0,0), radius=r_in, height=bottom_length+2, resolution=res).triangulate()
    bottom = o.boolean_difference(i).triangulate()
    if n_holes > 0:
        end_padding = max(2.0, hole_dia / 2 + 1.0)
        inner_len = max(0.1, bottom_length - 2 * end_padding)
        gap = inner_len / (n_holes - 1) if n_holes > 1 else 0
        hx_start = bend_r + end_padding if n_holes > 1 else bend_r + bottom_length / 2
        centers = [[hx_start + k * gap, bottom_y - (r_in + r_out) / 2, 0] for k in range(n_holes)]
        bottom = _drill_holes(bottom, hole_dia, r_out, r_in, centers, direction=(0,1,0), res=res)
    parts.append(bottom)
    
    result = parts[0]
    for p in parts[1:]: result = result.merge([p])
    return result

def join_models(pod, wall, nh, hd, half_case=True, case_file='case_with_adapters_sharp.stl', output_file='case_with_flute.stl'):
    print(f"Loading {case_file}...")
    case = pv.read(case_file)
    
    if half_case:
        print("Clipping case to bottom half (exact geometry)...")
        # Pure clip preserves the original triangle structure for the portion kept
        case = case.clip(normal=(0, 1, 0), origin=(0, 5.0, 0), invert=True).triangulate()
    
    # 1. Identify Case Dimensions
    b = case.bounds
    cx, cz, ymin = (b[0]+b[1])/2, (b[4]+b[5])/2, b[2]

    # 2. Generate Parametric Flute
    print(f"Generating parametric elbow flute (OD={pod}mm, holes={nh})...")
    # Using cylindrical arms for perfect fit with existing adapters
    flute = build_elbow_mesh(r_out=pod/2, r_in=(pod-2*wall)/2, wall=wall, n_holes=nh, hole_dia=hd)
    
    # 3. Align Flute to Adapters
    # Flute arms start at Y=0. We'll embed them 1mm into the adapter tips for a clean joint.
    tx = cx - 15.0
    ty = ymin + 1.0
    tz = cz
    flute = flute.translate([tx, ty, tz])
    
    # 4. Seamless Integration
    print("Fusing pipe and case (minimal processing)...")
    # Using simple merge to preserve the original geometry bit-for-bit
    combined = case.merge([flute])
    combined.compute_normals(inplace=True, auto_orient_normals=True)
    
    print(f"Saving to {output_file}...")
    combined.save(output_file)
    print("Done!")

if __name__ == "__main__":
    print("=== Unified Acoustic Case Generator ===")
    print("This script will join a custom pipe to the sharp phone case.\n")
    
    try:
        v = input("Pipe outer diameter mm [10.0]: ").strip()
        pod = float(v) if v else 10.0
        v = input("Wall thickness mm [2.0]: ").strip()
        w = float(v) if v else 2.0
        v = input("Number of acoustic holes [3]: ").strip()
        nh = int(v) if v else 3
        v = input("Hole diameter mm [2.5]: ").strip()
        hd = float(v) if v else 2.5
        v = input("Output filename [case_with_flute.stl]: ").strip()
        fn = v if v else "case_with_flute.stl"
        if not fn.endswith(".stl"): fn += ".stl"
    except ValueError:
        print("Bad input."); exit(1)

    join_models(pod, w, nh, hd, half_case=True, output_file=fn)
