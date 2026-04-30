import numpy as np
import pyvista as pv
import os

def create_adapter(output_filename="adapter.stl"):
    # Dimensions configuration
    N = 120 # number of radial points (resolution)
    Z_transition = 5.0  # gradient height
    Z_pipe = 10.0       # top pipe extension
    Z_total = Z_transition + Z_pipe
    
    # Base (Elliptical) inner dimensions (15x8mm) -> a=7.5, b=4.0
    a_in_base = 15.0 / 2.0
    b_in_base = 8.0 / 2.0
    # Base outer dimensions (19x12mm) -> a=9.5, b=6.0 (2mm wall thickness)
    a_out_base = 19.0 / 2.0
    b_out_base = 12.0 / 2.0
    
    # Top (Circular) inner dimensions (dia 6mm) -> r=3.0
    r_in_top = 3.0
    # Top outer dimensions (dia 10mm) -> r=5.0 (2mm wall thickness)
    r_out_top = 5.0
    
    # Z-slices
    # Use exponential or linear spacing in transition zone?
    # Linear is simplest and prints well
    Z_steps = np.concatenate([
        np.linspace(0, Z_transition, 25), 
        np.linspace(Z_transition, Z_total, 5)[1:]
    ])
    
    theta_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    vertices = []
    
    # 1. Generate Outer Skin Vertices
    outer_start_idx = len(vertices)
    for z in Z_steps:
        if z <= Z_transition:
            t = z / Z_transition
            a = (1 - t) * a_out_base + t * r_out_top
            b = (1 - t) * b_out_base + t * r_out_top
        else:
            a = r_out_top
            b = r_out_top
            
        for th in theta_vals:
            vertices.append([a * np.cos(th), b * np.sin(th), z])
            
    # 2. Generate Inner Skin Vertices
    inner_start_idx = len(vertices)
    for z in Z_steps:
        if z <= Z_transition:
            t = z / Z_transition
            a = (1 - t) * a_in_base + t * r_in_top
            b = (1 - t) * b_in_base + t * r_in_top
        else:
            a = r_in_top
            b = r_in_top
            
        for th in theta_vals:
            vertices.append([a * np.cos(th), b * np.sin(th), z])
            
    vertices = np.array(vertices)
    faces = []
    num_z = len(Z_steps)
    
    # Helpers to find vertex indices
    def get_outer_idx(z_idx, th_idx):
        return outer_start_idx + z_idx * N + (th_idx % N)
        
    def get_inner_idx(z_idx, th_idx):
        return inner_start_idx + z_idx * N + (th_idx % N)
        
    # -- Join Outer Surface --
    # v0 = (z, th), v1 = (z, th+1), v2 = (z+1, th), v3 = (z+1, th+1)
    for z_idx in range(num_z - 1):
        for th_idx in range(N):
            v0 = get_outer_idx(z_idx, th_idx)
            v1 = get_outer_idx(z_idx, th_idx + 1)
            v2 = get_outer_idx(z_idx + 1, th_idx)
            v3 = get_outer_idx(z_idx + 1, th_idx + 1)
            # Triangles ensure proper outward normal (CCW)
            faces.extend([[3, v0, v1, v2], [3, v1, v3, v2]])
            
    # -- Join Inner Surface --
    for z_idx in range(num_z - 1):
        for th_idx in range(N):
            v0 = get_inner_idx(z_idx, th_idx)
            v1 = get_inner_idx(z_idx, th_idx + 1)
            v2 = get_inner_idx(z_idx + 1, th_idx)
            v3 = get_inner_idx(z_idx + 1, th_idx + 1)
            # Flipped winding for inward normals
            faces.extend([[3, v0, v2, v1], [3, v1, v2, v3]])
            
    # -- Join Bottom Ring (z=0) --
    for th_idx in range(N):
        vo0 = get_outer_idx(0, th_idx)
        vo1 = get_outer_idx(0, th_idx + 1)
        vi0 = get_inner_idx(0, th_idx)
        vi1 = get_inner_idx(0, th_idx + 1)
        # Normal targets Down (-Z)
        faces.extend([[3, vo0, vi0, vo1], [3, vo1, vi0, vi1]])
        
    # -- Join Top Ring (z=Z_max) --
    last_z = num_z - 1
    for th_idx in range(N):
        vo0 = get_outer_idx(last_z, th_idx)
        vo1 = get_outer_idx(last_z, th_idx + 1)
        vi0 = get_inner_idx(last_z, th_idx)
        vi1 = get_inner_idx(last_z, th_idx + 1)
        # Normal targets Up (+Z)
        faces.extend([[3, vo0, vo1, vi0], [3, vo1, vi1, vi0]])
        
    # Flatten the faces list for PyVista
    faces_flat = np.hstack(faces)
    
    # Construct Mesh
    mesh = pv.PolyData(vertices, faces_flat)
    
    # Print status
    print(f"Mesh generated: {mesh.n_points} vertices, {mesh.n_cells} faces.")
    
    if not mesh.is_manifold:
        print("Warning: Mesh is not watertight (manifold)! Check geometry logic.")
    else:
        print("Validation: Mesh is watertight and ready for 3D printing.")
        
    print(f"Saving to {output_filename}...")
    mesh.save(output_filename)
    print(f"Success! The STL file was saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    create_adapter()
