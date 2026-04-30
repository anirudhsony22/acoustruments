import trimesh
import numpy as np

def create_attached_case(output_file='case_with_adapters.stl'):
    # Load case and adapter
    print("Loading meshes...")
    case = trimesh.load('day-10-phone-case-v7.stl')
    adapter = trimesh.load('adapter.stl')
    
    # Calculate case center in X and Z
    x_min, y_min, z_min = case.bounds[0]
    x_max, y_max, z_max = case.bounds[1]
    
    center_x = (x_min + x_max) / 2.0
    center_z = (z_min + z_max) / 2.0
    
    print(f"Case Center: X={center_x:.2f}, Z={center_z:.2f}")
    print(f"Case Y_min: {y_min:.2f}")
    
    # Phone case bottom face is at roughly Y = y_min.
    # The normal points in the -Y direction.
    # The adapter was built such that its base is at Z=0, pointing towards +Z.
    # We want the base at the case's bottom wall (Y=y_min), and the adapter pointing outwards (towards -Y).
    
    # Rotate adapter by 90 degrees around X-axis
    # This transforms +Z to -Y, and +Y to +Z
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    
    # We will create two adapters, one for mic, one for speaker
    
    # iPhone 13 Pro Max dimensions typically have mic to the left and speaker to the right of the charging port
    # Offset from center_x
    mic_offset_x = -15.0
    speaker_offset_x = 18.0
    
    # -- 1. Setup Microphone Adapter --
    mic_adapter = adapter.copy()
    mic_adapter.apply_transform(rot_matrix)
    
    # Translate to mic position
    # The base of the rotated adapter is now at Y=0.
    # Move it to Y=y_min. To make sure it embeds slightly into the case wall for a watertight union,
    # we can add a small offset, say Y = y_min + 1.0
    mic_trans = trimesh.transformations.translation_matrix([center_x + mic_offset_x, y_min + 1.0, center_z])
    mic_adapter.apply_transform(mic_trans)
    
    # -- 2. Setup Speaker Adapter --
    speaker_adapter = adapter.copy()
    speaker_adapter.apply_transform(rot_matrix)
    
    speaker_trans = trimesh.transformations.translation_matrix([center_x + speaker_offset_x, y_min + 1.0, center_z])
    speaker_adapter.apply_transform(speaker_trans)
    
    # -- Combine all meshes --
    # For many 3D slicers, simply concatenating the meshes is completely fine.
    print("Combining meshes...")
    combined = trimesh.util.concatenate([case, mic_adapter, speaker_adapter])
    
    # Export
    print(f"Exporting to {output_file}...")
    combined.export(output_file)
    print("Done!")

if __name__ == "__main__":
    create_attached_case()
