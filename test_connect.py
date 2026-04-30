import pyvista as pv
import create_flute
create_flute.create_flute_stl(style="elbow", num_holes=1, output="test_flute.stl")

mesh = pv.read("test_flute.stl")
components = mesh.split_bodies()
print(f"Number of disconnected bodies: {len(components)}")
