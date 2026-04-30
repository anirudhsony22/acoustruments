import pyvista as pv
import create_flute
create_flute.create_flute_stl(style="elbow", num_holes=1, output="test_flute.stl")

mesh = pv.read("test_flute.stl")
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='lightblue', show_edges=True)
plotter.camera_position = 'iso'
plotter.screenshot("flute_screenshot.png")
print("Saved screenshot to flute_screenshot.png")
