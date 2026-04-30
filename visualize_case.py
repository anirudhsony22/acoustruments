import pyvista as pv
import numpy as np

pv.global_theme.off_screen = True

case = pv.read('day-10-phone-case-v7.stl')

plotter = pv.Plotter()
plotter.add_mesh(case, color='lightblue', opacity=0.5)

# Assuming center X is 2.59
center_x = 2.59
y_min = case.bounds[2]
z_center = (case.bounds[4] + case.bounds[5]) / 2

# Add markers for mic and speaker
mic_center = [center_x - 15.0, y_min, z_center]
speaker_center = [center_x + 18.0, y_min, z_center]

plotter.add_mesh(pv.Sphere(radius=2, center=mic_center), color='red')
plotter.add_mesh(pv.Sphere(radius=2, center=speaker_center), color='green')

plotter.camera_position = 'xy'
plotter.camera.position = (center_x, y_min - 50, z_center + 50)
plotter.camera.focal_point = (center_x, y_min + 20, z_center)
try:
    plotter.screenshot('case_bottom.png')
    print("Screenshot saved to case_bottom.png")
except Exception as e:
    print("Screenshot failed:", e)
