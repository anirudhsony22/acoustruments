import pyvista as pv
import numpy as np

case = pv.read('day-10-phone-case-v7.stl')
bounds = case.bounds
print("Case Bounds (xmin, xmax, ymin, ymax, zmin, zmax):", bounds)
print("Lengths (x, y, z):", bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

# Find the bottom edge where z or y is minimal, or whichever axis is longest.
# usually the longest axis is for height.
# For iphone 13 pro max, the height is ~160.8mm, width ~78.1mm, depth ~7.65mm
