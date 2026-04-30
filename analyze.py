import trimesh
import numpy as np

case = trimesh.load('day-10-phone-case-v7.stl')
print("Case Bounds:", case.bounds)
print("Case Extents (x,y,z):", case.extents)
print("Center of Mass:", case.center_mass)

adapter = trimesh.load('adapter.stl')
print("Adapter Extents:", adapter.extents)
