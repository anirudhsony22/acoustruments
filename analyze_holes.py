import trimesh
import numpy as np

case = trimesh.load('day-10-phone-case-v7.stl')

# Let's slice at Y = -79.5 (near Y min)
slice_min = case.section(plane_origin=[0, -79.5, 0], plane_normal=[0,1,0])
if slice_min is not None:
    print("Slice near Y min (-79.5): number of entities =", len(slice_min.entities))
    for e in slice_min.entities:
        print(" Entity bounds:", slice_min.vertices[e.points].min(axis=0), "to", slice_min.vertices[e.points].max(axis=0))

# Let's slice at Y = 84.0 (near Y max)
slice_max = case.section(plane_origin=[0, 84.0, 0], plane_normal=[0,1,0])
if slice_max is not None:
    print("Slice near Y max (84.0): number of entities =", len(slice_max.entities))
    for e in slice_max.entities:
        print(" Entity bounds:", slice_max.vertices[e.points].min(axis=0), "to", slice_max.vertices[e.points].max(axis=0))
