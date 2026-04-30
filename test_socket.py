import pyvista as pv
import numpy as np
import create_flute

# Mock constants if not imported
create_flute.SOCKET_OD = 14.6
create_flute.SOCKET_ID = 10.6
create_flute.SOCKET_DEPTH = 8.0

sock = create_flute._make_socket(base_center=(0,0,0), direction=(0,0,1), r_pipe_out=5.0, r_pipe_in=3.0)
sock.save("test_socket.stl")
print("Saved test_socket.stl")
