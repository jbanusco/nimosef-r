import numpy as np
import pyvista as pv
import tempfile
import os
from nimosef.utils.viz3d import generate_simple_pc, generate_simple_mesh, visualize_4d_pyvista


def test_generate_simple_pc_and_mesh():
    seg = np.zeros((5,5,5), dtype=int)
    seg[1:3,1:3,1:3] = 1
    pc = generate_simple_pc(seg, labels=[1])
    assert isinstance(pc, list) and len(pc) > 0
    pts, color = pc[0]
    assert pts.shape[1] == 3

    mesh = generate_simple_mesh(seg, labels=[1])
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0


# def test_visualize_4d_pyvista_creates_gif(tmp_path):
#     # Use a simple mesh
#     seg = np.zeros((5,5,5), dtype=int)
#     seg[1:3,1:3,1:3] = 1
#     mesh = generate_simple_mesh(seg, labels=[1])

#     output_dir = tmp_path / "out"
#     visualize_4d_pyvista([ [mesh] ], output_dir=str(output_dir), gif_name="test.gif", method="mesh")

#     gif_file = output_dir / "test.gif"
#     assert gif_file.exists()
