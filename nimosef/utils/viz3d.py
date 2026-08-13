import os
import numpy as np
from skimage import measure
import pyvista as pv
from PIL import Image
import subprocess


def visualize_4d_pyvista(data, output_dir="output", gif_name="animation.gif", method="mesh"):
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.background_color = "white"

    for t, frame in enumerate(data):
        plotter.clear()
        if method == "mesh":
            plotter.add_mesh(frame, color="white", opacity=0.10, show_edges=True, edge_color="black")
        elif method == "point_cloud":
            for points, color in frame:
                plotter.add_points(points, color=color, point_size=3, render_points_as_spheres=True, opacity=0.75)

        png_path = os.path.join(output_dir, f"frame_{t:03d}.png")
        plotter.screenshot(png_path)
        frames.append(png_path)

    plotter.close()

    gif_frames = [Image.open(frame) for frame in frames]
    gif_frames[0].save(
        os.path.join(output_dir, gif_name),
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0,
    )

    subprocess.run([f"rm -f {output_dir}/frame_*.png"], shell=True)


def generate_simple_pc(segmentation, labels=None, spacing=(1.0, 1.0, 1.0), flip_z=False):
    label_colors = {1: "red", 2: "green", 3: "blue", 4: "yellow"}
    points_list = []

    if labels is None:
        labels = np.unique(segmentation)
        labels = labels[labels > 0]

    for label in labels:
        indices = np.argwhere(segmentation == label)
        if flip_z:
            indices[:, 2] = -indices[:, 2]
        points = indices * np.array(spacing)
        color = label_colors.get(label, "gray")
        points_list.append((points, color))

    return points_list


def generate_simple_mesh(segmentation, labels=None, spacing=(1.0, 1.0, 1.0), flip_z=False):
    meshes = []
    if labels is None:
        labels = np.unique(segmentation)
        labels = labels[labels > 0]

    for label in labels:
        mask = segmentation == label
        if mask.sum() == 0:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0)
        verts *= np.array(spacing)
        if flip_z:
            verts[:, 2] = -verts[:, 2]
        faces = np.hstack([[3] + list(face) for face in faces])
        mesh = pv.PolyData(verts, faces)
        meshes.append(mesh)

    combined_mesh = pv.PolyData()
    for mesh in meshes:
        combined_mesh += mesh

    return combined_mesh
