import numpy as np
import torch
from nimosef.utils.visualization import label_to_color, compute_label_volumes
from nimosef.metrics.pc_metrics import compute_distance_pts


def test_label_to_color_default_and_custom():
    seg = np.array([[0,1],[2,3]])
    img = label_to_color(seg)
    assert img.shape == (2,2,3)
    assert (img[0,1] == [255,0,0]).all()

    cmap = {0:[1,1,1],1:[2,2,2],2:[3,3,3],3:[4,4,4]}
    img2 = label_to_color(seg, colormap=cmap)
    assert (img2[1,1] == [4,4,4]).all()


def test_compute_label_volumes_total_and_slice():
    seg = np.zeros((4,4,4,2), dtype=int)
    seg[:2,:2,:2,0] = 1
    seg[2:,:2,:2,1] = 2

    vols = compute_label_volumes(seg, labels=[1,2], to_ml=False)
    assert 1 in vols and 2 in vols
    assert vols[1][0] > 0
    assert vols[2][1] > 0

    vols_slice = compute_label_volumes(seg, labels=[1], z_slice=0, to_ml=True)
    assert isinstance(vols_slice, dict)


def test_compute_distance_pts_hf95_and_chamfer():
    pts1 = torch.rand(1,20,3)
    pts2 = torch.rand(1,25,3)
    hf95 = compute_distance_pts(pts1, pts2)
    chamfer = compute_distance_pts(pts1, pts2, chamfer=True)
    assert hf95.shape == (1,)
    assert chamfer.shape == (1,)
