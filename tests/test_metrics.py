import numpy as np
from nimosef.metrics.dice import dice_coefficient


def test_dice_perfect_match():
    pred = np.zeros((4, 4, 4, 2), dtype=int)
    target = np.zeros_like(pred)
    pred[1:3, 1:3, 1:3, :] = 1
    target[1:3, 1:3, 1:3, :] = 1

    dice = dice_coefficient(pred, target, per_slice=False, per_frame=False)
    assert np.allclose(dice, 1.0, atol=1e-6)


def test_dice_no_overlap():
    pred = np.zeros((4, 4, 4, 2), dtype=int)
    target = np.zeros_like(pred)
    pred[0:2, 0:2, 0:2, 0] = 1
    target[2:4, 2:4, 2:4, 1] = 1

    dice = dice_coefficient(pred, target, per_slice=False, per_frame=False)
    print(dice)
    assert np.all(dice < 1e-2), f"Expected near-zero Dice, got {dice}"
    assert np.allclose(dice, 0.0, atol=1e-6)


def test_dice_multiclass():
    pred = np.zeros((4, 4, 4, 1), dtype=int)
    target = np.zeros_like(pred)
    pred[:2, :, :, :] = 1
    target[:2, :, :, :] = 1
    pred[2:, :, :, :] = 2
    target[2:, :, :, :] = 2

    dice = dice_coefficient(pred, target, per_slice=False, per_frame=False)
    print(dice.shape)
    assert dice.shape[0] == 2  # two classes (1 and 2)
    assert np.allclose(dice, 1.0, atol=1e-6)


def test_dice_per_slice_and_frame():
    pred = np.ones((4, 4, 4, 2), dtype=int)
    target = np.ones_like(pred)

    dice_slice = dice_coefficient(pred, target, per_slice=True, per_frame=False)
    dice_frame = dice_coefficient(pred, target, per_slice=False, per_frame=True)
    dice_global = dice_coefficient(pred, target, per_slice=False, per_frame=False)

    assert np.allclose(dice_slice, 1.0)
    assert np.allclose(dice_frame, 1.0)
    assert np.allclose(dice_global, 1.0)


def test_dice_smooth_factor():
    """Check that smooth avoids div-by-zero."""
    pred = np.zeros((4, 4, 4, 1), dtype=int)
    target = np.zeros_like(pred)

    dice = dice_coefficient(pred, target, per_slice=False, per_frame=False, smooth=1.0)
    assert np.allclose(dice, 1.0, atol=1e-6)  # empty preds/targets treated as perfect
