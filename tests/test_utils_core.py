import torch
import pytest
import tempfile
import os
from nimosef.utils.core import (
    str2bool, freeze_model_parameters, unfreeze_model_parameters,
    save_checkpoint, LossTracker
)


def test_str2bool_variants():
    true_vals = ["yes", "true", "t", "y", "1"]
    false_vals = ["no", "false", "f", "n", "0"]
    for v in true_vals:
        assert str2bool(v)
    for v in false_vals:
        assert not str2bool(v)    


def test_freeze_unfreeze_model_parameters():
    model = torch.nn.Linear(3, 2)
    unfreeze_model_parameters(model)
    assert all(p.requires_grad for p in model.parameters())
    freeze_model_parameters(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_save_checkpoint_and_load(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    tracker = {"loss": [1.0, 0.5]}

    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(str(ckpt_file), 5, model, opt, scheduler, tracker)
    ckpt = torch.load(ckpt_file, weights_only=False)

    assert "epoch" in ckpt and ckpt["epoch"] == 5
    assert "model_state_dict" in ckpt


def test_loss_tracker_add_save_load(tmp_path):
    tracker = LossTracker()
    tracker.add_loss({"loss": 1.0}, epoch=1)
    tracker.add_loss({"loss": 0.5}, epoch=2)

    file = tmp_path / "losses.pt"
    tracker.save_losses(str(file))

    new_tracker = LossTracker()
    new_tracker.load_losses(str(file))

    assert new_tracker.epoch == [1, 2]
    assert new_tracker.losses[0]["loss"] == 1.0
