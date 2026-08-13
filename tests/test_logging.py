import os
from nimosef.training.logger_utils import TrainingLogger, LossTracker


def test_traininglogger_writes(tmp_path):
    logfile = tmp_path / "log.csv"
    logger = TrainingLogger(log_filename=str(logfile))
    logger.on_epoch_end(0, logs=0.123)
    with open(logfile) as f:
        content = f.read()
    assert "0,0.123" in content


def test_losstracker_roundtrip(tmp_path):
    tracker = LossTracker()
    tracker.add_loss({"loss": 1.0}, 0)
    fname = tmp_path / "loss.json"
    tracker.save_losses(fname)
    tracker2 = LossTracker()
    tracker2.load_losses(fname)
    assert tracker2.losses[0]["loss"] == 1.0
