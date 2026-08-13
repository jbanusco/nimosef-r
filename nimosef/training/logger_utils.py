import json

class TrainingLogger:
    def __init__(self, log_interval=10, log_filename=None):
        self.log_interval = log_interval
        self.log_filename = log_filename
        if log_filename:
            with open(log_filename, 'w') as f:
                f.write("epoch,loss\n")

    def on_epoch_start(self, epoch):
        pass  # placeholder if you want to track timings

    def on_epoch_end(self, epoch, logs=None):
        if self.log_filename and logs is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"{epoch},{logs}\n")


class LossTracker:
    def __init__(self):
        self.epoch = []
        self.losses = []

    def add_loss(self, loss_dict, epoch):
        self.epoch.append(epoch)
        self.losses.append(loss_dict)

    def save_losses(self, filename):
        data = {"epoch": self.epoch, "losses": self.losses}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_losses(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.epoch = data["epoch"]
        self.losses = data["losses"]
