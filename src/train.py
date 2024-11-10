#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import torch
import torch.backends.cudnn
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import MIT
import argparse
from pathlib import Path
from src.MrCNN import MrCNN
from src.metrics import calculate_auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Republish Predicting Eye Fixations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=1,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--data-aug-hflip",
    action = "store_true",
)
parser.add_argument(
    "--data-aug-brightness",
    default=0,
    type=float
)
parser.add_argument(
    "--dropout",
    default=0,
    type=float,
)

# checkpoints
parser.add_argument(
    "--checkpoint-path",
    default= Path("checkpoints"),
    type=Path
)

parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default = 1,
    help="Save a checkpoint every N epochs"
)

parser.add_argument(
    "--resume-checkpoint",
    default= Path("checkpoints"),
    type=Path
)

parser.add_argument(
    "--start-epoch",
    type = int,
    default = 0,
)

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

def main(args):
    transform_train_list = []

    if args.data_aug_hflip:
        transform_train_list.append(transforms.RandomHorizontalFlip())

    if args.data_aug_brightness > 0:
        transform_train_list.append(transforms.ColorJitter(brightness=args.data_aug_brightness))

    # Only apply data augmentation to the training data
    transform_train_list.append(transforms.ToTensor())
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.ToTensor()


    train_dataset_path = '../dataset/train_data.pth.tar'
    train_dataset = MIT(train_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset_path = '../dataset/val_data.pth.tar'
    val_dataset = MIT(val_dataset_path)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    test_dataset_path = '../dataset/test_data.pth.tar'
    test_dataset = MIT(test_dataset_path)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # TODO: some duplidate code here
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # MrCNN model
    # TODO: Change the parameters
    model = MrCNN()


    # # checkpoint model resume
    # if args.resume_checkpoint.exists():
    #     start_dict = torch.load(args.resume_checkpoint)
    #     print(f"Loading model from {args.resume_checkpoint} that achieved {start_dict['accuracy'] * 100:.2f}% accuracy")
    #     model.load_state_dict(start_dict)
    # else:
    #     print("Training from scratch")

    criterion = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=0.001)

    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, device, schedular,
        checkpoint_path = args.checkpoint_path,
        checkpoint_frequency = args.checkpoint_frequency,
        args = args
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
            schedular: torch.optim.lr_scheduler.StepLR,
            checkpoint_path: Path,
            checkpoint_frequency: int,
            args
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.schedular = schedular
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency
        self.args = args

    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0,
            args = None
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            total_loss = 0

            for batch in self.train_loader:
                X,y = batch
                X_400_crop = X[:, 0, :, :, :].to(device)
                X_250_crop = X[:, 1, :, :, :].to(device)
                X_150_crop = X[:, 2, :, :, :].to(device)
                data_load_end_time = time.time()

                output = self.model(X_400_crop, X_250_crop, X_150_crop)
                y = y.view(-1, 1).float().to(device)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    preds = output
                    accuracy = get_accuracy(preds,y)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1

                data_load_start_time = time.time()
            # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
            # Average validation loss and accuracy
            # avg_val_loss = loss / len(self.val_loader)
            self.schedular.step()
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
            # if ((epoch + 1) % self.checkpoint_frequency) == 0 and self.checkpoint_path:
            #     checkpoint_file = os.path.join(self.checkpoint_path, f"model_epoch_{epoch + 1}.pth")
            #     # torch.save({
            #     #     'args': args,
            #     #     'model': self.model.state_dict(),
            #     #     'accuracy': accuracy,
            #     #     'epoch': epoch,
            #     # }, checkpoint_file)
            #     torch.save(self.model.state_dict(), checkpoint_file)
            #     print(f"Checkpoint saved at {checkpoint_file}")

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy :2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy",
            {"train": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"train": float(loss.item())},
            self.step
        )
        self.summary_writer.add_scalar(
            "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
            "time/data", step_time, self.step
        )

    # TODO: Implement the validate method
    def validate(self):
        preds = {}
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                X, _ = batch  # no ground truth
                X_400_crop = X[:, 0, :, :, :].to(device)
                X_250_crop = X[:, 1, :, :, :].to(device)
                X_150_crop = X[:, 2, :, :, :].to(device)

                output = self.model(X_400_crop, X_250_crop, X_150_crop)
                preds[i] = output.squeeze().cpu().numpy()



def get_accuracy (preds, y):
    assert len(preds) == len(y)
    preds = (preds > 0.5).float()
    return float((y == preds).sum().item() / len(y))


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix =(
            f"Mr-CNN"
            f"dropout={args.dropout}_"
            f"bs={args.batch_size}_"
            f"lr={args.learning_rate}_"
            f"momentum=0.9_" +
            ("hflip_" if args.data_aug_hflip else "") +
            f"brightness={args.data_aug_brightness}"
            f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())