#!/usr/bin/env python3
import os
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
from dataset import MIT, load_ground_truth
import argparse
from pathlib import Path
from MrCNN import MrCNN
from metrics import calculate_auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

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
    default=10,
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
    "--dropout",
    default=0,
    type=float,
)

parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default = 5,
    help="Save a checkpoint every N epochs"
)

def main(args):
    train_dataset_path = '../dataset/train_data.pth.tar'
    train_dataset = MIT(train_dataset_path)

    val_dataset_path = '../dataset/val_data.pth.tar'
    val_dataset = MIT(val_dataset_path)

    val_ground_truth_path = '../dataset/val_ground_truth'
    if not Path(val_ground_truth_path).exists():
        load_ground_truth(dataset=val_dataset, img_dataset_path='../dataset/ALLFIXATIONMAPS', target_folder_path=val_ground_truth_path)

    # MrCNN model
    # TODO: Change the parameters
    model = MrCNN()

    criterion = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=0.001)

    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")

    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    checkpoint_path = f'./checkpoint/run_{log_dir.split("_")[-1]}'
    if not Path(checkpoint_path).exists():
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model, train_dataset, val_dataset, criterion, optimizer, summary_writer, device, schedular,
        checkpoint_path = checkpoint_path,
        checkpoint_frequency= args.checkpoint_frequency,
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
            train_dataset: MIT,
            val_dataset: MIT,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
            schedular: torch.optim.lr_scheduler.StepLR,
            checkpoint_path: str,
            checkpoint_frequency: int,
            args
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.schedular = schedular
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency
        self.args = args
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count,
        )

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
        best_auc = 0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            total_loss = 0
            total_accuracy = 0
            num_batches = len(self.train_loader)
            
            
            for batch in self.train_loader:
                img,label = batch
                img_400_crop = img[:, 0, :, :, :].to(device)
                img_250_crop = img[:, 1, :, :, :].to(device)
                img_150_crop = img[:, 2, :, :, :].to(device)
                data_load_end_time = time.time()

                output = self.model(img_400_crop, img_250_crop, img_150_crop)
                label = label.view(-1, 1).float().to(device)
                loss = self.criterion(output, label)
                total_loss += loss.item() # accumulate loss
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    preds = output
                    accuracy = get_accuracy(preds,label)
                    total_accuracy += accuracy # accumulate accuracy

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1

                data_load_start_time = time.time()
                
                # The average batch loss and batch accuracy
                avg_epoch_loss = total_loss / num_batches
                avg_epoch_accuracy = total_accuracy / num_batches
                print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.5f}, Average Accuracy: {avg_epoch_accuracy:.2f}%')
                
            # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
            # Average validation loss and accuracy
            # avg_val_loss = loss / len(self.val_loader)
            self.schedular.step()
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if ((epoch + 1) % val_frequency) == 0:
                self.model.eval()
                auc = self.validate()
                print(f"Epoch {epoch} validation AUC score {auc}")
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                # self.model.train()
                if auc > best_auc:
                    best_auc = auc
                    checkpoint_file = os.path.join(self.checkpoint_path, f"best_model.pth")
                    torch.save(self.model.state_dict(), checkpoint_file)


            if ((epoch + 1) % self.checkpoint_frequency) == 0:
                checkpoint_file = os.path.join(self.checkpoint_path, f"model_epoch_{epoch}.pth")
                torch.save({
                    'args': args,
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")

            if (epoch + 1) == epochs:
                checkpoint_file = os.path.join(self.checkpoint_path, f"final_model.pth")
                torch.save({
                    'args': args,
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")

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
        ground_truth = {}
        preds = {}
        single_img_preds = torch.zeros(50, 50)
        with torch.no_grad():
            for index in tqdm(range(self.val_dataset.__len__()), desc="Validation"):
                crop_index = index % 2500
                crop_x = crop_index % 50
                crop_y = crop_index // 50
                img, label = self.val_dataset.__getitem__(index)  # no label
                img_400_crop = img[0, :, :, :].unsqueeze(0).to(device)
                # print(img_400_crop.shape)
                img_250_crop = img[1, :, :, :].unsqueeze(0).to(device)
                # print(img_250_crop.shape)
                img_150_crop = img[2, :, :, :].unsqueeze(0).to(device)
                # print(img_150_crop.shape)

                output = self.model(img_400_crop, img_250_crop, img_150_crop)
                single_img_preds[crop_y][crop_x] = output.squeeze().cpu()

                if (index+1) % 2500 == 0:
                    filename = self.val_dataset.__getfile__(index)["file"]
                    gt = plt.imread(f'../dataset/val_ground_truth/{filename}_fixMap.jpg')
                    gt_height, gt_width = gt.shape[:2]

                    resized_preds = F.interpolate(
                        single_img_preds.unsqueeze(0).unsqueeze(0),
                        size=(gt_height, gt_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    preds[filename]= resized_preds
                    ground_truth[filename] = gt
                    # preds[sample_index] = single_img_preds
                    # val_dataset.__getfile__(index)["file"]
                    # gt = plt.imread(f'../dataset/val_ground_truth/{self.val_dataset.__getfile__(index)["file"]}_fixMap.jpg')
                    # gt = torch.tensor(gt).float()
                    # my_dict['city'] = 'New York'

            # calculate AUC
            auc = calculate_auc(preds, ground_truth)
            return auc



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