#!/usr/bin/env python3
import os
import time
from multiprocessing import cpu_count
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
from MrCNNs import MrCNNs

from metrics import calculate_auc, calculate_auc_with_shuffle

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
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
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

parser.add_argument(
    "--model",
    choices=['MrCNN', 'MrCNNs'], default='MrCNNs',
    help="Choose model type: 'MrCNN' for separate branches or 'MrCNNs' for shared branches"
)

def main(args):
    # load the cooresponding dataset
    train_dataset_path = '../dataset/train_data.pth.tar'
    train_dataset = MIT(train_dataset_path)

    val_dataset_path = '../dataset/val_data.pth.tar'
    val_dataset = MIT(val_dataset_path)

    # load the ground truth for the validation dataset, if not already loaded
    val_ground_truth_path = '../dataset/val_ground_truth'
    if not Path(val_ground_truth_path).exists():
        load_ground_truth(dataset=val_dataset, img_dataset_path='../dataset/ALLFIXATIONMAPS', target_folder_path=val_ground_truth_path)

    if args.model == 'MrCNN':
        model = MrCNN()
    elif args.model == 'MrCNNs':
        model = MrCNNs(first_batch_only=True, visualize=True)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")

    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    checkpoint_path = f'./checkpoint/{args.model}_run_{log_dir.split("_")[-1]}'
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
        args = args
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
        args = args if args is not None else self.args
        best_auc = 0
        current_auc = 0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            # For feature map visulisation:
            # Reset `first_batch_processed` at the start of each epoch
            self.model.first_batch_processed = False
            data_load_start_time = time.time()
            total_loss = 0
            total_accuracy = 0
            num_batches = len(self.train_loader)
            
            for batch in self.train_loader:
                img, label = batch
                # separate the three inputs, each sampled from different resolution
                img_400_crop = img[:, 0, :, :, :].to(device)
                img_250_crop = img[:, 1, :, :, :].to(device)
                img_150_crop = img[:, 2, :, :, :].to(device)
                data_load_end_time = time.time()

                output = self.model(img_400_crop, img_250_crop, img_150_crop)
                # training set label
                label = label.view(-1, 1).float().to(device)
                loss = self.criterion(output, label)
                total_loss += loss.item() # accumulate loss

                # optimising the model
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    preds = output
                    accuracy = get_accuracy(preds, label)
                    total_accuracy += accuracy  # accumulate accuracy

                # print the metrics
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
            print(f"epoch: [{epoch + 1}], " f"Average Loss: {avg_epoch_loss:.5f}," f"Average Accuracy: {avg_epoch_accuracy:.2f}")
                
            self.schedular.step()
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # get the validation AUC score
            if ((epoch + 1) % val_frequency) == 0:
                self.model.eval()

                auc, shuffle_auc = self.validate()

                # using the shuffled AUC score as the benchmark for the best model
                current_auc = shuffle_auc
                print(f"Epoch {epoch + 1} validation AUC score {auc}")
                print(f"Epoch {epoch + 1} validation Shuffled AUC score {shuffle_auc}")

                # add the auc and shuffle_auc to tensorboard
                self.summary_writer.add_scalar("AUC", auc, self.step)
                self.summary_writer.add_scalar("Shuffled AUC", shuffle_auc, self.step)

                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                # self.model.train() line 198

            # save the best model, using the shuffled AUC score as the benchmark
            if current_auc > best_auc:
                best_auc = current_auc
                checkpoint_file = os.path.join(self.checkpoint_path, f"best.pth")
                torch.save(self.model.state_dict(), checkpoint_file)
                print(f"Best model so far saved at {checkpoint_file}")

            # checkpoint frequency saving
            if ((epoch + 1) % self.checkpoint_frequency) == 0 and (epoch + 1) != epochs:
                checkpoint_file = os.path.join(self.checkpoint_path, f"epoch_{epoch+ 1}.pth")
                torch.save({
                    'args': args,
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")

            # final model saving, if the current model is the best model, save it as the final best model
            if (epoch + 1) == epochs:
                if current_auc > best_auc:
                    checkpoint_file = os.path.join(self.checkpoint_path, f"final_best_model.pth")
                    torch.save({
                        'args': args,
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_file)
                    print(f"the final model is the best model, saved at {checkpoint_file}")
                else:
                    checkpoint_file = os.path.join(self.checkpoint_path, f"final_model.pth")
                    torch.save({
                        'args': args,
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_file)
                    print(f"final model saved at {checkpoint_file}")


    # print the metrics, in each step
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch + 1}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy :2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    # summary writer log metrics
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

    # validate the model
    def validate(self):
        # two dictionaries to store the ground truth and the predictions
        ground_truth = {}
        preds = {}
        # initialise the single prediction tensor
        single_img_preds = torch.zeros(50, 50)
        with torch.no_grad():
            for index in tqdm(range(self.val_dataset.__len__()), desc="Validation"):
                # get the crop index
                crop_index = index % 2500
                # put them in the 50x50 grid
                crop_x = crop_index % 50
                crop_y = crop_index // 50
                # get the validation inputs
                img, label = self.val_dataset.__getitem__(index)  # no label
                img_400_crop = img[0, :, :, :].unsqueeze(0).to(device)
                img_250_crop = img[1, :, :, :].unsqueeze(0).to(device)
                img_150_crop = img[2, :, :, :].unsqueeze(0).to(device)

                output = self.model(img_400_crop, img_250_crop, img_150_crop)

                # put the output in the single prediction tensor
                single_img_preds[crop_y][crop_x] = output.squeeze().cpu()

                # if the 50x50 grid is filled, resize the prediction to the ground truth size
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
            # cause the calculate_auc is averaging the auc score of all the images
            # so we just pass the two dictionaries once to get the average auc score

            # calculate AUC
            auc = calculate_auc(preds, ground_truth)

            # calculate shuffled-AUC
            shuffle_auc = calculate_auc_with_shuffle(preds, ground_truth)

            return auc, shuffle_auc

def get_accuracy(preds, y,threshold=0.5):
    tp = tn = fp = fn = 0
    pred_binary = [1 if p > threshold else 0 for p in preds]
    for i in range(len(y)):
        if y[i] == 1 and pred_binary[i] == 1:
            tp += 1
        elif y[i] == 0 and pred_binary[i] == 0:
            tn += 1
        elif y[i] == 0 and pred_binary[i] == 1:
            fp += 1
        elif y[i] == 1 and pred_binary[i] == 0:
            fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return accuracy


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
            f"{args.model}_"
            f"bs={args.batch_size}_"
            f"lr={args.learning_rate}_"
            f"Adam_" +
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