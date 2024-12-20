#!/usr/bin/env python3
import torch
import torch.backends.cudnn
# from accelerate.commands.config.update import description

from dataset import MIT, load_ground_truth
import argparse
from pathlib import Path
from MrCNN import MrCNN
from metrics import calculate_auc, calculate_auc_with_shuffle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from MrCNNs import MrCNNs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Republish Predicting Eye Fixations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--model',
    choices=['MrCNN', 'MrCNNs'], default='MrCNNs',
    help="Choose model type: 'MrCNN' for separate branches or 'MrCNNs' for shared branches"
)

parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
)

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the model.pth file",
)


def main(args):
    test_data_path = '../dataset/test_data.pth.tar'
    test_dataset = MIT(test_data_path)

    test_ground_truth_path = '../dataset/test_ground_truth'
    if not Path(test_ground_truth_path).exists():
        load_ground_truth(dataset=test_dataset, img_dataset_path='../dataset/ALLFIXATIONMAPS', target_folder_path=test_ground_truth_path)

    if args.model == 'MrCNN':
        model = MrCNN(dropout=args.dropout)
    elif args.model == 'MrCNNs':
        model = MrCNNs(dropout=args.dropout, first_batch_only=False, visualize=False)
    else:
        raise ValueError("Invalid model type")

    checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    model.load_state_dict(state_dict, strict=False)
    auc, shuffle_auc = validate(model, test_dataset)

    print("Test auc score: ", auc )
    print("Shuffled Test auc score: ", shuffle_auc)




def validate(model, dataset):
    model.to(device)
    model.eval()
    ground_truth = {}
    preds = {}
    single_img_preds = torch.zeros(50, 50)
    with torch.no_grad():
        for index in tqdm(range(dataset.__len__()), desc="Testing"):
            crop_index = index % 2500
            crop_x = crop_index % 50
            crop_y = crop_index // 50
            img, label = dataset.__getitem__(index)  # no label
            img_400_crop = img[0, :, :, :].unsqueeze(0).to(device)
            # print(img_400_crop.shape)
            img_250_crop = img[1, :, :, :].unsqueeze(0).to(device)
            # print(img_250_crop.shape)
            img_150_crop = img[2, :, :, :].unsqueeze(0).to(device)
            # print(img_150_crop.shape)

            output = model(img_400_crop, img_250_crop, img_150_crop)
            single_img_preds[crop_y][crop_x] = output.squeeze().cpu()

            if (index+1) % 2500 == 0:
                filename = dataset.__getfile__(index)["file"]
                gt = plt.imread(f'../dataset/test_ground_truth/{filename}_fixMap.jpg')
                gt_height, gt_width = gt.shape[:2]
                resized_preds = F.interpolate(
                    single_img_preds.unsqueeze(0).unsqueeze(0),
                    size=(gt_height, gt_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().numpy()
                preds[filename]= resized_preds
                ground_truth[filename] = gt
                # VISUALISE THE RESIZED PREDICTIONS
                # plt.imshow(resized_preds)
                # plt.show()

        # calculate AUC
        auc = calculate_auc(preds, ground_truth)
        # print("Non-shuffled Test auc score: ", auc)

        # calculate shuffled-AUC
        shuffled_auc = calculate_auc_with_shuffle(preds, ground_truth)
        # print("Shuffled Test auc score: ", shuffled_auc)

        return auc, shuffled_auc




if __name__ == "__main__":
    main(parser.parse_args())