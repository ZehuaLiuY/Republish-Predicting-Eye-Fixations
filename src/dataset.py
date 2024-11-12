import torch.nn.functional as F
import torch

from torch.utils import data
from torch import Tensor
from typing import Tuple
from pathlib import Path
import shutil

def crop_to_region(coords: Tuple[int], img: Tensor, crop_size: int=42) -> Tensor:
    """ 
    Given coordinates in the form Tuple[int](y, x), return a cropped
    sample of the input imaged centred at (y, x), matching the input size.
    Args:
        coords (Tuple[int]): The input coordinates (y, x) where the crop will be
        centred.
        img (Tensor): The input image, either 3x400x400, 3x250x250, 3x150x150
        crop_size (int, optional): The size of the returned crop. Defaults to 42.

    Returns:
        Tensor: The image cropped with central coordinates at (y, x) of size 
        (3 x size x size)
    """
    _, H, W = img.shape
    y, x = coords
    y_min, x_min = max(0, y-crop_size//2), max(0, x-crop_size//2)
    y_max, x_max = min(H, y+crop_size//2), min(W, x+crop_size//2)
    region = img[:, y_min:y_max, x_min:x_max]
    if region.shape[1] < crop_size:
        to_pad = crop_size - region.shape[1]
        padding = (0, 0, to_pad, 0) if (y-crop_size//2) < 0 else (0, 0, 0, to_pad)
        region = F.pad(region, padding, mode='replicate')

    if region.shape[2] < crop_size:
        to_pad = crop_size - region.shape[2]
        padding = (to_pad, 0, 0, 0) if (x-crop_size//2) < 0 else (0, to_pad, 0, 0)
        region = F.pad(region, padding, mode='replicate')
    return region

class MIT(data.Dataset):
    def __init__(self, dataset_path: str):
        """
        Given the dataset path, create the MIT dataset. Creates the
        variable self.dataset which is a list of dictionaries with three keys:
            1) X: For train the crop of image. This is of shape [3, 3, 42, 42]. The 
                first dim represents the crop across each different scale
                (400x400, 250x250, 150x150), the second dim is the colour
                channels C, followed by H and W (42x42). For inference, this is 
                the full size image of shape [3, H, W].
            2) y: The label for the crop. 1 = a fixation point, 0 = a
                non-fixation point. -1 = Unlabelled i.e. val and test
            3) file: The file name the crops were extracted from.
            
        If the dataset belongs to val or test, there are 4 additional keys:
            1) X_400: The image resized to 400x400
            2) X_250: The image resized to 250x250
            3) X_150: The image resized to 150x150
            4) spatial_coords: The centre coordinates of all 50x50 (2500) crops
            
        These additional keys help to load the different scales within the
        dataloader itself in a timely manner. Precomputing all crops requires too
        much storage for the lab machines, and resizing/cropping on the fly
        slows down the dataloader, so this is a happy balance.
        Args:
            dataset_path (str): Path to train/val/test.pth.tar
        """
        self.dataset = torch.load(dataset_path, weights_only=True)
        self.mode = 'train' if 'train' in dataset_path else 'inference'
        self.num_crops = 2500 if self.mode == 'inference' else 1

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        """
        Given the index from the DataLoader, return the image crop(s) and label
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            Tuple[Tensor, int]: A two-element tuple consisting of: 
                1) img (Tensor): The image crop of shape [3, 3, 42, 42]. The 
                first dim represents the crop across each different scale
                (400x400, 250x250, 150x150), the second dim is the colour
                channels C, followed by H and W (42x42).
                2) label (int): The label for this crop. 1 = a fixation point, 
                0 = a non-fixation point. -1 = Unlabelled i.e. val and test.
        """
        sample_index = index // self.num_crops
        
        img = self.dataset[sample_index]['X']
        
        # Inference crops are not precomputed due to file size, do here instead
        if self.mode == 'inference': 
            _, H, W = img.shape
            crop_index = index % self.num_crops
            crop_y, crop_x = self.dataset[sample_index]['spatial_coords'][crop_index]
            scales = []
            for size in ['X_400', 'X_250', 'X_150']:
                scaled_img = self.dataset[sample_index][size]
                y_ratio, x_ratio = scaled_img.shape[1] / H, scaled_img.shape[2] / W
                
                # Need to rescale the crops central coordinate.
                scaled_coords = (int(y_ratio * crop_y), int(x_ratio * crop_x))
                crops = crop_to_region(scaled_coords, scaled_img)
                scales.append(crops)
            img = torch.stack(scales, axis=1)
            
        label = self.dataset[sample_index]['y']

        return img, label

    def __len__(self):
        """
        Returns the length of the dataset (length of the list of dictionaries * number
        of crops). 
        __len()__ always needs to be defined so that the DataLoader
            can create the batches
        Returns:
            len(self.dataset) (int): the length of the list of dictionaries * number of
            crops.
        """
        return len(self.dataset) * self.num_crops


    def __getfile__(self, index):
        """
        Given the index from the DataLoader, return the image crop(s) and label
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            Tuple[Tensor, int]: A two-element tuple consisting of: 
                1) img (Tensor): The image crop of shape [3, 3, 42, 42]. The 
                first dim represents the crop across each different scale
                (400x400, 250x250, 150x150), the second dim is the colour
                channels C, followed by H and W (42x42).
                2) label (int): The label for this crop. 1 = a fixation point, 
                0 = a non-fixation point. -1 = Unlabelled i.e. val and test.
        """
        sample_index = index // self.num_crops
        
        

        # print all key in self.dataset[sample_index] dictionary
        # for key in self.dataset[sample_index]:
        #     print(key)

        img_filename = self.dataset[sample_index]['file'].split('.')[0]

        img = self.dataset[sample_index]['X']
        _, H, W = img.shape

        return {"file": img_filename, "img": img, "H": H, "W": W}


def load_ground_truth(dataset: MIT, img_dataset_path: str, target_folder_path: str):
    """
    Given the dataset and the path to the image dataset, load the ground truth
    fixation maps and save them to the target folder path.

    Args:
        dataset (MIT): The dataset object.
        img_dataset_path (str): The path to the image dataset.
        target_folder_path (str): The path to save the ground truth fixation maps.
    """

    if dataset is not None and dataset.__len__() > 0:
        index_list = list(range(0, dataset.__len__(), 2500))
        for index in index_list:
            file  = dataset.__getfile__(index)
            filename = file['file'] + "_fixMap.jpg"
            # if do not exist test_ground_truth, then create the folder
            if not Path(target_folder_path).exists():
                Path(target_folder_path).mkdir(parents=True, exist_ok=True)
            
            # if img_dataset_path folder not exists, then raise the error
            if not Path(img_dataset_path).exists():
                raise FileNotFoundError(f"Path {img_dataset_path} not found")

            # search the filename in the ALLFIXATIONMAPS folder, if it exists, then copy this image file
            if Path(f"{img_dataset_path}/{filename}").exists():
                shutil.copy(f"{img_dataset_path}/{filename}", f"{target_folder_path}/{filename}")
            else:
                # if not exists, then raise an warning
                print(f"File {filename} not found in {img_dataset_path}")
    

