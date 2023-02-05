"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: predict_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import os

# --- Predict dataset --- #
# --- Validation/test dataset --- #
class PredictData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.val_data_dir = val_data_dir
        self.img_names = sorted(os.listdir(val_data_dir))
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.val_data_dir, img_name))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transformed_img = transform_haze(img)

        return transformed_img, img_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.img_names)