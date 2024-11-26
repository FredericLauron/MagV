from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import numpy as np

import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True




class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        
        return self.transform(image)

    def __len__(self):
        return len(self.image_path)