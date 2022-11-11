from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd
import torch

class LoadImage(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)