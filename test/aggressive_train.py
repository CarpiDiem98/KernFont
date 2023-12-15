from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
import tensorfont as tf
import glob
import library.path_font as path
from library.conversion_to_array import array_conversion
from library.resize import resize_cv


class Font(Dataset):
    def __init__(self, folder_csv: str, scale_value: int, transform=None):
        self.folder_csv = folder_csv
        csv_paths = sorted(glob.glob(self.folder_csv + "/*.csv"))  # 3091
        self.csv_paths = csv_paths
        self.transform = transform
        self.scale_value = scale_value

    def __getitem__(self, idx):
        df = array_conversion(self.csv_paths[idx])
        letter1 = []
        letter2 = []
        label = []
        for x1, x2, y in zip(df.X1, df.X2, df.y):
            letter1.append(resize_cv(x1, self.scale_value))
            letter2.append(resize_cv(x2, self.scale_value))
            label.append(y)

        return letter1, letter2, label

    def __len__(self):
        return len(self.csv_paths)


folder_csv = '/ext/home/emanuele/kernfont/progettoKerning/dataframe'
dataset = Font(folder_csv=folder_csv)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print('len: dataset {}, train {}, test {}'.format(
    len(dataset), len(train_dataset), len(test_dataset)))
