import numpy as np
import torch
from torchvision.transforms import transforms
import pandas as pd
import cv2


def resize_row(row, scale_value):
    if row.shape[1] < scale_value:
        # calcoliamo la quantità di padding da aggiungere su ogni lato (colonne)
        padding_cols = (scale_value - row.shape[1]) // 2
        # creiamo un array di padding con i valori desiderati (in questo caso 0)
        padding = ((0, 0), (padding_cols, padding_cols))
        padded_array = np.pad(row, padding, mode='constant', constant_values=0)
        # verifica che la dimensione sia (32, 32)
        if padded_array.shape == (scale_value, scale_value-1):
            padded_array = np.pad(
                padded_array, [(0, 0), (0, 1)], mode='constant', constant_values=0)
        #    raise ValueError("La dimensione dell'array non è (32, 32)")
    elif row.shape[1] == scale_value+1:
        padded_array = row[:, :-1]
    else:
        padded_array = row
    return np.float32(padded_array)


def resize_cv(matrix, scale_value):
    return cv2.resize(matrix, dsize=(scale_value, scale_value), interpolation=cv2.INTER_NEAREST)


def resize_torch(row, scale_value):  # NON FUNZIONA
    resize = transforms.Resize(size=(scale_value, scale_value))
    return resize(row)


def resize_col(series):
    X = []
    for let in series:
        let = resize_row(let)
        tensor = torch.tensor(let)
        tensor = tensor[None, :]
        X.append(tensor)
    return pd.Series(X)
