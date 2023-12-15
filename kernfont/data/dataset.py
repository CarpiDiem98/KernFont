import cv2
import os
import numpy as np
import tensorfont as tf
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset
from kernfont.logger.logger import logger
import torch


class Font(Dataset):
    def __init__(self, annotations: str, transform=None):
        self.df = pd.read_csv(annotations)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __dictionary_to_glyphs_matrix(
        self, otf: str, sx: str, dx: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert two glyphs from an OpenType font file to matrices with sidebearings.

        Args:
            otf (str): Path to the OpenType font file.
            sx (str): Name of the first glyph.
            dx (str): Name of the second glyph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two matrices, one for each glyph.
            Each matrix has the same shape and dtype, and represents the glyph image with sidebearings.
        """
        font = tf.Font(otf)
        sx = font.glyph(sx.replace("public.kern1.", "")).as_matrix().with_sidebearings()
        dx = font.glyph(dx.replace("public.kern2.", "")).as_matrix().with_sidebearings()
        return sx, dx

    def __getitem__(self, index):
        otf, _, sx, dx, kern_value = self.df.iloc[index]
        sx_m, dx_m = self.__dictionary_to_glyphs_matrix(otf, sx, dx)
        if self.transform is not None:
            sx_m = sx_m / 255
            dx_m = dx_m / 255
            sx_m = self.transform(sx_m).to(torch.float32)
            dx_m = self.transform(dx_m).to(torch.float32)
            kern_value = torch.tensor(kern_value, dtype=torch.float32)
        return sx_m, dx_m, kern_value


# TODO: remove this function
def __resize_cv(matrix, scale_value=500):
    return cv2.resize(
        matrix,
        dsize=(scale_value, scale_value),
        interpolation=cv2.INTER_NEAREST,
    )
