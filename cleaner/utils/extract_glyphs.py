from typing import Tuple
import numpy as np
import os
import tensorfont as tf
from cleaner.logger.logger import logger


def dictionary_to_glyphs_matrix(
    otf: str, sx: str, dx: str
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


def process_partition(partition):
    index = []
    for row in partition.iterrows():
        try:
            _, _ = dictionary_to_glyphs_matrix(
                row[1]["otf"], row[1]["sx"], row[1]["dx"]
            )
        except Exception as e:
            index.append(row[0])
    return index
