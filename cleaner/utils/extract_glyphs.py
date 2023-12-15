from typing import Tuple
import numpy as np
import tensorfont as tf


def dictionary_to_glyphs_matrix(otf: str, sx: str, dx: str) -> Tuple[np.ndarray, np.ndarray]:
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
