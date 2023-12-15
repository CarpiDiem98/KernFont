import numpy as np
import plistlib
import pandas as pd


def create_kerning_dataframe(otf: str, ufo: str):
    # open plist file
    with open(ufo, "rb") as fp:
        pl = plistlib.load(fp)

    # create list of tuples
    result = []
    for key, dict1 in pl.items():
        for key2, dict2 in dict1.items():
            result.extend([[otf, ufo, key, key2, dict2]])

    # append element in dataframe
    return pd.DataFrame(result, columns=["otf", "ufo", "sx", "dx", "kern_value"])


def create_kerning_dictionary(kern_file: str):
    # open plist file
    with open(kern_file, "rb") as fp:
        pl = plistlib.load(fp)

    # create dictionary using dict comprehension
    return {
        (key, key2): dict2 for key, dict1 in pl.items() for key2, dict2 in dict1.items()
    }


def compact_to_single_matrix(sx, dx, kerning_value):
    # Crea una matrice di zeri
    # shape = (righe, (sx_colonne + dx_colonne - kerning_value))
    if kerning_value < 0:
        matrix = np.zeros(
            (sx.shape[0], (sx.shape[1] + dx.shape[1] - kerning_value)),
            dtype=float,
        )
    else:
        matrix = np.zeros(
            (sx.shape[0], (sx.shape[1] + dx.shape[1] + kerning_value)),
            dtype=float,
        )
    # Copia sx nella parte sinistra della matrice risultante
    matrix[:, : sx.shape[1]] = sx
    # Sovrapponi dx nella parte destra della matrice risultante
    # Somma i pixel sovrapposti
    col = sx.shape[1] - kerning_value
    final_col = col + dx.shape[1]
    matrix[:, col:final_col] += dx
    return matrix
