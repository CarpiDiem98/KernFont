# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorfont as tf
import string
import plistlib
import os


def glyphRenderToMatrix(let):
    return [np.array(row) for row in let]


def getGlyph(path_font):
    return tf.Font(path_font).glyph('o').as_matrix().with_sidebearings()


def spazio_vuoto(path_font):
    m, n = np.shape(getGlyph(path_font))
    return m*n - np.count_nonzero(getGlyph(path_font))


def matrix_kern(source, decompressed_dict, scale_value):
    font = tf.Font(source)
    new_let1 = []
    new_let2 = []
    value = []
    for k1, k2, v in decompressed_dict:
        try:
            let1 = glyphRenderToMatrix(np.matrix(font.glyph(k1.replace(
                'public.kern1.', '')).as_matrix().with_sidebearings().scale_to_height(scale_value)))
            let2 = glyphRenderToMatrix(np.matrix(font.glyph(k2.replace(
                'public.kern2.', '')).as_matrix().with_sidebearings().scale_to_height(scale_value)))
            val = v
            new_let1.append(let1)
            new_let2.append(let2)
            value.append(val)
        except Exception:
            pass
    return new_let1, new_let2, value


def getAllLetterFont(path_font):
    list_matrix = []
    alphabet = list(string.ascii_letters)
    font = tf.Font(path_font)
    for letter in alphabet:
        let = font.glyph(letter)
        letter_matrix = let.as_matrix().with_sidebearings().scale_to_height(100)
        list_matrix.append([letter, letter_matrix])
    return list_matrix


def takeKernDict(kern_file):
    """Generate a list of kern_dict from a kern file

    Args:
        kern_file (_type_): _description_

    Returns:
        dict: dict of tuple of kerning
    """
    with open(kern_file, 'rb') as fp:
        pl = plistlib.load(fp)
    for key, value1 in pl.items():
        return [[key1, key, value] for key1, value1 in pl.items() for key, value in value1.items()]


def create_kerning_dictionary(kern_file):
    # open plist file
    with open(kern_file, 'rb') as fp:
        pl = plistlib.load(fp)

    dictionary = {}
    # iterate through all kerning entries
    # then add them to the dictionary
    for key, dict1 in pl.items():
        for key2, dict2 in dict1.items():
            dictionary[(key, key2)] = dict2

    return dictionary


def dictionary_to_glyphs_matrix(path_font: str, tupla: tuple):
    font = tf.Font(path_font)
    sx = font.glyph(tupla[0][0].replace('public.kern1.', '')
                    ).as_matrix().with_sidebearings()
    dx = font.glyph(tupla[0][1].replace('public.kern2.', '')
                    ).as_matrix().with_sidebearings()
    return sx, dx, tupla[1]


def compact_to_single_matrix(sx, dx, kerning_value):
    # Crea una matrice di zeri
    # shape = (righe, (sx_colonne + dx_colonne - kerning_value))
    matrice_risultante = np.zeros(
        (sx.shape[0], (sx.shape[1] + dx.shape[1] - kerning_value)), dtype=float)

    # Copia sx nella parte sinistra della matrice risultante
    matrice_risultante[:, :sx.shape[1]] = sx

    # Sovrapponi dx nella parte destra della matrice risultante
    # Somma i pixel sovrapposti
    colonna_iniziale = sx.shape[1] - kerning_value
    colonna_finale = colonna_iniziale + dx.shape[1]
    matrice_risultante[:, colonna_iniziale:colonna_finale] += dx
    return matrice_risultante
