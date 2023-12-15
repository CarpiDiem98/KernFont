#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:47:12 2022

@author: emanuele
"""

import time
import os
from progettoKerning.library.font_to_matrix import getAllLetterFont, matrix_kern, takeKernDict
import plistlib
import tensorfont as tf
import pandas as pd
from path_font import path_kerning_file, path_otf_font
from os.path import dirname
start_time = time.time()
print(time.asctime(time.localtime(start_time)))


# source_OTF = '../Dataset/train/OTF'
# source_UFO = '../Dataset/train/UFO'

# font_otf = path_otf_font(source_OTF)
# kern_path_list = path_kerning_file(source_UFO)


source = '../Dataset/train/OTF/Arkit-Regular.otf'
kern_file = '../Dataset/train/UFO/Arkit-Regular.ufo/kerning.plist'

decompressed_dict = takeKernDict(kern_file)
list_matrix = matrix_kern(source, decompressed_dict, 32)


# def decompress_list_matrix(list_matrix):
#     X = []
#     y = []
#     for i in list_matrix:
#         x1, x2, value = i
#         y.append(value)
#         X.append([x1, x2])
#     return X, y


# X, y = decompress_list_matrix(list_matrix)


source_OTF = '../Dataset/train/OTF'
source_UFO = '../Dataset/train/UFO'

font_otf = path_otf_font(source_OTF)
kern_path_list = path_kerning_file(source_UFO)


def dataframeMatrix(scale_value):
    """
    Funzione che restituisce tutto il dataset di valori di kern con le rispettive
    immagini in forma di matrice in coppia. 

    Parameters
    ----------
    scale_value : numeric
        Valore di rescaling dell'altezza delle matrici.
        Esempio -> h della font = 1490x500 -> dopo il rescaling = 100x33.5

    Returns
    -------
    df : Dataframe
        Dataframe due colonne:
            font -> nome font
            kern -> lista di liste contenti 
                    [ [ [let1], [let2], kern_value ], [], [] ,[] ].
    """
    i = 0
    font_name = []
    matrix1 = []
    matrix2 = []
    y = []
    for path_otf, path_kern in zip(sorted(font_otf), sorted(kern_path_list)):
        p = (dirname(path_kern).split('/')[-1].split('.')[0])
        # print(p)
        font_name.append(p)
        decompressed_dict = takeKernDict(path_kern)
        # print(len(decompressed_dict))
        let1, let2, value = matrix_kern(
            path_otf, decompressed_dict, scale_value)
        
        y.append(value)
        if i == 1:
            print(i)
            break
        i += 1
    df = pd.DataFrame(columns=('font', 'X1', 'X2', 'y'))
    df['font'] = font_name
    df['X1'] = let1
    df['X2'] = let2
    df['y'] = y
    df = df.sort_values(by=['font'])
    return df


df = dataframeMatrix(32)

# lol = path_kerning_file('../Dataset/train/UFO/')
# for i in lol:
#     p = dirname(i).split('/')[-1].split('.')[0]
#     print(p)
#     decompressed_dict = takeKernDict(i)

# dict_no_number = []
# for i in decompressed_dict:
#     k1, k2, value = i
#     if k1 not in number:
#         dict_no_number.append([k1, k2, value])

# import re
# for i in number:
#     r = re.compile('.*'+i)
#     newList = list(filter(r.match, decompressed_dict))


# strings_of_text = ['data0', 'data23', 'data2', 'data55', 'data_mismatch', 'green']
# strings_to_keep = []
# expression_to_use = r'zero*'

# for string in strings_of_text:
#     # If the string is data#
#     if (re.match(expression_to_use, string)):
#         strings_to_keep.append(string)

# print(strings_to_keep)


###############################################################################
os.system('play -nq -t alsa synth {} sine {}'.format(1, 440))  # secondi, Hz
print("--- %s secondi ---" % ((time.time() - start_time)))
