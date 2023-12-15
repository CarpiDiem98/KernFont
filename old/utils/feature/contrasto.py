#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:20:20 2022
@author: emanuele
"""


import numpy as np
import tensorfont as tf
from path_font import path_otf_font
import pandas as pd
from os.path import basename


def glyph(path_font):
    font = tf.Font(path_font)
    m = font.glyph('m')
    m_matrix = m.as_matrix().with_sidebearings().crop_descender()
    return m_matrix


def contrasto(path_font):
    m, n = np.shape(glyph(path_font))
    spazio_nero = np.count_nonzero(glyph(path_font))
    spazio_bianco = m*n - spazio_nero
    contrasto = spazio_nero / spazio_bianco
    return contrasto


def getDFContrasto(source_otf):
    path_list = path_otf_font(source_otf)
    contrasti = []
    font_name = []
    for path in path_list:
        font_name.append(basename(path).split('.')[0])
        contrasti.append(contrasto(path))

    df_contrasto = pd.DataFrame(columns=('font', 'contrasto'))
    df_contrasto['font'] = font_name
    df_contrasto['contrasto'] = contrasti
    df_contrasto = df_contrasto.sort_values(by=['font'])
    return df_contrasto
