#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:40:13 2022

@author: emanuele
"""

import numpy as np
import tensorfont as tf


def get_x_height(path_font):
    font = tf.Font(path_font)
    x_height = int(font.x_height)
    return x_height


def average_l(path_font):
    font = tf.Font(path_font)
    l = font.glyph('l')
    l_matrix = l.as_matrix().mask_to_x_height().crop_descender()
    h, l = l_matrix.shape
    altezza = h // 2
    row = l_matrix[altezza, :].tolist()
    row_no_zero = np.count_nonzero(row)
    return row_no_zero


def peso(path_font):
    x_height = get_x_height(path_font)
    distance = average_l(path_font)
    peso = float(distance / x_height)
    return peso
