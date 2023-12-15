#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dic 02 09:39:20 2022
@author: emanuele
"""

import tensorfont as tf

def box(path_font, let1, let2):
    font = tf.Font(path_font)
    let1 = font.glyph(let1)
    let2 = font.glyph(let2)
    matrix1 = let1.as_matrix()
    dim1 = matrix1.shape
    matrix2 = let2.as_matrix()
    dim2 = matrix2.shape
    return dim1, dim2
