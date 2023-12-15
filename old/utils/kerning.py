#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:35:26 2022
@author: emanuele
"""

from fontTools.ufoLib import UFOReader
from os.path import dirname
import numpy as np
import plistlib


def getFontKerning(font_path):
    font = UFOReader(font_path)
    kerning = font.readKerning()
    return kerning


def getAllKerning(kern_path_list, letter):
    """
    Parameters
    ----------
    kern_path_list : string
        path dei file di kerning di ogni font.
    letter1 : list
        lettere da kernare posizione sinistra.
    letter2 : list
        lettere da kernare posizione sinistra.

    Returns
    -------
    kern : lis
        lista di tutti i kern che ci interessano.

    """
    kern1 = 'public.kern1.'
    kern2 = 'public.kern2.'
    plus = '2'
    kern = []
    for path in kern_path_list:
        p = dirname(path).split('/')[-1].split('.')[0]
        # print(p)
        with open(path, 'rb') as fp:
            pl = plistlib.load(fp)
        for key, value1 in pl.items():
            decompressed_dict = [[key1, key, value] for key1,
                                 value1 in pl.items() for key, value in value1.items()]
        # i=0
        gg = []
        # vacand = True
        for let1 in letter:
            if let1.isupper() == True:
                for let2 in letter:
                    vacand = True
                    for k in decompressed_dict:
                        if ((let1 == k[0] or let1+plus == k[0] or kern1+let1 == k[0] or kern1+let1+plus == k[0]) and
                                (let2 == k[1] or let2+plus == k[1] or kern2+let2 == k[1] or kern2+let2+plus == k[1])):
                            vacand &= False
                            k3 = k[2]
                            k = [i.replace('public.kern1.', '')
                                 for i in k if isinstance(i, str)]
                            k = [i.replace('public.kern2.', '')
                                 for i in k if isinstance(i, str)]
                            k = [i.replace('2', '')
                                 for i in k if isinstance(i, str)]
                            # k.append(k3)
                            gg.append(k3)
                            break
                    if vacand == True:
                        k = [let1, let2, np.nan]
                        gg.append(np.nan)
        iola = [p] + gg
        kern.append(iola)
    return kern


def getSingleKerning(kern_path_list, let1, let2):
    kern1 = 'public.kern1.'
    kern2 = 'public.kern2.'
    plus = '2'
    kern = []
    for path in kern_path_list:
        p = dirname(path).split('/')[-1].split('.')[0]
        # print(p)
        with open(path, 'rb') as fp:
            pl = plistlib.load(fp)
        for key, value1 in pl.items():
            decompressed_dict = [[key1, key, value] for key1,
                                 value1 in pl.items() for key, value in value1.items()]
        # i=0
        # gg = []
        vacand = True
        for k in decompressed_dict:
            if ((let1 == k[0] or let1+plus == k[0] or kern1+let1 == k[0] or kern1+let1+plus == k[0]) and (let2 == k[1] or let2+plus == k[1] or kern2+let2 == k[1] or kern2+let2+plus == k[1])):
                vacand &= False
                k3 = k[2]
                # k = [i.replace('public.kern1.', '')
                #      for i in k if isinstance(i, str)]
                # k = [i.replace('public.kern2.', '')
                #      for i in k if isinstance(i, str)]
                # k = [i.replace('2', '')
                #      for i in k if isinstance(i, str)]
                # k.append(k3)
                kern.append([p, k3])
        if vacand == True:
            # k = [let1, let2, 0]
            kern.append([p, np.nan])
    return kern  # lista nome_font, valore_kern
