#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:35:10 2022

@author: emanuele
"""

import pandas as pd
from os.path import basename
from font_to_matrix import takeKernDict, matrix_kern


def create_all_df(font_otf, kern_path_list, scale_value):
    for path_otf, path_kern in zip(sorted(font_otf), sorted(kern_path_list)):
        df = pd.DataFrame(columns=("font", "X1", "X2", "y"))
        print(basename(path_otf).split(".")[0], len(takeKernDict(path_kern)))
        let1, let2, value = matrix_kern(path_otf, takeKernDict(path_kern), scale_value)
        print("Matrix Kern creata")
        df["font"] = [basename(path_otf).split(".")[0] for i in range(len(let1))]
        print(len(let1), len(let2), len(value))
        df["X1"] = let1
        df["X2"] = let2
        df["y"] = value
        df.to_csv(
            "/ext/home/emanuele/kernfont/progettoKerning/dataframe/{}.csv".format(
                basename(path_otf).split(".")[0]
            ),
            index=False,
        )
        del df
        print("Df salvato in locale")
        print("===================================")


def create_df(path_otf, path_kern, scale_value):
    print(basename(path_otf), len(takeKernDict(path_kern)))
    let1, let2, value = matrix_kern(path_otf, takeKernDict(path_kern), scale_value)
    df = pd.DataFrame(columns=("font", "X1", "X2", "y"))
    df["font"] = [basename(path_otf).split(".")[0] for i in range(len(let1))]
    df["X1"] = let1
    df["X2"] = let2
    df["y"] = value
    return df


def create_complete_df(font_otf, kern_path_list, scale_value):
    df_completed = pd.DataFrame(columns=("font", "X1", "X2", "y"))
    for path_otf, path_kern in zip(sorted(font_otf), sorted(kern_path_list)):
        print(basename(path_otf).split(".")[0], len(takeKernDict(path_kern)))
        let1, let2, value = matrix_kern(path_otf, takeKernDict(path_kern), scale_value)
        print("Matrix Kern creata")
        df_temp = pd.DataFrame(columns=("font", "X1", "X2", "y"))
        df_temp["font"] = [basename(path_otf).split(".")[0] for i in range(len(let1))]
        df_temp["X1"] = let1
        df_temp["X2"] = let2
        df_temp["y"] = value
        df = pd.concat([df_completed, df_temp], ignore_index=True)
        print("============================================================")

    df.to_csv("full_data.csv", index=False)
    print("Dataframe completo salvato in locale")
    return df_completed
