import pandas as pd
import os
import re
import numpy as np
from os.path import basename, dirname
import time
import datetime

from ..font_to_matrix import takeKernDict, matrix_kern
from ..kerning import getAllKerning, getSingleKerning
from ..path_font import path_otf_font, path_ufo_list


from squadratura import squadratura
from peso import peso
from dimension import box



def df_dataset_matrix(directory):
    df_append = pd.DataFrame()
    pattern = r'\[(.*?)\]'

    pattern_list = sorted(os.listdir(directory))

    for path in pattern_list:
        start_time = time.time()
        print(datetime.datetime.now())
        df = pd.read_csv(directory + path)
        df1 = df.copy()
        array_nome = []
        matrix1 = []
        matrix2 = []
        nome = path.split('.')[0]
        print('Inizio creazione matrice:', nome)

        for row1, row2 in zip(df.X1, df.X2):
            array_nome.append(nome)
            row1 = row1.replace('\n', '').replace('  ', ' ')[1:-1]
            row2 = row2.replace('\n', '').replace('  ', ' ')[1:-1]
            row1 = re.findall(pattern, row1)
            row2 = re.findall(pattern, row2)
            print(row1)
            new_row1 = [[] for _ in range(len(row1))]
            new_row2 = [[] for _ in range(len(row2))]

            for index, row in enumerate(row1):
                for value in row.split(' '):
                    new_row1[index].append(float(value))

            for index, row in enumerate(row2):
                for value in row.split(' '):
                    new_row2[index].append(float(value))

            matrix1.append(new_row1)
            matrix2.append(new_row2)
        df1.font = array_nome
        df1.X1 = np.matrix(matrix1)
        df1.X2 = np.matrix(matrix2)
        delta = round((time.time() - start_time), 2)
        print('Dataframe', nome, 'creato con successo')
        print('%s secondi' % delta)
        df_append = df_append.append(df1, ignore_index=True)

    print('Dataframe finale creato con successo')
    return df_append


def dataframeMatrix(font_otf, kern_path_list, scale_value):
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
    font_name = []
    matrix1 = []
    matrix2 = []
    y = []
    i = 0
    for path_otf, path_kern in zip(sorted(font_otf), sorted(kern_path_list)):
        p = (dirname(path_kern).split('/')[-1].split('.')[0])
        print(p)
        font_name.append(p)
        decompressed_dict = takeKernDict(path_kern)
        print(len(decompressed_dict))
        let1, let2, value = matrix_kern(
            path_otf, decompressed_dict, scale_value)
        matrix1.append(let1)
        matrix2.append(let2)
        y.append(value)
    df = pd.DataFrame(columns=('font', 'X1', 'X2', 'y'))
    df['font'] = font_name
    df['X1'] = matrix1
    df['X2'] = matrix2
    df['y'] = y
    df = df.sort_values(by=['font'])
    return df


def dataframeSquadratura(source_UFO):
    font_name = []
    font_ufo = path_ufo_list(source_UFO)
    squad = []
    for path in font_ufo:
        # print(path)
        font_name.append(basename(path).split('.')[0])
        squad.append(squadratura(path))
    df_squad = pd.DataFrame(columns=('font', 'squadratura'))
    df_squad['font'] = font_name
    df_squad['squadratura'] = squad
    df_squad = df_squad.sort_values(by=['font'])

    for index, row in df_squad.iterrows():
        if df_squad.loc[index, 'squadratura'] > 1:
            df_squad.loc[index, 'squadratura'] = 1
    return df_squad


def dataframePesi(source_OTF):
    font_name = []
    font_otf = path_otf_font(source_OTF)
    pesi = []
    for path in font_otf:
        # print(path)
        font_name.append(basename(path).split('.')[0])
        pesi.append(peso(path))
    df_pesi = pd.DataFrame(columns=('font', 'peso'))
    df_pesi['font'] = font_name
    df_pesi['peso'] = pesi
    df_pesi = df_pesi.sort_values(by=['font'])
    return df_pesi


def dataframeBox(source_OTF, let1, let2):
    font_name = []
    font_otf = path_otf_font(source_OTF)
    let1_box_l = []
    let2_box_l = []
    for path in font_otf:
        font_name.append(basename(path).split('.')[0])
        let1_box, let2_box = box(path, let1, let2)
        let1_box_l.append(let1_box[1])
        let2_box_l.append(let2_box[1])
    df_box = pd.DataFrame(columns=('font', let1+'_box', let2+'_box'))
    df_box['font'] = font_name
    df_box[let1+'_box'] = let1_box_l
    df_box[let2+'_box'] = let2_box_l
    df_box = df_box.sort_values(by=['font'])
    return df_box


def dataframeSingleKern(kern_path_list, let1, let2):
    kern = getSingleKerning(kern_path_list, let1, let2)
    df_SingKern = pd.DataFrame(data=kern, columns=('font', 'kern'))
    df_SingKern = df_SingKern.sort_values(by=['font'])
    return df_SingKern


def dataframeAllKern(kern_path_list, letter):
    kerning = getAllKerning(kern_path_list, letter)
    columns = ['font']
    match = [(let1, let2)
             for let1 in letter if let1.islower() != True for let2 in letter]
    for i in match:
        columns.append(i)
    df_AllKern = pd.DataFrame(data=kerning, columns=(columns))
    df_AllKern = df_AllKern.sort_values(by=['font'])
    return df_AllKern


def concatSingleKernDataFrame(kern_path_list, let1, let2):
    df_pesi = dataframePesi()
    df_squad = dataframeSquadratura()
    df_box = dataframeBox(let1, let2)
    df_SingKern = dataframeSingleKern(kern_path_list, let1, let2)
    df_SK = pd.merge(df_pesi, df_squad).merge(df_box).merge(df_SingKern)
    return df_SK


def concatDataFrame(kern_path_list, letter):
    df_pesi = dataframePesi()
    df_squad = dataframeSquadratura()
    df_kerning = dataframeAllKern(kern_path_list, letter)
    df_All = pd.merge(df_pesi, df_squad).merge(df_kerning)
    return df_All
