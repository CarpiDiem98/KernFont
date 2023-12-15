#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:38:35 2022

@author: emanuele
"""

from fontTools.ufoLib import UFOReader
import xml.etree.cElementTree as et
import numpy as np
import pandas as pd
import re


def squadratura(font_path):
    m1, m = take_4_row_dataframe(font_path)
    squad = (m1 / m)
    return squad


def getO(font_path):
    font = UFOReader(font_path)
    glyphs = font.getGlyphSet()
    o = glyphs.getGLIF('o')
    return o


def getPointO(font_path):
    o = getO(font_path)
    xml = et.fromstring(o)
    esterni = []
    interni = []
    for idx, contour in enumerate(xml.iter('contour')):
        if idx == 0:
            for child in contour:
                esterni.append(et.tostring(child, encoding='unicode'))
        if idx == 1:
            for child in contour:
                interni.append(et.tostring(child, encoding='unicode'))
    return esterni, interni


def getDataFramePoint(font_path):
    e, i = getPointO(font_path)
    pattern_curve_smooth = r'x="\d+" y="\d+" type="curve" smooth="yes"'
    pattern_curve = r'x="\d+" y="\d+" type="curve"'
    pattern_line = r'x="\d+" y="\d+" type="line"'
    pattern_line_smooth = r'x="\d+" y="\d+" type="line" smooth="yes"'
    pattern_no_type = r'x="\d+" y="\d+"'

    value = []
    for cord in i:
        curve_smooth = re.search(pattern_curve_smooth, cord)
        curve = re.search(pattern_curve, cord)
        line = re.search(pattern_line, cord)
        line_smooth = re.search(pattern_line_smooth, cord)
        no_type = re.search(pattern_no_type, cord)
        if curve_smooth:
            # Sprint(curve_smooth.group())
            value.append(curve_smooth.group().split())
        elif curve:
            # print(curve.group())
            value.append(curve.group().split() + [''])
        elif line_smooth:
            # print(line_smooth.group())
            value.append(line_smooth.group().split())
        elif line:
            # print(line.group())
            value.append(line.group().split() + [''])
        elif no_type:
            # print(no_type.group())
            value.append(no_type.group().split() + ['', ''])
    """            
    for cord in e:
        curve_smooth = re.search(pattern_curve_smooth, cord)
        curve = re.search(pattern_curve, cord)
        line = re.search(pattern_line, cord)
        line_smooth = re.search(pattern_line_smooth, cord)
        no_type = re.search(pattern_no_type, cord)
        if curve_smooth:
            # Sprint(curve_smooth.group())
            value.append(curve_smooth.group().split())
        elif curve:
            # print(curve.group())
            value.append(curve.group().split() + [''])
        elif line_smooth:
            # print(line_smooth.group())
            value.append(line_smooth.group().split())
        elif line:
            # print(line.group())
            value.append(line.group().split() + [''])
        elif no_type:
            # print(no_type.group())
            value.append(no_type.group().split() + ['', ''])
        else:
            print('Nessun valore trovato')
        """

    df = pd.DataFrame(value, columns=('x', 'y', 'type', 'smooth'))
    listaX = []
    listaY = []

    for i in value:
        for j in i:
            # print(j)
            if j.startswith('x='):
                # print(j)
                m = re.findall('\d+', j)
                listaX.append(m)
            elif j.startswith('y='):
                # print(j)
                m = re.findall('\d+', j)
                listaY.append(m)
    listaX = np.array(listaX, dtype='int')
    listaY = np.array(listaY, dtype='int')

    df['x'] = listaX
    df['y'] = listaY
    df['type'] = df['type'].replace(['type="line"'], 'line')
    df['type'] = df['type'].replace(['type="curve"'], 'curve')
    df['smooth'] = df['smooth'].replace(['smooth="yes"'], True)
    # print(df)
    return df


def manipolatori(x, y):
    # print('x', x)
    # print('y', y)
    # manipolatore orizzontale
    if (y[0] - y[1]) in range(25):
        m1 = abs(x[0] - x[1])
        m = abs(x[0] - x[2])
    else:
        m1 = abs(y[0] - y[1])
        m = abs(y[0] - y[2])
    return m1, m


def take_4_row_dataframe(font_path):
    df = getDataFramePoint(font_path)
    first_index = df.index[df['type'] == 'curve'][0]
    df = pd.DataFrame((df[first_index:first_index+4]))
    x = df['x'].values
    y = df['y'].values
    m1, m = manipolatori(x, y)
    # print(m1/m)
    # print(m1,m)
    if m1/m < 0.5:
        df = getDataFramePoint(font_path)
        first_index = df.index[df['type'] == 'curve'][1]
        # print(first_index)
        df = pd.DataFrame((df[first_index:first_index+4]))
        # print(df)
        x = df['x'].values
        y = df['y'].values
        m1, m = manipolatori(x, y)
        # print(m1 / m)
    elif m1/m < 0.5:
        df = getDataFramePoint(font_path)
        first_index = df.index[df['type'] == 'curve'][-1]
        df = pd.DataFrame((df[first_index:first_index+4]))
        # print(df)
        x = df['x'].values
        y = df['y'].values
        m1, m = manipolatori(x, y)
        # print(m1,m)
    # else :
    #     m1 = 1
    #     m = 1
    # print(m1, m)
    return m1, m
