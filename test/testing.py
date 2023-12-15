
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:26:59 2022

@author: emanuele
"""

from library.path_font import path_kerning_file
from library.dataframe import concatDataFrame, dataframeAllKern, concatSingleKernDataFrame, dataframeKernPlus
from library.kerning import getAllKerning, getSingleKerning
import os
import time
start_time = time.time()

source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)
letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']
# df = concatDataFrame(kern_path_list, letter)
# df_sin = concatSingleKernDataFrame(kern_path_list, 'A', 'T')

df = dataframeAllKern(kern_path_list, letter)

# concatSingleKernDataFrame(kern_path_list, let1, let2)
# df_kerning = dataframeAllKern(kern_path_list, letter)
# kerning = getAllKerning(kern_path_list, letter)
# let1 = 'A'
# let2 = 'T'
# df = concatSingleKernDataFrame(let1, let2)
# print('GG')

# df = pd.DataFrame(data=kern)
# lista = []
# value = []

# for i in kern:
#     print(i[0], len(i[1]))
#     lista.append([i[0], len(i[1])])
#     value.append(len(i[1]))

duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

print("--- %s minuti ---" % ( (time.time() - start_time)/60 )) 
# sono circa 5 minuti ad esecuzione per la creazione della componente kerning TUTTTIIIII
###############################################################################
# frequency = {}

# # iterating over the list
# for item in value:
#    # checking the element in dictionary
#    if item in frequency:
#       # incrementing the counr
#       frequency[item] += 1
#    else:
#       # initializing the count
#       frequency[item] = 1

# # printing the frequency
# print(frequency)






