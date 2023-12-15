from dataframe import create_all_df
from path_font import path_kerning_file, path_otf_font

font_oft = path_otf_font('Dataset/OTF')
font_ufo = path_kerning_file('Dataset/UFO')
scale_value = 32

create_all_df(font_oft, font_ufo, scale_value)
