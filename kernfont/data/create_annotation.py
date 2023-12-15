import pandas as pd
from path import generate_set, path_kerning_file, path_otf_font
import os
from tqdm import tqdm
from utils import create_kerning_dataframe
import timeit


if __name__ == "__main__":
    folder = "/home/emanuele/Workspace/kernfont/Dataset"
    otf = path_otf_font(os.path.join(folder, "OTF"))
    ufo = path_kerning_file(os.path.join(folder, "UFO"))
    data_start = timeit.default_timer()
    dataset = generate_set(otf, ufo)
    data_end = timeit.default_timer()
    print(f"Time to generate dataset: {data_end - data_start}")

    df = pd.DataFrame(columns=["otf", "ufo", "sx", "dx", "kern_value"])
    for otf, ufo in tqdm(dataset):
        temp_df = create_kerning_dataframe(otf, ufo)
        df = pd.concat([df, temp_df], ignore_index=True)

    print(df.shape, df.head())
    df.to_csv("dataset.csv", index=False)
