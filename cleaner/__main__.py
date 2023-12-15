from cleaner.logger.logger import logger
from cleaner.utils.extract_glyphs import dictionary_to_glyphs_matrix
import pandas as pd
import os
from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=4, split_factor=2, disable_pr_bar=True)


def process_row(row):
    index = []
    try:
        dictionary_to_glyphs_matrix(row["otf"], row["sx"], row["dx"])
    except Exception as e:
        index.append(row[0])
        logger.error(
            f"Error: {os.path.basename(row['otf'])} -- {row['sx']} -- {row['dx']} \n {e}"
        )
    return index


if __name__ == "__main__":
    logger.info("Starting cleaner...")
    df = pd.read_csv("annotations.csv")
    df_parallel = ParallelPandas(df)
    
    indices = df_parallel.map_partitions(process_row).compute()
    index = [idx for sublist in indices for idx in sublist]  # Flatten the list
    
    new_df = df.drop(index=index)
    new_df.to_csv("annotations_cleaned.csv", index=False)
    logger.info("Done!")
