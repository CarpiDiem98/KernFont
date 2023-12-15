from cleaner.logger.logger import logger
from cleaner.utils.extract_glyphs import dictionary_to_glyphs_matrix
import dask.dataframe as dd
import os


if __name__ == "__main__":
    logger.info("Starting cleaner...")
    df = dd.read_csv("annotations.csv")

    index = []

    for row in df.itertuples():
        try:
            dictionary_to_glyphs_matrix(row["otf"], row["sx"], row["dx"])
        except Exception as e:
            index.append(row[0])
            logger.error(
                f"Error: {os.path.basename(row['otf'])} -- {row['sx']} -- {row['dx']} \n {e}"
            )
    

    new_df = df.drop(index=index)
    new_df.to_csv("annotations_cleaned.csv", index=False)
    logger.info("Done!")
