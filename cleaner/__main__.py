from cleaner.logger.logger import logger
from cleaner.utils.extract_glyphs import process_partition
import dask.dataframe as dd

if __name__ == "__main__":
    logger.info("Starting cleaner...")
    df = dd.read_csv("annotations.csv")
    result = df.map_partitions(process_partition, meta=("index", "int")).compute()
    logger.info("Removing rows...")
    df = df.drop(result)
    logger.info("Saving results...")
    result.to_csv("annotations_clean.csv", index=False)
    logger.info("Done!")
