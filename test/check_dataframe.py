from library.conversion_to_array import array_conversion
import glob
import logging
import os
import pandas as pd
import chardet

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


folder = '/home/emanuele/Workspace/kernfont/dataframe/'
csv_paths = sorted(glob.glob(folder + "/*.csv"))
logger.info('Numero di path: %s', len(csv_paths))

flag = False

for path in csv_paths:
    logger.info('{}{}'.format(os.path.basename(path), flag))
    if flag:
        if not isinstance(array_conversion(path), pd.DataFrame):
            logger.error('rotto =/')
    elif os.path.basename(path) == 'Xanti-Typewriter-RegularItalic.csv':
        flag = True
        if not isinstance(array_conversion(path), pd.DataFrame):
            logger.error('rotto =/')
    else:
        continue