import os
import extractor
import defcon
from kernfont.logger.logger import logger


def convert_otf_to_ufo(font, dest):
    """Converter OTF font file into UFO font file

    Args:
        font (str): path of OTF file
        dest (str): path of destination folder of the UFO file
    """
    try:
        ufo = defcon.Font()
        extractor.extractUFO(font, ufo)
        _, tail = os.path.split(os.path.normpath(font))
        destination = os.path.join(dest, tail)
        ufo.save(destination)
        logger.info(destination)
    except Exception as e:
        logger.error(e)
