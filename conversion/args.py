import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Folder for module converter")
    parser.add_argument(
        "--init_folder",
        type=str,
        help="folder OTF file",
        default="./Dataset/OTF",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        help="folder destination UFO file",
        default="./Dataset/UFO",
    )

    return parser.parse_args()
