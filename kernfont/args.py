import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--annotations",
        type=str,
        help="csv path with annotations",
        default="annotations.csv",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        help="yaml path with parameters",
        default="parameters.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run experiment",
        default="cpu",
    )

    return parser.parse_args()
