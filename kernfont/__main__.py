from kernfont.args import init_parser
from kernfont.experiment import run_experiment

if __name__ == "__main__":
    args = init_parser()
    run_experiment(args)
