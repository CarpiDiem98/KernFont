from conversion.args import init_parser
from conversion.otf_to_ufo import convert_otf_to_ufo

if __name__ == "__main__":
    args = init_parser()
    convert_otf_to_ufo(args.init_folder, args.out_folder)
