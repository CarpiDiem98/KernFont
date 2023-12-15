import os


def path_otf_font(folder):
    return sorted(
        [
            os.path.join(dirpath, name)
            for dirpath, dirnames, filenames in os.walk(folder, topdown=False)
            for name in filenames
        ]
    )


def path_kerning_file(folder):
    return sorted(
        [
            os.path.join(dirpath, name)
            for dirpath, dirnames, filenames in os.walk(folder, topdown=False)
            for name in filenames
            if name.endswith("kerning.plist")
        ]
    )


def generate_set(otf, ufo):
    file_set = set()
    for otf_file in otf:
        otf_file_name = os.path.basename(otf_file).split(".")[0]
        for ufo_file in ufo:
            ufo_file_name = os.path.basename(ufo_file.split("/")[-2].split(".")[0])
            if otf_file_name == ufo_file_name:
                file_set.add((otf_file, ufo_file))
                break
    return file_set
