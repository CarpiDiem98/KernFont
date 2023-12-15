import os


def path_ufo_font(source):
    return sorted([dirpath for dirpath, dirnames, filenames in os.walk(source, topdown=False) if dirpath.endswith('.ufo')])


def path_otf_font(source):
    return sorted([os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk(source, topdown=False) for name in filenames])


def path_kerning_file(source):
    return sorted([os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk(source, topdown=False) for name in filenames if name.endswith('kerning.plist')])
