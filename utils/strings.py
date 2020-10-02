

def ensure_file_extension(path, extension):

    if extension[0] == '.':
        extension = extension[1:]

    # e.g. extension = mp3
    split_path = path.split('.')

    if not (len(split_path) == 1 or len(split_path) == 2):
        raise RuntimeError

    if len(split_path) == 1:
        path = split_path[0] + '.' + extension
        return path
    elif len(split_path) > 1 and split_path[-1] == extension:
        return path
    else:
        raise ValueError

