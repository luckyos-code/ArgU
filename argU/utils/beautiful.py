from utils.reader import FindArgumentIterator


def print_argument_texts(ids, path):
    """Gib zu den gefundenen Argument IDs die passenden Meta-Informationen aus

    Args:
        ids (list): Argument IDs
        path (str): Pfad zur CSV Datei mit den Argumenten
    """
    for argument in FindArgumentIterator(path, ids):
        print((
            f"Length = {len(argument.text)}, "
            f"Text = {' '.join(argument.text_raw.split()[:25])} ..."
        ))
