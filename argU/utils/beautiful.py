from utils.reader import FindArgumentIterator


def print_argument_texts(ids, path, print_all=False):
    """Gib zu den gefundenen Argument IDs die passenden Meta-Informationen aus

    Args:
        ids (list): Argument IDs
        path (str): Pfad zur CSV Datei mit den Argumenten
        print_all (bool): Soll alles angezeigt werden?
    """
    if print_all:
        for argument in FindArgumentIterator(path, ids):
            print(f"{argument.id}\n")
            print(f"\t1. Raw -> {argument.text_raw}\n")
            print(f"\t2. Clean -> {argument.text_nl}\n")
            print(f"\t3. Model -> {argument.text_machine}")
            print('\n', '=' * 40, '\n')
    else:
        for argument in FindArgumentIterator(path, ids):
            print((
                f"Length = {len(argument.text_nl)}, "
                f"Text = {' '.join(argument.text_raw.split()[:25])} ...\n"
            ))

def print_embedding_examples(model, words):
    for word in words:
        print()
        print(f"<b>{word}</b>")
        most_similar_words = model.wv.most_similar(word, topn=10)
        for msw in most_similar_words:
            print(f"* {msw[0]}, Score: {msw[1]:.4f}")
