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
            print((
                f"Length = {len(argument.text)}, "
                f"Text = {' '.join(argument.text_raw.split())}\n"
            ))
    else:
        for argument in FindArgumentIterator(path, ids):
            print((
                f"Length = {len(argument.text)}, "
                f"Text = {' '.join(argument.text_raw.split()[:25])} ...\n"
            ))


# def print_debate_titles(ids, path):
#     """Gib zu den gefundenen Debatten IDs die passenden Titel aus

#     Args:
#         ids (list): Debatten IDs
#         path (str): Pfad zur CSV Datei mit den Debatten
#     """
#     for debate in FindDebateIterator(path, ids):
#         print((
#             f"Length = {len(debate.text)}, "
#             f"Text = {' '.join(debate.text[:25])} ..."
#         ))


def print_embedding_examples(model, words):
    for word in words:
        print()
        print(f"<b>{word}</b>")
        most_similar_words = model.wv.most_similar(word, topn=10)
        for msw in most_similar_words:
            print(f"* {msw[0]}, Score: {msw[1]:.4f}")
