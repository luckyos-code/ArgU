from argU.utils.reader import FindArgumentIterator


def print_argument_texts(ids, path, print_full_texts=False, only_original=True):
    """Gib zu den gefundenen Argument IDs die passenden Meta-Informationen aus

    Args:
        ids (list): Argument IDs
        path (str): Pfad zur CSV Datei mit den Argumenten
        print_full_texts (bool): Soll alles angezeigt werden
            oder nur ein Bruchteil?
    """
    for argument in FindArgumentIterator(path, ids):
        text_raw = argument.text_raw
        text_sentiment = argument.text_sentiment
        text_machine = argument.text_machine

        print(f"{argument.id}\n")
        if print_full_texts:
            print(f"\t1. Original -> {text_raw}\n")
            if not only_original:
                print(f"\t2. Sentiment -> {text_sentiment}\n")
                print(f"\t3. Embedding + BM25 -> {text_machine}")
        else:
            print(f"\t1. Original -> {text_raw[:40]}\n")
            if not only_original:
                print(f"\t2. Sentiment -> {text_sentiment[:40]}\n")
                print(f"\t3. Embedding + BM25 -> {text_machine[:40]}")
        print('\n', '=' * 80, '\n')


def print_embedding_examples(model, words):
    for word in words:
        print()
        print(f"<b>{word}</b>")
        most_similar_words = model.wv.most_similar(word, topn=10)
        for msw in most_similar_words:
            print(f"* {msw[0]}, Score: {msw[1]:.4f}")
