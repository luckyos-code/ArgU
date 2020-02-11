import csv
import os
import sys
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize
import rootpath
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing import tools
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)


def read_csv(path, max_rows=-1, delimiter=','):
    """Lies CSV zeilenweise ein

    Arguments:
        path (str): Pfad zur CSV-Datei
        max_rows (int, default=-1): Maximale Anzahl an Zeilen.
            Beim Default-Wert wird die ganze Datei eingelesen.

    Yields:
        row (list): Zeile der CSV-Datei
    """

    with open(path, 'r', newline='', encoding='utf-8') as f_in:
        csv_reader = csv.reader(f_in, delimiter=delimiter)
        next(csv_reader)

        for i, row in enumerate(csv_reader):
            if i == max_rows:
                break
            yield row


def read_csv_header(path, delimiter=','):
    """Lies CSV Header ein

    Arguments:
        path (str): Pfad zur CSV-Datei

    Returns:
        header (list): CSV-Header
    """

    with open(path, 'r', newline='', encoding='utf-8') as f_in:
        csv_reader = csv.reader(f_in, delimiter=delimiter)
        return next(csv_reader)


class Argument:
    """Argument, das aus einer Zeile der CSV generiert wird"""

    def __init__(self, row):
        self.id = row[9]
        self.debate_id = row[2]
        self.text_raw = row[0]
        self.stance = row[1]
        self.text_nl = tools.natural_language_clean(self.text_raw)
        # self.previous_argument = row[3]
        # self.next_argument = row[8]

    @property
    def text_machine(self):
        """Text for CBOW, BM25"""
        return tools.machine_model_clean(self.text_nl)

    @property
    def text_sentiment(self):
        """Text für die Sentiment Analyse"""
        return tools.sentiment_clean(self.text_raw)

    @staticmethod
    def to_vec(split_text, model, vector_size):
        """Erstelle einen Vektor aus dem Dokument

        Returns:
            (numpy.array or None, list or None): Embedding und unbekannte Wörter.
                Ist keines der Wörter bekannt, wird (None, None) zurückgegeben.
        """

        emb_matrix = np.zeros(
            (len(split_text), vector_size)
        )

        unk_word_count = 0
        unk_words = []
        for i, word in enumerate(split_text):
            try:
                emb_vec = model.wv[word]
                emb_matrix[i] = emb_vec / LA.norm(emb_vec)
            except Exception as e:
                unk_word_count += 1
                unk_words.append(word)

        if emb_matrix.shape[0] == 0:
            return (None, None)

        vec = np.sum(emb_matrix, axis=0) / (emb_matrix.shape[0])
        return (vec, unk_words)


class ArgumentTextIterator:
    """Iterator für die Texte alle Argumente"""

    def __init__(self, args_me_path, max_args=-1, split=True):
        self.args_me_path = args_me_path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.args_me_path, self.max_args):
            if split:
                yield argument.text.split()
            else:
                yield argument.text


class ArgumentCbowIterator:
    """Iterator für bestimmte Formatierung, umd as CBOW Modell
        zu trainieren"""

    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in tqdm(read_arguments(self.path, self.max_args)):
            yield model_text(argument.text).split()


class ArgumentIterator:
    """Iterator für alle Argument Objekte"""

    def __init__(self, max_args=-1):
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(setup.ARGS_ME_CSV_PATH, self.max_args):
            yield argument


class TrainArgsIterator:
    def __init__(self, only_texts=False, max_args=-1):
        self.only_texts = only_texts
        self.max_args = max_args

    def __iter__(self):
        with open(
            setup.TRAIN_ARGS_PATH, 'r', newline='', encoding='utf-8'
        ) as f_in:
            reader = csv.reader(f_in, **setup.TRAIN_ARGS_CONFIG)

            for i, (arg_id, arg_text) in enumerate(reader):

                arg_text = arg_text.strip().split()
                if self.only_texts:
                    yield arg_text
                else:
                    yield (arg_id, arg_text)

                if (i + 1) == self.max_args:
                    break


def read_arguments(path, max_args=-1):
    """Generator um alle CSV direkt in Argument-Objekte zu konvertieren

    Args:
        path (str): Pfad zur CSV-Datei
        max_args (int, default=-1): Maximale Anzahl verschiedener Argumente.
            Bei -1 werden alle Argumente eingelesen.

    Yields:
        Argument
    """

    for row in read_csv(path, max_rows=max_args):
        yield Argument(row)


class FindArgumentIterator:
    """Suche gegebene Argumente anhand vorgegebener IDs

    Args:
        path (str): Pfad zur CSV Datei
        ids (list): Liste von IDs

    Yield:
        Argument
    """

    def __init__(self, ids, raw_texts_only=False):
        self.ids = set(ids)
        self.raw_texts_only = raw_texts_only

    def __iter__(self):
        if not self.raw_texts_only:
            for argument in read_arguments(setup.ARGS_ME_CSV_PATH):
                if argument.id in self.ids:
                    yield argument

                    self.ids.remove(argument.id)
                    if len(self.ids) == 0:
                        return
        else:
            for row in read_csv(setup.ARGS_ME_CSV_PATH):
                id = row[9]
                text = row[0]
                if id in self.ids:
                    yield (id, text)

                    self.ids.remove(id)
                    if len(self.ids) == 0:
                        return

        print('Not all arguments found...')
        return self.ids
