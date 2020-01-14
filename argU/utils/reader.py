import csv
import numpy as np
from preprocessing import tools
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from preprocessing.tools import machine_model_clean
from tqdm import tqdm


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
        self.previous_argument = row[3]
        self.next_argument = row[8]

    @property
    def text_nl(self):
        """Natural Language Text"""
        return tools.natural_language_clean(self.text_raw)

    @property
    def text_machine(self):
        """Text for CBOW, BM25"""
        return tools.machine_model_clean(self.text_raw)

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

    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument

class TrainCSVIterator:
    def __init__(self, train_cbow_path, only_texts=False, max_rows=-1):
        self.train_cbow_path = train_cbow_path
        self.only_texts = only_texts
        self.max_rows = max_rows

    def __iter__(self):
        with open(self.train_cbow_path, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(
                f_in,
                delimiter='|',
                quotechar='"',
            )

            for i, row in enumerate(reader):
                if len(row) <= 1:
                    continue
                if row[0] == '' or row[1] == '':
                    continue

                row[1] = row[1].strip().split()
                if self.only_texts:
                    yield row[1]
                else:
                    yield row

                if (i+1) == self.max_rows:
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

    def __init__(self, path, ids):
        self.path = path
        self.ids = ids

    def __iter__(self):
        found_arguments = 0
        for argument in read_arguments(self.path, -1):
            if argument.id in self.ids:
                print(argument.id)
                yield argument

                found_arguments += 1
                if found_arguments == len(self.ids):
                    return
        print('Not all arguments found...')