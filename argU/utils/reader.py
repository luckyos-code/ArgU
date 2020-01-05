import csv
import numpy as np
from preprocessing import tools
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from preprocessing.tools import model_text


def read_csv(path, max_rows=-1):
    """Lies CSV zeilenweise ein

    Arguments:
        path (str): Pfad zur CSV-Datei
        max_rows (int, default=-1): Maximale Anzahl an Zeilen.
            Beim Default-Wert wird die ganze Datei eingelesen.

    Yields:
        row (list): Zeile der CSV-Datei
    """

    with open(path, 'r', newline='', encoding='utf-8') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        next(csv_reader)

        for i, row in enumerate(csv_reader):
            if i == max_rows:
                break
            yield row


def read_csv_header(path):
    """Lies CSV Header ein

    Arguments:
        path (str): Pfad zur CSV-Datei

    Returns:
        header (list): CSV-Header
    """

    with open(path, 'r', newline='', encoding='utf-8') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        return next(csv_reader)


# class Text:
#     """Allgemeines Textobjekt, das als Basis für Argumente und Debatten dient

#     Args:
#         raw_text (str): Unbearbeiteter Text
#     """

#     def __init__(self, row):
#         self.raw_text = None

#     @property
#     def text(self):
#         return self.raw_text


# class Debate:
#     """Debatte, die aus einer Zeile der CSV generiert wird"""

#     def __init__(self, row):
#         self.id = row[2]
#         self.text = row[5].split()


# def read_debates(path, max_debates=-1):
#     """Generator, um die CSV schrittweise in Debatten-Objekte umzuwandelm

#     Args:
#         path (str): Pfad zur CSV-Datei
#         max_debates (int, default=-1): Maximale Anzahl verschiedener Debatten.
#             Bei -1 werden alle Debatten eingelesen.

#     Yields:
#         Debate
#     """

#     debate_ids = set([])
#     for row in read_csv(path, max_rows=-1):
#         debate = Debate(row)

#         if debate.id not in debate_ids:
#             debate_ids.add(debate.id)
#             yield debate

#         if len(debate_ids) == max_debates:
#             break

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
    def text(self):
        return tools.clean_text(self.text_raw)

    def get_vec(self, model, vector_size):
        """Erstelle einen Vektor aus dem Dokument

        Returns:
            (numpy.array or None, list or None): Embedding und unbekannte Wörter.
                Ist keines der Wörter bekannt, wird (None, None) zurückgegeben.
        """

        emb_matrix = np.zeros(
            (len(self.text), vector_size)
        )

        unk_word_count = 0
        unk_words = []
        for i, word in enumerate(self.text):
            try:
                emb_vec = model.wv[word]
                emb_matrix[i] = emb_vec / LA.norm(emb_vec)
            except Exception as e:
                unk_word_count += 1
                unk_words.append(word)

        if emb_matrix.shape[0] == 0:
            return (None, None)

        if len(self.text) == len(unk_words):
            return (None, None)

        return (
            np.sum(emb_matrix, axis=0) / (emb_matrix.shape[0]),
            unk_words
        )


class ArgumentTextIterator:
    """Iterator für die Texte alle Argumente"""

    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument.text.split()


class ArgumentCbowIterator:
    """Iterator für bestimmte Formatierung, umd as CBOW Modell
        zu trainieren"""

    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield model_text(argument.text).split()


class ArgumentIterator:
    """Iterator für alle Argument Objekte"""

    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument


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
                yield argument

                found_arguments += 1
                if found_arguments == len(self.ids):
                    return
        print('Not all arguments found...')


# class DebateTitelsIterator:
#     def __init__(self, path, max_debates=-1):
#         self.path = path
#         self.max_debates = max_debates

#     def __iter__(self):
#         for debate in read_debates(self.path, self.max_debates):
#             yield debate.text


# class DebateIterator:
#     def __init__(self, path, max_debates=-1, attribute=None):
#         self.path = path
#         self.max_debates = max_debates
#         self.attribute = attribute

#     def __iter__(self):
#         for debate in read_debates(self.path, self.max_debates):
#             if self.attribute is None:
#                 yield debate
#             else:
#                 yield getattr(debate, self.attribute)


# class FindDebateIterator:
#     """Suche gegebene Debatten anhand vorgegebener IDs

#     Args:
#         path (str): Pfad zur CSV Datei
#         ids (list): Liste von IDs

#     Yield:
#         Debate
#     """

#     def __init__(self, path, ids):
#         self.path = path
#         self.ids = ids

#     def __iter__(self):
#         found_debates = 0
#         for debate in read_debates(self.path, -1):
#             if debate.id in self.ids:
#                 yield debate

#                 found_debates += 1
#                 if found_debates == len(self.ids):
#                     break
