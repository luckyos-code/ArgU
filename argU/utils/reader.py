import csv
from preprocessing import tools


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


class Text:
    """Allgemeines Textobjekt, das als Basis für Argumente und Debatten dient

    Args:
        raw_text (str): Unbearbeiteter Text
    """

    def __init__(self, row):
        self.raw_text = None

    @property
    def text(self):
        return self.raw_text


class Argument:
    """Argument, das aus einer Zeile der CSV generiert wird"""

    def __init__(self, row):
        self.id = row[9]
        self.text_raw = row[0]
        self.stance = row[1]
        self.previous_argument = row[3]
        self.next_argument = row[8]

    @property
    def text(self):
        return tools.denoise(self.text_raw)


class Debate:
    """Debatte, die aus einer Zeile der CSV generiert wird"""

    def __init__(self, row):
        self.id = row[2]
        self.text = row[5].split()


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


def read_debates(path, max_debates=-1):
    """Generator, um die CSV schrittweise in Debatten-Objekte umzuwandelm

    Args:
        path (str): Pfad zur CSV-Datei
        max_debates (int, default=-1): Maximale Anzahl verschiedener Debatten.
            Bei -1 werden alle Debatten eingelesen.

    Yields:
        Debate
    """

    debate_ids = set([])
    for row in read_csv(path, max_rows=-1):
        debate = Debate(row)

        if debate.id not in debate_ids:
            debate_ids.add(debate.id)
            yield debate

        if len(debate_ids) == max_debates:
            break


class ArgumentTextsIterator:
    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument.text


class DebateTitelsIterator:
    def __init__(self, path, max_debates=-1):
        self.path = path
        self.max_debates = max_debates

    def __iter__(self):
        for debate in read_debates(self.path, self.max_debates):
            yield debate.text


class DebateIterator:
    def __init__(self, path, max_debates=-1, attribute=None):
        self.path = path
        self.max_debates = max_debates
        self.attribute = attribute

    def __iter__(self):
        for debate in read_debates(self.path, self.max_debates):
            if self.attribute is None:
                yield debate
            else:
                yield getattr(debate, self.attribute)


class ArgumentIterator:
    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument


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
                    break


class FindDebateIterator:
    """Suche gegebene Debatten anhand vorgegebener IDs

    Args:
        path (str): Pfad zur CSV Datei
        ids (list): Liste von IDs

    Yield:
        Debate
    """

    def __init__(self, path, ids):
        self.path = path
        self.ids = ids

    def __iter__(self):
        found_debates = 0
        for debate in read_debates(self.path, -1):
            if debate.id in self.ids:
                yield debate

                found_debates += 1
                if found_debates == len(self.ids):
                    break
