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


class Argument():
    """Argument, das aus einer Zeiler der CSV generert wird"""

    def __init__(self, row):
        self.id = row[9]
        self.text = tools.denoise(row[0])
        self.text_raw = row[0]
        self.stance = row[1]
        self.previous_argument = row[3]
        self.next_argument = row[8]


def read_arguments(path, max_args=-1):
    """Generator um alle CSV direkt in Argument-Objekte zu konvertieren

    Arguments:
        path (str) Pfad zur CSV-Datei
        max_args (int, default=-1): Maximale Anzahl an Argumente.
            Beim Default-Wert werden alle Argumente eingelesen.

    Yields:
        Argument
    """

    for row in read_csv(path, max_rows=max_args):
        argument = Argument(row)
        yield argument


class ArgumentTextsIterator:
    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument.text


class ArgumentIterator:
    def __init__(self, path, max_args=-1):
        self.path = path
        self.max_args = max_args

    def __iter__(self):
        for argument in read_arguments(self.path, self.max_args):
            yield argument


class FindArgumentIterator:
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
