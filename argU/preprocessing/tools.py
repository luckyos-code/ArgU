import re
import os
import rootpath


@DeprecationWarning
def denoise(text):
    """Entferne Rauschen aus einem Text

    1. Lower Case
    2. Alles in eckigen Klammern wird entfernt
    3. Entferne Web-Adressen
    4. Entferne Sonderzeichen außer Apostrophe

    Returns:
        list mit den einzelnen, aufeinanderfolgenden Wörtern
    """

    denoised_text = []

    text = re.sub(r'\[[^]]*\]', '', text)
    for word in text.split():
        if not any(w in word for w in ['www.', '.com', 'com/']):
            word = re.sub(r'[^\w\s]', '', word).strip()
            word_split = word.split()
            if word != '':
                denoised_text.append(word)
    return denoised_text


URL_TOKEN = '<URL>'
NUM_TOKEN = '<NUM>'
NUM_RANGE_TOKEN = NUM_TOKEN + '-' + NUM_TOKEN
PERCENT_TOKEN = '<PERCENT>'
INFO_TOKEN = '<INFO>'
special_tokens = [
    URL_TOKEN,
    NUM_TOKEN,
    NUM_RANGE_TOKEN,
    PERCENT_TOKEN,
    INFO_TOKEN,
]

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')


def full_text_cleaning(text):
    """Text wird als ein String betrachtet und gesäubert
        Das ist der erste Schritt.

    Args:
        text (str): Argument Text raw

    Returns:
        str: gesäuberten Text
    """

    # URLs säubern und durch `URL_TOKEN` ersetzen.
    # Nicht alle URLs werden gefunden! (s. Schritt 2)
    text = text.replace('http://', ' http://')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    found_urls = re.findall(url_regex, text)
    text = re.sub(url_regex, f' {URL_TOKEN} ', text)

    # Kommas splitten für bessere Verarbeitung von Nummern
    text = text.replace(', ', ' , ')

    # Entferne Sonderzeichen die zu oft hintereinander stehen
    text = re.sub(r'([a-zA-Z])\.{2,}', r"\1. ", text)
    text = re.sub(r'\"{2,}', '"', text)
    text = re.sub(r'\.{3,}', " ... ", text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'(\?\!|\!\?)+', '?!', text)
    text = text.replace('=', ' ')

    # Ersetze Buchstaben, die mehr als 2 mal hintereinander stehen durch einen
    text = re.sub(
        r'([a-zA-Z])\1{3,}',
        r'\1',
        text
    )

    # Entferne eckige Klammern und Inhalt
    barcket_pattern = r'\[.*?\]'
    found_brackets = re.findall(barcket_pattern, text)
    text = re.sub(barcket_pattern, '', text)

    # Prozente finden und durch `PERCENT_TOKEN`
    percent_pattern = r'\d+%'
    text = re.sub(percent_pattern, f' {PERCENT_TOKEN} ', text)

    # Runde Klammern
    text = text.replace('{', '(')
    text = text.replace('}', ')')

    text = text.replace(')', ') ')
    text = text.replace('(', ' (')

    # Unterargumente finden und trennen: 1) ... 1. ... 1- ...
    text = re.sub(r'([1-9]+[0-9]*[-][A-Za-z])', r' \1 ', text)
    text = re.sub(r'([1-9]+[0-9]*[).])', r' \1 ', text)

    # Bindestriche
    text = re.sub(r'\-{2,}', ' - ', text)
    text = re.sub(r'([a-zA-Z0-9]\-) ([a-zA-Z])', r'\1\2', text)

    # Zitierung korrigieren
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('’', '\'')

    # Sternchen entfernen
    text = text.replace('*', '')

    # Doppelpunkte formatieren
    text = text.replace(':', ': ')

    return text


def term_cleaning(text):
    """Text wird gesplittet und Terme werden einzeln bereinigt
        Schritt 2

    Args:
        text (str): Argument text nach 1. Schritt

    Returns:
        str: bereinigter Text
    """

    new_terms = []
    splits = text.split()

    for i, term in enumerate(splits):
        # Großgeschriebene Wörter klein machen
        if len(term) >= 5 and term.isupper() and term not in special_tokens:
            term = term.lower()

        # Einzelne Punkte löschen
        if term == '.':
            continue

        # Zu kurze Sätze mit (!, ?) werden gelöscht
        if i != 0:
            if splits[i - 1][-1] in '.?!' and splits[i][-1] in '!?':
                continue

        # Was in Klammern steht verarbeiten
        if term[0] == '(' and term[-1] == ')':
            enclosed = term[1:-1]
            if enclosed == '':
                continue
            else:
                try:
                    enclosed = int(enclosed)
                    if enclosed > 100:
                        raise ValueError
                except ValueError:
                    term = INFO_TOKEN

        # Zu lange Elemente entfernen
        if len(term) > 45:
            continue

        # Freie Zahlen Token
        try:
            int(term)
            term = NUM_TOKEN
        except ValueError:
            pass

        # Wertspannen Token
        try:
            nums = term.split('-')
            if len(nums) == 2:
                int(nums[0])
                int(nums[1])
                term = NUM_RANGE_TOKEN
        except ValueError:
            pass

        # Weitere URLs, die vorher noch nicht abgefangen wurden
        if '.com' in term:
            term = URL_TOKEN

        # Apostrophe korrigieren
        if '"' in term[1:-1]:
            matches = re.findall(r'[a-z]\"[a-z]', term)
            if matches:
                if len(term.split('"')[1]) <= 2:
                    term = re.sub(r'([a-z])\"([a-z])', r"\1'\2", term)

        # ENDE: Term einfügen
        new_terms.append(term)

    return ' '.join(new_terms)


def full_text_cleaning_end(text):
    """Nachbearbeitung
        Schritt 3

    Args:
        text (str): Argument text nach 1. Schritt

    Returns:
        str: bereinigter Text
    """

    text = text.replace(' )', ')')
    text = text.replace('( ', '(')
    text = text.replace(' ,', ',')
    text = text.replace(' . ', '. ')

    return text


def clean_text(text):
    text = full_text_cleaning(text)
    text = term_cleaning(text)
    text = full_text_cleaning_end(text)

    return text


def model_text(text):
    """Entferne noch mehr Sonderzeichen, um einen sauberen Text zu erzeugen.
        Damit können Wahrscheinlich keine Sätze mehr sauber getrennt werden!

    Args:
        text (str): Eingabetext

    Returns:
        str: bearbeiteter Text
    """

    stopwords = open(STOPWORDS_PATH, "r").read().split("\n")
    text = re.sub(r'[,"]', '', text)
    text = re.sub(r' - |:|;', ' ', text)

    clean_terms = []
    for term in text.split():
        # Stopwörter entfernen
        if term.lower() in stopwords:
            continue

        clean_terms.append(term)

    return ' '.join(clean_terms)
