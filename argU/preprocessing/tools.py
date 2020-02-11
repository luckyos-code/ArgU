import re
import os
import rootpath


URL_TOKEN = '<URL>'
NUM_TOKEN = '<NUM>'
PERCENT_TOKEN = '<PERCENT>'
INFO_TOKEN = '<INFO>'
special_tokens = [
    URL_TOKEN,
    NUM_TOKEN,
    PERCENT_TOKEN,
    INFO_TOKEN,
]

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
STOPWORDS_PATH = os.path.join(ROOT_PATH, 'stopwords_eng.txt')


def url_cleaning(text):
    """URLs säubern und durch `URL_TOKEN` ersetzen.
        Nicht alle URLs werden gefunden! (s. Schritt 2)
    """

    text = text.replace('http://', ' http://')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    found_urls = re.findall(url_regex, text)
    text = re.sub(url_regex, f' {URL_TOKEN} ', text)
    return text


def comma_separation_cleaning(text):
    return text.replace(', ', ' , ')


def multi_special_char_cleaning(text):
    text = re.sub(r'([a-zA-Z])\.{2,}', r"\1. ", text)
    text = re.sub(r'\"{2,}', '"', text)
    text = re.sub(r'\.{3,}', " ... ", text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'(\?\!|\!\?)+', '?!', text)
    text = re.sub(r'[~#§&@]', '', text)
    text = text.replace('=', ' ')
    return text


def multi_letter_delete(text):
    return re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)


def square_bracket_cleaning(text):
    barcket_pattern = r'\[.*?\]'
    found_brackets = re.findall(barcket_pattern, text)
    text = re.sub(barcket_pattern, '', text)
    return text


def parenthesis_cleaning(text):
    text = text.replace('{', '(')
    text = text.replace('}', ')')
    text = text.replace(')', ') ')
    text = text.replace('(', ' (')
    return text


def sub_args_cleaning(text):
    text = re.sub(r'([1-9]+[0-9]*[-][A-Za-z])', r' \1 ', text)
    text = re.sub(r'([1-9]+[0-9]*[).])', r' \1 ', text)
    return text


def simple_special_char_changes(text):
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


def number_cleaning(text):
    text = re.sub(r'\d*\.\d+%|\d+%', PERCENT_TOKEN, text)
    text = re.sub(r'\d+', NUM_TOKEN, text)
    return text


def full_text_cleaning(text):
    """Text wird als ein String betrachtet und gesäubert
        Das ist der erste Schritt.

    Args:
        text (str): Argument Text raw

    Returns:
        str: gesäuberten Text
    """

    text = url_cleaning(text)
    text = comma_separation_cleaning(text)
    text = multi_special_char_cleaning(text)
    text = multi_letter_delete(text)
    text = square_bracket_cleaning(text)
    text = parenthesis_cleaning(text)
    text = sub_args_cleaning(text)
    text = simple_special_char_changes(text)
    text = number_cleaning(text)

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
        if len(term) >= 5 and term.isupper():
            if not any(s in term for s in special_tokens):
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

        # Weitere URLs, die vorher noch nicht abgefangen wurden
        if '.com' in term or 'www.' in term or '.org' in term:
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


def natural_language_clean(text):
    text = full_text_cleaning(text)
    text = term_cleaning(text)
    text = full_text_cleaning_end(text)

    return text


def sentiment_clean(text):
    text = url_cleaning(text)
    text = comma_separation_cleaning(text)
    text = multi_special_char_cleaning(text)
    text = multi_letter_delete(text)
    text = square_bracket_cleaning(text)
    text = parenthesis_cleaning(text)
    text = sub_args_cleaning(text)
    text = simple_special_char_changes(text)

    text = term_cleaning(text)
    text = full_text_cleaning_end(text)

    text = text.replace(URL_TOKEN, ' ')
    text = text.replace(INFO_TOKEN, ' ')
    text = ' '.join(text.split()).strip()

    return text


with open(STOPWORDS_PATH, "r") as f_in:
    stopwords = f_in.read().split("\n")


def machine_model_clean(text):
    """Entferne noch mehr Sonderzeichen, um einen sauberen Text zu erzeugen.
        Damit können Wahrscheinlich keine Sätze mehr sauber getrennt werden!

    Args:
        text (str): Eingabetext

    Returns:
        str: bearbeiteter Text
    """

    text = re.sub(r'[,"]', '', text)
    text = re.sub(r' - |:|;', ' ', text)

    clean_terms = []
    for term in text.split():
        # Stopwörter entfernen
        if term.lower() in stopwords:
            continue

        clean_terms.append(term)

    return ' '.join(clean_terms)
