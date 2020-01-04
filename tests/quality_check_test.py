import os
import rootpath
import test_settings
import re
import sys

from utils.beautiful import print_argument_texts
from utils.reader import ArgumentIterator

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')

URL_TOKEN = '<URL>'
NUM_TOKEN = '<NUM>'
NUM_RANGE_TOKEN = NUM_TOKEN + '-' + NUM_TOKEN
PERCENT_TOKEN = '<PERCENT>'
INFO_TOKEN = '<INFO>'


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
    text = re.sub(r'\.{3,}', " ... ", text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'(\?\!|\!\?)+', '?!', text)
    text = text.replace('=', ' ')

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
    text = re.sub(r'(\-{2,})', r' \1 ', text)
    text = re.sub(r'([a-zA-Z0-9]\-) ([a-zA-Z])', r'\1\2', text)

    # Zitierung korrigieren
    text = text.replace('“', '"')
    text = text.replace('”', '"')

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

    return text


# def split_concatenated_words(text):
#     new_splits = []
#     for token in text.split():
#         if not token.isupper() and not token.islower():
#             new_splits.extend(re.findall('[a-zA-Z][^A-Z]*', token))
#         else:
#             new_splits.append(token)

#     return ' '.join(new_splits)


# def get_sub_args(text):
#     p = re.compile("[1-9][\).-]")
#     sub_arg_pos = []
#     sub_args = []

#     for m in p.finditer(text):
#         sub_arg_pos.append(m.start())

#     for i in range(len(sub_arg_pos)):
#         if i + 1 < len(sub_arg_pos):
#             sub_args.append(text[sub_arg_pos[i]:sub_arg_pos[i + 1]])
#         else:
#             sub_args.append(text[sub_arg_pos[i]:])

#     return sub_args

short_args = []
for a in ArgumentIterator(CSV_PATH, max_args=10000):
    a.text_raw = full_text_cleaning(a.text_raw)
    a.text_raw = term_cleaning(a.text_raw)
    a.text_raw = full_text_cleaning_end(a.text_raw)

    terms = a.text_raw.split()
    if len(terms) <= 25:
        print(a.text_raw)
        short_args.append(a.text_raw)

print(len(short_args))
# print(a.text_raw)
# print()
