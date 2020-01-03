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


def full_text_cleaning(text):

    # 1. URLs säubern und durch <URL> ersetzen
    text = text.replace('http://', ' http://')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    found_urls = re.findall(url_regex, text)
    text = re.sub(url_regex, '<URL>', text)

    # Kommas splitten für bessere Verarbeitung von Nummern
    text = text.replace(', ', ' , ')

    # 2. Entferne Sonderzeichen die zu oft hintereinander stehen
    text = re.sub(r'\.{3,}', " ... ", text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'(\?\!|\!\?)+', '?!', text)
    text = text.replace('=', ' ')

    # 3. Entferne eckige Klammern und Inhalt
    barcket_pattern = r'\[.*?\]'
    found_brackets = re.findall(barcket_pattern, text)
    text = re.sub(barcket_pattern, '', text)

    # 4. Prozente finden
    percent_pattern = r'\d+%'
    # found_percent = re.findall(percent_pattern, text)
    text = re.sub(percent_pattern, ' <PERCENT> ', text)

    # 5. Runde Klammern

    text = text.replace('{', '(')
    text = text.replace('}', ')')

    text = text.replace(')', ') ')
    text = text.replace('(', ' (')

    # 6. Unterargumente finden und trennen
    text = re.sub(r'([1-9]+[0-9]*[).-])', r'\1 ', text)

    # 7. Bindestriche
    text = re.sub(r'(\-{2,})', r' \1 ', text)
    text = re.sub(r'([a-zA-Z0-9]\-) ([a-zA-Z])', r'\1\2', text)

    # Zitierung
    text = text.replace('“', '"')
    text = text.replace('”', '"')

    return text


def term_cleaning(text):
    new_terms = []
    splits = text.split()

    for i, term in enumerate(splits):
        # 1. Zu kurze Sätze werden gelöscht (< 2 Wörter, aber nur für ? und !)
        if i != 0:
            if splits[i - 1][-1] in '.?!' and splits[i][-1] in '!?':
                continue

        # 2. Was in Klammern steht verarbeiten
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
                    term = '<INFO>'

        # 3. Zu lange Elemente entfernen
        if len(term) > 45:
            continue

        # 4. Freie Zahlen
        try:
            int(term)
            term = '<NUM>'
        except ValueError:
            pass
        # Term einfügen
        new_terms.append(term)

    return ' '.join(new_terms)


def full_text_cleaning_end(text):
    text = text.replace(' )', ')')
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


for a in list(ArgumentIterator(CSV_PATH, max_args=50))[-10:]:
    a.text_raw = full_text_cleaning(a.text_raw)
    a.text_raw = term_cleaning(a.text_raw)
    a.text_raw = full_text_cleaning_end(a.text_raw)

    print(a.text_raw)
    print()
