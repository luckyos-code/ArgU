import re
import os
import rootpath
import sys
import nltk
from nltk.corpus import wordnet

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
except Exception as e:
    print(e)
    sys.exit(0)

with open(setup.STOPWORDS_PATH, "r") as f_in:
    stopwords = f_in.read().split("\n")

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


def clean_to_nl(text):
    """Clean to generate a natural language text"""

    text = __full_text_cleaning(text)
    text = __term_cleaning(text)
    text = __cleanup(text)
    return text


def clean_to_sentiment(text):
    text = __url_cleaning(text)
    text = __separate_commas(text)
    text = __special_char_cleaning(text)
    text = __delete_multi_letters(text)
    text = __remove_square_brackets(text)
    text = __parenthesis_cleaning(text)
    text = __clean_sub_points(text)
    text = __special_char_fixes(text)

    text = __term_cleaning(text)
    text = __cleanup(text)

    text = text.replace(URL_TOKEN, '')
    text = text.replace(INFO_TOKEN, '')

    return text.strip()


def clean_to_train(text):
    text = re.sub(r'[,".!?]', '', text)
    text = re.sub(r' - |:|;', ' ', text)

    clean_terms = []
    for term in text.split():
        if term.lower() in stopwords:
            continue
        clean_terms.append(term)
    return ' '.join(clean_terms)


def __full_text_cleaning(text):
    """Text wird als ein String betrachtet und gesäubert
        Das ist der erste Schritt.

    Args:
        text (str): Argument Text raw

    Returns:
        str: gesäuberten Text
    """

    text = __url_cleaning(text)
    text = __separate_commas(text)
    text = __special_char_cleaning(text)
    text = __delete_multi_letters(text)
    text = __remove_square_brackets(text)
    text = __parenthesis_cleaning(text)
    text = __clean_sub_points(text)
    text = __special_char_fixes(text)
    text = __tokenize_numbers(text)

    return text


def __url_cleaning(text):
    text = text.replace('http://', ' http://')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, f' {URL_TOKEN} ', text)
    return text


def __separate_commas(text):
    return text.replace(',', ' ')


def __special_char_cleaning(text):
    text = re.sub(r'([a-zA-Z])\.{2,}', r"\1. ", text)
    text = re.sub(r'\"{2,}', '"', text)
    text = re.sub(r'\.{3,}', " ... ", text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'(\?\!|\!\?)+', '?!', text)
    text = re.sub(r'[~#§&@]', '', text)
    text = text.replace('=', ' ')
    return text


def __delete_multi_letters(text):
    return re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)


def __remove_square_brackets(text):
    text = re.sub(r'\[.*?\]', '', text)
    return text


def __parenthesis_cleaning(text):
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace(')', ') ')
    text = text.replace('(', ' (')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    return text


def __clean_sub_points(text):
    text = re.sub(r'([1-9]+[0-9]*[-][A-Za-z])', r' \1 ', text)
    text = re.sub(r'([1-9]+[0-9]*[).])', r' \1 ', text)
    return text


def __special_char_fixes(text):
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


def __tokenize_numbers(text):
    text = re.sub(r'\d*\.\d+%|\d+%', ' ' + PERCENT_TOKEN + ' ', text)
    text = re.sub(r'\d+', ' ' + NUM_TOKEN + ' ', text)
    return text


def __term_cleaning(text):
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


def __cleanup(text):
    text = text.replace(' )', ')')
    text = text.replace('( ', '(')
    text = text.replace(' ,', ',')
    text = text.replace(' . ', '. ')
    return text


def tokenize_query(query):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    stopwords = open(setup.STOPWORDS_PATH, "r").read().split("\n")
    q = query.split(" ")
    aaa = []
    for i in q:
        for char in punctuations:
            i = i.strip(char)
        if i.strip().lower() not in stopwords:
            # print(nltk.pos_tag(i))
            list = []
            list.append(i)

            try:
                pos_list = nltk.pos_tag(list)
            except LookupError as le:
                nltk.download('averaged_perceptron_tagger')
                pos_list = nltk.pos_tag(list)

            tokens = dict()
            synonyms = []
            antonyms = []
            try:
                for syn in wordnet.synsets(i):
                    for l in syn.lemmas():
                        synonyms.append(l.name())
            except LookupError as le:
                nltk.download('wordnet')
            for s, syno in enumerate(synonyms):
                syno.replace("_", " ")
                synonyms[s] = syno
            for s, syno in enumerate(synonyms):
                for j, syno2 in enumerate(synonyms):
                    if j > s:
                        if syno2 == syno:
                            synonyms[j] = ""
            synonyms = [x for x in synonyms if not x == ""]

            tokens['token'] = i
            tokens['pos_tag'] = pos_list[0][1]
            tokens['synonyms'] = synonyms
            # if l.antonyms():
            # tokens['anton'] = antonyms
            yield tokens


def clean_pos_tags(query):
    tokens = tokenize_query(query)
    new_query = []
    for t in tokens:
        if t['pos_tag'] in ('NN', 'NNS'):
            if sum(1 for c in t['token'] if c.isupper()) > 1:
                new_query.append(t['token'].lower())
            new_query.append(t['token'])
        else:
            new_query.append(t['token'].lower())
    new_query = ' '.join(new_query)
    return new_query


if __name__ == '__main__':
    queries = [
        'Teachers Get Tenure',
        'Vaping E-Cigarettes Safe',
        'Insider Trading Allowed',
        'Corporal Punishment Used Schools',
        'Social Security Privatized',
    ]

    for q in queries:
        print(clean_pos_tags(q))
