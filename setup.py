import csv
import os
import rootpath

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')

if not os.path.isdir(RESOURCES_PATH):
    os.mkdir(RESOURCES_PATH)
    print('Resource directory created...')

# Input and output files
TOPICS_PATH = os.path.join(ROOT_PATH, 'topics.xml')
RUN_PATH = os.path.join(ROOT_PATH, 'run.txt')
ARGS_ME_JSON_PATH = os.path.join(RESOURCES_PATH, 'args-me.json')

# Processed files
ARGS_ME_CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
TRAIN_ARGS_PATH = os.path.join(RESOURCES_PATH, 'train_args.csv')
TRAIN_ARGS_CONFIG = {'delimiter': '|',
                     'quotechar': '"',
                     'quoting': csv.QUOTE_MINIMAL}

CBOW_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
BM25_META_PATH = os.path.join(RESOURCES_PATH, 'bm25.meta.json')
BM25_DOCS_PATH = os.path.join(RESOURCES_PATH, 'bm25.docs.csv')
BM25_DOCS_CONFIG = {'delimiter': '*',
                    'quotechar': '|',
                    'quoting': csv.QUOTE_MINIMAL}

INDEX_PATH = os.path.join(RESOURCES_PATH, 'index.csv')
INDEX_CONFIG = {'delimiter': '|',
                'quotechar': '*',
                'quoting': csv.QUOTE_MINIMAL}
SCORES_PATH = os.path.join(RESOURCES_PATH, 'scores.csv')
SCORES_CONFIG = {
    'delimiter': '|',
    'quotechar': '*',
    'quoting': csv.QUOTE_MINIMAL,
}

SENTIMENTS_PATH = os.path.join(
    ROOT_PATH, 'argU/sentiment/results/argument_sentiments.csv'
)
SENTIMENTS_CONFIG = {
    'delimiter': ',',
    'quotechar': '"',
    'quoting': csv.QUOTE_MINIMAL,
}

IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images/')


def assert_file_exists(path):
    assert os.path.isfile(path), f'File \"{path}\" does not exist...'
