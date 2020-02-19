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

IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images/')
STOPWORDS_PATH = os.path.join(ROOT_PATH, 'stopwords_eng.txt')

MONGO_DB_NAME = 'argU'
MONGO_DB_URL = 'mongodb://localhost:27017/'
MONGO_DB_COL_ARGS = 'args'
MONGO_DB_COL_TRAIN = 'train'
MONGO_DB_COL_TRANSLATION = 'trans'
MONGO_DB_COL_SENTIMENTS = 'sents'
MONGO_DB_COL_EMBEDDINGS = 'emb'
MONGO_DB_COL_RESULTS = 'results'

TREC_PATH = os.path.join(RESOURCES_PATH, 'args-me.trec')
TREC_PATH_TOPICS = os.path.join(RESOURCES_PATH, 'topics.trec')
CBOW_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
TERRIER_RESULTS_PATH = os.path.join(RESOURCES_PATH, 'terrier.res')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output.res')

SENTIMENTS_PATH = os.path.join(
    ROOT_PATH, 'argU/sentiment/results/argument_sentiments.csv'
)
SENTIMENTS_CONFIG = {
    'delimiter': ',',
    'quotechar': '"',
    'quoting': csv.QUOTE_MINIMAL,
}

METHOD = 'ulT1DetroitnitzCbowDPHSent'


def assert_file_exists(path):
    assert os.path.isfile(path), f'File \"{path}\" does not exist...'


def file_exists(path):
    return os.path.isfile(path)
