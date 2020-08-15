import csv
import os
import rootpath

ROOT_PATH = rootpath.detect()
OUTPUT_PATH = os.path.join(ROOT_PATH)

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images/')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')
TREC_PATH = os.path.join(RESOURCES_PATH, 'args-me.trec')
TREC_PATH_TOPICS = os.path.join(RESOURCES_PATH, 'topics.trec')
CBOW_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
TERRIER_RESULTS_PATH = os.path.join(RESOURCES_PATH, 'terrier.res')

SENTIMENTS_PATH = os.path.join(ROOT_PATH, 'argU', 'sentiment', 'results', 'argument_sentiments.csv')
SENTIMENTS_CONFIG = {
    'delimiter': ',',
    'quotechar': '"',
    'quoting': csv.QUOTE_MINIMAL,
}

MONGO_DB_NAME = 'argU'
MONGO_DB_URL = 'mongodb://localhost:27017/'
MONGO_DB_COL_ARGS = 'args'
MONGO_DB_COL_TRAIN = 'train'
MONGO_DB_COL_TRANSLATION = 'trans'
MONGO_DB_COL_SENTIMENTS = 'sents'
MONGO_DB_COL_SENTIMENTS_TRAIN = 'sents_train'
MONGO_DB_COL_EMBEDDINGS = 'emb'
MONGO_DB_COL_EMBEDDINGS_BACKUP = 'emb_back'
MONGO_DB_COL_RESULTS = 'results'
MONGO_DB_COL_RESULTS_BACKUP = 'results_back'

METHOD_NO = 'ulT1DetroitnitzCbowDPHSentNo'
METHOD_EMOTIONAL = 'ulT1DetroitnitzCbowDPHSentEmotional'
METHOD_NEUTRAL = 'ulT1DetroitnitzCbowDPHSentNeutral'


def assert_file_exists(path):
    assert os.path.isfile(path), f'File \"{path}\" does not exist...'


def file_exists(path):
    return os.path.isfile(path)
