import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR)

RESOURCES_PATH = os.path.join(BASE_DIR, 'resources/')

ARGS_ME_JSON_PATH = os.path.join(RESOURCES_PATH, 'args-me.json')
TOPICS_PATH = os.path.join(RESOURCES_PATH, 'topics.xml')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')

MAPPING_PATH = os.path.join(RESOURCES_PATH, 'mapping.json')
CBOW_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
TREC_PATH = os.path.join(RESOURCES_PATH, 'args-me.trec')
TREC_PATH_TOPICS = os.path.join(RESOURCES_PATH, 'topics.trec')

TERRIER_RESULTS_PATH = os.path.join(RESOURCES_PATH, 'terrier.res')

SENTIMENTS_PATH = os.path.join(BASE_DIR, 'argU', 'sentiment', 'results', 'argument_sentiments.csv')
SENTIMENTS_CONFIG = {
    'delimiter': ',',
    'quotechar': '"',
    'quoting': csv.QUOTE_MINIMAL,
}

MONGO_DB_NAME = 'ArgU'
MONGO_DB_URL = 'mongodb://localhost:27017/'
MONGO_DB_COL_ARGS = 'args'

METHOD_NO = 'ulT1DetroitnitzCbowDPHSentNo'
METHOD_EMOTIONAL = 'ulT1DetroitnitzCbowDPHSentEmotional'
METHOD_NEUTRAL = 'ulT1DetroitnitzCbowDPHSentNeutral'
