import os
import rootpath
import test_settings
import sys

from utils.beautiful import print_argument_texts
from indexing.models import CBOW
from utils.reader import ArgumentIterator, ArgumentTextIterator, ArgumentCbowIterator

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, f'cbow.model')

cbow = CBOW()
cbow.load(CBOW_MODEL_PATH)

for tokens in ArgumentCbowIterator(CSV_PATH, max_args=100):
    for i, token in enumerate(tokens):
        try:
            cbow.model.wv[token]
        except Exception as e:
            tokens[i] = '<UNK>'
    print(' '.join(tokens), '\n')
