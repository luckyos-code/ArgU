import os
import rootpath
import test_settings
import re
import sys
from sentence_splitter import SentenceSplitter

from utils.beautiful import print_argument_texts
from utils.reader import ArgumentIterator
from preprocessing.tools import clean_text

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')


splitter = SentenceSplitter(language='en')

for a in ArgumentIterator(CSV_PATH, max_args=100):
    a.text_raw = clean_text(a.text_raw)
    sentences = splitter.split(text=a.text_raw)
    for s in sentences:
        print("\t-> ", s)
    print("\n============================================\n")
