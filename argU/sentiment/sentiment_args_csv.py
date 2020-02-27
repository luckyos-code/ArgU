import csv
import os
import rootpath
import sys
from tqdm import tqdm
from difflib import SequenceMatcher

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

# Imports nach Pfad-Erweiterung
from utils.reader import ArgumentIterator
from preprocessing.tools import clean_text_simple

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
SENTIMENT_CSV_PATH = os.path.join(RESOURCES_PATH, 'sentiment_args.csv')

# Erstelle eine CSV mit dem Header: <argument_id, stance, cleaned_text>

with open(SENTIMENT_CSV_PATH, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    headers = ['argument_id', 'stance', 'cleaned_text']
    writer.writerow(headers)

    for argument in tqdm(ArgumentIterator(CSV_PATH, max_args=-1)):
        line = [argument.id, argument.stance, clean_text_simple(argument.text_raw)]
        writer.writerow(line)
