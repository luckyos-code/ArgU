import os
import rootpath
import sys
import csv
from tqdm import tqdm

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from utils.reader import ArgumentIterator
from preprocessing.tools import model_text

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
CBOW_TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')

with open(CBOW_TRAIN_PATH, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(
        f_out,
        delimiter='|',
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )

    for arg in tqdm(ArgumentIterator(CSV_PATH, max_args=-1)):
        if len(arg.text.split()) >= 25:
            id = arg.id
            text = model_text(arg.text)
            writer.writerow([id.strip(), text.strip()])
