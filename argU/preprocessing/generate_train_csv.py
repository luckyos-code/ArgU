import os
import rootpath
import sys
import csv
from tqdm import tqdm

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from utils.reader import ArgumentIterator


def generate_cbow_train_file(source_path, train_file_path):
    with open(train_file_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(
            f_out,
            delimiter='|',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        for arg in tqdm(ArgumentIterator(source_path, max_args=-1)):
            if len(arg.text_nl.split()) >= 25:
                writer.writerow([arg.id, arg.text_machine.strip()])
