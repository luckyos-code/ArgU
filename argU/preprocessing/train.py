import csv
import os
import sys
import rootpath
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.utils.reader import ArgumentIterator
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)


def generate_train_file(max_args=-1, len_threshold=25):
    """Generate training file for BM25 and CBOW"""

    with open(
        setup.TRAIN_ARGS_PATH, 'w', encoding='utf-8', newline=''
    ) as f_out:

        writer = csv.writer(f_out, **setup.TRAIN_ARGS_CONFIG)

        for arg in tqdm(ArgumentIterator(max_args=max_args)):
            if len(arg.text_nl.split()) >= len_threshold:
                writer.writerow([arg.id, arg.text_machine.strip()])
