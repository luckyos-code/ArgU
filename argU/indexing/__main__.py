import os
import rootpath
import argparse

from utils.reader import read_arguments
from indexing.models import CBOW

parser = argparse.ArgumentParser()
parser.add_argument('status', default='train', choices=['train', 'load'])
args = parser.parse_args()

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')

cbow = CBOW(min_count=3)

if args.status == 'train':
    cbow.build(CSV_PATH, max_args=10000, store_path=MODEL_PATH)
    # cbow.store(MODEL_PATH)
elif args.status == 'load':
    cbow.load(MODEL_PATH)

model = cbow.model

main_word = 'christianity'
similar_words = model.most_similar(main_word)

print()
print(main_word)
print('=' * 20)
for w, score in similar_words:
    print(f"{w} - {score:.4f}")
