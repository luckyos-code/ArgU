import os
import rootpath
import sys
import csv
from tqdm import tqdm

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from indexing.models import CBOW
from utils.reader import TrainCSVIterator
from utils.utils import path_not_found_exit

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')

train = False

if train:
    path_not_found_exit(TRAIN_PATH)

    cbow = CBOW()
    cbow.build(
        TrainCSVIterator(TRAIN_PATH, only_texts=True), 
        store_path=MODEL_PATH, 
        min_count=5,
    )
else:
    path_not_found_exit(MODEL_PATH)

    CBOW.load(MODEL_PATH)
    model = cbow.model

    print(f"|Vokab| = {len(model.wv.vocab)}")

    word = 'Trump'
    most_sim = model.wv.most_similar(word)
    print(f"Am ähnlichsten zu \"{word}\" -> {most_sim}")
