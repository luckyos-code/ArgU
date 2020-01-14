import os
import rootpath
import sys

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from indexing.models import BM25Manager
from utils.reader import TrainCSVIterator
from utils.utils import path_not_found_exit
from utils.beautiful import print_argument_texts

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.json')

train = False

if train:
    path_not_found_exit(TRAIN_PATH)

    bm25_manager = BM25Manager()
    bm25_manager.build(TrainCSVIterator(TRAIN_PATH, max_rows=30000))
    bm25_manager.store(BM25_PATH)
else:
    path_not_found_exit(BM25_PATH)

    bm25_manager = BM25Manager.load(BM25_PATH)
    scores = bm25_manager.norm_scores('Trump is bad')

    print()
    best_args = []
    for i, (arg, score) in enumerate(scores.items()):
        best_args.append(arg)
        print(arg, score)

        if i+1 == 10:
            break
    
    print("\nDas beste Argument:")
    print_argument_texts(best_args, CSV_PATH)