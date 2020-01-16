import os
import rootpath
import sys
from tqdm import tqdm

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from indexing.models import CBOW
from indexing.models import Argument2Vec
from utils.reader import TrainCSVIterator
from utils.utils import path_not_found_exit

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')
A2V_MODEL_PATH = os.path.join(RESOURCES_PATH, 'a2v.json')

train = False

if train:
    path_not_found_exit(TRAIN_PATH)

    CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
    path_not_found_exit(CBOW_MODEL_PATH)
    cbow = CBOW.load(CBOW_MODEL_PATH)

    a2v = Argument2Vec(cbow)
    a2v.build(TrainCSVIterator(TRAIN_PATH, max_rows=-1))
    a2v.store(A2V_MODEL_PATH)
else:
    path_not_found_exit(A2V_MODEL_PATH)

    a2v = Argument2Vec.load(A2V_MODEL_PATH)
    
    print(f"Dict Größe: {len(a2v.av)}")
    for i, (id, (vec, unk)) in enumerate(a2v.av.items()):
        print(id, unk)
        if i >= 100:
            break
