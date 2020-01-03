import os
import rootpath
import test_settings
import numpy as np

from utils.reader import ArgumentIterator
from utils.beautiful import print_argument_texts
from indexing.models import Argument2Vec, CBOW, DualEmbedding, BM25Manager, MixtureModel

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.arguments.model')
A2V_MODEL_PATH = os.path.join(RESOURCES_PATH, 'a2v.model')

cbow = CBOW()
cbow.load(CBOW_MODEL_PATH)

max_args = 1000

train = False
if train:
    a2v = Argument2Vec(cbow.model, CSV_PATH)
    a2v.build(max_args=max_args)
    a2v.store(A2V_MODEL_PATH)
else:
    a2v = Argument2Vec.load(A2V_MODEL_PATH)

bm25_manager = BM25Manager()
bm25_manager.build(CSV_PATH, max_args=max_args)

mixture_model = MixtureModel(a2v, bm25_manager)
sims = mixture_model.mmsims('Donald Trump')

best_arg = max(sims.items(), key=lambda item: item[1][2])[0]
# print(best_arg)
print(print_argument_texts([best_arg], CSV_PATH, print_all=True))

# for (a, s) in sims.items():
# print(a, s)
