import os
import rootpath
import sys

from utils.reader import read_arguments, ArgumentTextIterator
from utils.beautiful import print_argument_texts
from indexing.models import CBOW, BM25Manager, Argument2Vec, MixtureModel

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
A2V_MODEL_PATH = os.path.join(RESOURCES_PATH, 'a2v.model')
BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.model')

######################################################################
# CBOW

cbow = CBOW.load(CBOW_MODEL_PATH)

######################################################################
# Arg2Vec

a2v = Argument2Vec.load(A2V_MODEL_PATH)

######################################################################
# BM25

bm25_manager = BM25Manager.load(BM25_PATH)

######################################################################
# Run query

query = 'gay marriage'

mixture_model = MixtureModel(a2v, bm25_manager)
sims = mixture_model.mmsims(query, alpha=0.6)

best_arg = max(sims.items(), key=lambda item: item[1][2])[0]
print(print_argument_texts([best_arg], CSV_PATH, print_all=True))
