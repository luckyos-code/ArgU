<<<<<<< HEAD
import os
import rootpath
import sys

from utils.reader import read_arguments, ArgumentTextIterator
from utils.beautiful import print_argument_texts
from indexing.models import CBOW, BM25Manager, Argument2Vec, MixtureModel

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, f'cbow.arguments.model')
A2V_MODEL_PATH = os.path.join(RESOURCES_PATH, 'a2v.model')
BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.model')

first_n_args = 1000
train_cbow = False
train_a2v = False
train_bm25 = False

######################################################################
# CBOW

cbow = CBOW()
if train_cbow:
    cbow.build(
        ArgumentTextIterator(
            CSV_PATH,
            max_args=first_n_args,
        ),
        store_path=MODEL_PATH,
        min_count=8,
    )
else:
    cbow.load(CBOW_MODEL_PATH)

######################################################################
# Arg2Vec

if train_a2v:
    a2v = Argument2Vec(cbow.model, CSV_PATH)
    a2v.build(max_args=max_args)
    a2v.store(A2V_MODEL_PATH)
else:
    a2v = Argument2Vec.load(A2V_MODEL_PATH)

######################################################################
# BM25

if train_bm25:
    bm25_manager = BM25Manager()
    bm25_manager.build(CSV_PATH, max_args=first_n_args)
    bm25_manager.store(BM25_PATH)
else:
    bm25_manager = BM25Manager.load(BM25_PATH)

######################################################################
# Run query

query = 'gay marriage'

mixture_model = MixtureModel(a2v, bm25_manager)
sims = mixture_model.mmsims(query, alpha=0.6)

best_arg = max(sims.items(), key=lambda item: item[1][2])[0]
print(print_argument_texts([best_arg], CSV_PATH, print_all=True))
=======
import os
import rootpath
import argparse

from utils.reader import read_arguments, ArgumentIterator
from indexing.models import CBOW, Argument2Vec

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
elif args.status == 'load':
    cbow.load(MODEL_PATH)
model = cbow.model

max_args = 1000
query = 'christianity is bad'

a2v = Argument2Vec(model, CSV_PATH, max_args=max_args)
most_simiar_arguments = a2v.most_similar(query)


print()
print(f"Query: {query}")
print('=' * 20)
argument_iter = ArgumentIterator(CSV_PATH, max_args)
for i, argument in enumerate(argument_iter):
    if argument.id in most_simiar_arguments:
        print(f"{i}: Len = {len(argument.text)}, Arg = {argument.text[:15]}")
>>>>>>> rank_test
