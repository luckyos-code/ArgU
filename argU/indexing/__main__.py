import os
import rootpath
import argparse

from utils.reader import read_arguments, ArgumentIterator
from utils.beautiful import print_argument_texts
from indexing.models import CBOW, Argument2Vec, BM25

parser = argparse.ArgumentParser()
parser.add_argument('status', default='train', choices=['train', 'load'])
args = parser.parse_args()

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')

max_args = 1000
query = 'christianity is good'

######################################################################
# CBOW

# cbow = CBOW()
# if args.status == 'train':
#     cbow.build(CSV_PATH, max_args=10000, store_path=MODEL_PATH)
# elif args.status == 'load':
#     cbow.load(MODEL_PATH)

# a2v = Argument2Vec(cbow.model, CSV_PATH, max_args=max_args)
# most_similar_arguments = a2v.most_similar(query)


# print()
# print(f"Query: {query}")
# print('=' * 20)
# argument_iter = ArgumentIterator(CSV_PATH, max_args)
# for i, argument in enumerate(argument_iter):
#     if argument.id in most_similar_arguments:
#         print(f"{i}: Len = {len(argument.text)}, Arg = {argument.text[:15]}")

######################################################################
# BM25

bm25 = BM25()
bm25.build(CSV_PATH, max_args=max_args)
best_argument_ids = bm25.get_top_n_ids(query, top_n=15)

print_argument_texts(best_argument_ids, CSV_PATH)
