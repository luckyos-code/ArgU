import os
import rootpath
import argparse
import sys

from utils.reader import read_arguments, ArgumentIterator
from utils.reader import ArgumentTextIterator, read_csv_header
from utils.beautiful import print_argument_texts, print_embedding_examples
from indexing.models import CBOW, Text2Vec, BM25, Argument2Vec

parser = argparse.ArgumentParser()
parser.add_argument('status', default='train', choices=['train', 'load'])
args = parser.parse_args()

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, f'cbow.arguments.model')

first_n_args = 5000

######################################################################
# CBOW

cbow = CBOW()
if args.status == 'train':
    cbow.build(
        ArgumentTextIterator(
            CSV_PATH,
            max_args=first_n_args,
        ),
        store_path=MODEL_PATH,
        min_count=8,
    )
elif args.status == 'load':
    cbow.load(MODEL_PATH)

print(f"Vokabular Länge: {len(cbow.model.wv.vocab)}")

# print_embedding_examples(cbow.model, ['drugs', 'Trump', 'abortion', 'islam'])


######################################################################
# Arg2Vec und Test

arguments = ArgumentIterator(CSV_PATH, max_args=first_n_args)
a2v_model = Argument2Vec(cbow.model, arguments)

query = 'gay marriage'
ids, similarities = a2v_model.most_similar(query, topn=10)
print()
print(f"Query: {query}")
print_argument_texts(ids, CSV_PATH)

######################################################################
# BM25

bm25 = BM25()
# bm25.build(CSV_PATH, max_args=max_args)
# best_argument_ids = bm25.get_top_n_ids(query, top_n=15)

# print_argument_texts(best_argument_ids, CSV_PATH)
