import os
import rootpath
import argparse
import sys

from utils.reader import read_arguments, read_debates, ArgumentIterator
from utils.reader import DebateTitelsIterator, DebateIterator
from utils.reader import ArgumentTextsIterator, read_csv_header
from utils.beautiful import print_argument_texts, print_embedding_examples
from utils.beautiful import print_debate_titles
from indexing.models import CBOW, Text2Vec, BM25, Argument2Vec

parser = argparse.ArgumentParser()
parser.add_argument('status', default='train', choices=['train', 'load'])
args = parser.parse_args()


ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

MODEL_TYPE = 'debate'
MODEL_PATH = os.path.join(RESOURCES_PATH, f'cbow.{MODEL_TYPE}.model')

max_args = 5000
max_debates = 5000
query = 'gay marriage'

# print(read_csv_header(CSV_PATH))

######################################################################
# CBOW

cbow = CBOW()
if args.status == 'train':
    if MODEL_TYPE == 'debate':
        cbow.build(
            DebateIterator(
                CSV_PATH, max_debates=max_debates, attribute='text'
            ),
            store_path=MODEL_PATH,
            min_count=1,
        )
    elif MODEL_TYPE == 'args':
        cbow.build(
            ArgumentTextIterator(
                CSV_PATH,
                max_debates=max_debates,
            ),
            store_path=MODEL_PATH,
        )
elif args.status == 'load':
    cbow.load(MODEL_PATH)

print(f"Vokabular Länge: {len(cbow.model.wv.vocab)}")
# print_embedding_examples(cbow.model, ['drugs', 'Trump', 'abortion', 'islam'])

if MODEL_TYPE == 'args':
    arguments = ArgumentIterator(CSV_PATH, max_args=max_args)
    t2v = Argument2Vec(cbow.model, arguments)
elif MODEL_TYPE == 'debate':
    debates = DebateIterator(CSV_PATH, max_debates=max_debates)
    t2v = Text2Vec(cbow.model, debates)

ids, similarities = t2v.most_similar(query, topn=10)
print()
print(f"Query: {query}")
print_debate_titles(ids, CSV_PATH)

# print('=' * 20)
# argument_iter = ArgumentIterator(CSV_PATH, max_args)
# for i, argument in enumerate(argument_iter):
#     if argument.id in most_similar_arguments:
#         print(f"{i}: Len = {len(argument.text)}, Arg = {argument.text[:15]}")

######################################################################
# BM25

# bm25 = BM25()
# bm25.build(CSV_PATH, max_args=max_args)
# best_argument_ids = bm25.get_top_n_ids(query, top_n=15)

# print_argument_texts(best_argument_ids, CSV_PATH)
