import os
import rootpath
import sys
import test_settings
import operator

import numpy as np
from statistics import mean
from numpy import linalg as LA
from indexing.models import BM25Manager
from utils.reader import ArgumentIterator
from utils.beautiful import print_argument_texts

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.model')

first_n_args = 20000
query = "Trump"

train = False
if train:
    bm25 = BM25Manager()
    bm25.build(CSV_PATH, max_args=first_n_args)
    bm25.store(BM25_PATH)
else:
    bm25 = BM25Manager.load(BM25_PATH)

# for i, (k, v) in enumerate(bm25.index.idf.items()):
    # print(k, v)
    # if i == 20:
    # break

for arg in ArgumentIterator(CSV_PATH, max_args=1000):
    nd = {}
    idf_terms = []
    for token in arg.text.split():
        nd[token] = nd.get(token, 0) + 1
        idf_terms.append(bm25.index.idf[token])
    for token in nd:
        nd[token] = nd[token] * bm25.index.idf[token]

    avg_relevance = mean(idf_terms)
    if avg_relevance < 3:
        print(arg.text)
    # best_word = max(nd.items(), key=operator.itemgetter(1))[0]
    # best_val = nd[best_word]
    # print(best_word, best_val, bm25.index.idf[best_word])

# scores_dict = bm25.norm_scores(query)
# best_arg_id = max(scores_dict.items(), key=operator.itemgetter(1))[0]

# best_score = np.max(np.array(list(scores_dict.values())))
# print(best_score)

# print_argument_texts([best_arg_id], CSV_PATH, print_all=True)
