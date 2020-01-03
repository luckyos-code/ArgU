import os
import rootpath
import sys
import test_settings
import operator

import numpy as np
from numpy import linalg as LA
from indexing.models import BM25Manager
from utils.reader import ArgumentIterator
from utils.beautiful import print_argument_texts

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

first_n_args = 5000
query = "Trump"

bm25 = BM25Manager()
bm25.build(CSV_PATH, max_args=first_n_args)

scores_dict = bm25.norm_scores(query)
# print(scores_dict)
best_arg_id = max(scores_dict.items(), key=operator.itemgetter(1))[0]

# best_score = np.max(np.array(list(scores_dict.values())))
# print(best_score)
# sys.exit(0)

# best_arg = ''
# for (arg_id, score) in scores.items():
# if score == best_score:
# best_arg = arg_id
# break
# print(best_arg)

# print_argument_texts([best_arg_id], CSV_PATH, print_all=True)
