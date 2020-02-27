import sys
import os
import rootpath
from tqdm import tqdm
from collections import Counter

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.mongodb import load_db
    from argU.utils.arguments import Argument
except Exception as e:
    print(e)
    sys.exit(0)

db = load_db()

coll_args = db[setup.MONGO_DB_COL_ARGS]
coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]
coll_args_train = db[setup.MONGO_DB_COL_TRAIN]
coll_args_trans = db[setup.MONGO_DB_COL_TRANSLATION]

number_of_args = 100
max_arg_len = 4000
score_abs_min = 0.6

count = 0
for arg_train in coll_args_train.find():
    sents = coll_sents.find_one({'_id': arg_train['_id']})
    if abs(sents['score']) >= score_abs_min:
        arg = coll_args.find_one({'_id': arg_train['_id']})
        text = Argument.get_text(arg)

        if len(text) <= max_arg_len:
            dicussion_title = arg["context"]["discussionTitle"]
            arg_id = coll_args_trans.find_one({'_id': arg['_id']})['arg_id']
            count += 1

            print(f'{dicussion_title}\n{"="*60}\n')
            print(Argument.get_text(arg))
            print('\t>', arg_id)
            print('\t>', sents)
            print()

            if count >= number_of_args:
                break
