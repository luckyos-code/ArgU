import math
import os
import re
import sys

import rootpath
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import settings
    from argU.preprocessing.nlp import clean_to_nl
    from argU.preprocessing.nlp import NUM_TOKEN
    from argU.preprocessing.nlp import PERCENT_TOKEN
    from argU.preprocessing.nlp import URL_TOKEN
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)


class Argument:
    """Simple functions to read data from argument dicts"""

    @staticmethod
    def get_text(arg):
        return arg['premises'][0]['text']

    @staticmethod
    def get_discussion_title(arg):
        return arg['context']['discussionTitle']


class TrainArgsIterator:
    """Iterate over texts for the train collection in MongoDB"""

    def __init__(self, coll, max_args=-1, full_data=False):
        self.coll = coll
        self.max_args = max_args
        self.full_data = full_data

    def __iter__(self):
        for i, arg in tqdm(enumerate(self.coll.find())):
            if i == self.max_args:
                break
            if self.full_data:
                yield arg
            else:
                yield arg['text'].split()


def get_NDCG(scores):
    """Scores as list of values between 1 and 3"""

    res = []
    res_opt = []

    scores_opt = sorted(scores, reverse=True)
    # print(scores)
    # print(scores_opt)

    v = 0
    v_opt = 0

    for j, (s, s_opt) in enumerate(zip(scores, scores_opt)):
        i = j + 1
        v += ((2 ** s) - 1) / math.log2(1 + i)
        v_opt += (2 ** s_opt - 1) / math.log2(1 + i)
    return v / v_opt


def get_precision(relevances):
    """relevances is a list of bools"""

    relevances = [True if r > 1 else False for r in relevances]
    return sum(relevances) / len(relevances)


def find(coll, ids):
    if type(ids) is str:
        ids = [ids]
    return coll.find({'_id': {'$in': ids}})


def fancy_print(coll, ids, arg_len=300, trans_dict=None):
    data = {}
    for arg in find(coll, ids):
        data[arg['_id']] = arg
    for i, id in enumerate(ids):
        if trans_dict is not None:
            arg_id = trans_dict[data[id]['_id']]
        else:
            arg_id = data[id]['_id']
        print(f"{i}. ID: {arg_id} ({data[id]['premises'][0]['stance']}) r = ")
        text = data[id]['premises'][0]['text'][:arg_len]
        text = text.replace('$', '')
        print(f"\t- {text} [...]")
        print('<br>')


def find_short(coll, threshold, max_amount=-1):
    short_args = []
    for arg in coll.find():
        nl_text = clean_to_nl(Argument.get_text(arg))
        if len(nl_text.split()) <= threshold:
            short_args.append(arg)
            if len(short_args) == max_amount:
                break

    return short_args


def get_token_stats(arg):
    nl_text = clean_to_nl(Argument.get_text(arg))
    num_regex = re.compile(NUM_TOKEN)
    percent_regex = re.compile(PERCENT_TOKEN)
    url_regex = re.compile(URL_TOKEN)

    return {
        NUM_TOKEN: len(num_regex.findall(nl_text)),
        PERCENT_TOKEN: len(percent_regex.findall(nl_text)),
        URL_TOKEN: len(url_regex.findall(nl_text)),
    }


def find_tokens(coll):
    for arg in coll.find():
        token_stats = get_token_stats(arg)
        if token_stats[NUM_TOKEN] >= 5:
            print(Argument.get_discussion_title(arg))
            print('\t', Argument.get_text(arg))
            print()


if __name__ == '__main__':

    ids = [
        'c67482ba-2019-04-18T13:32:05Z-00000-000',
        '446913e7-2019-04-18T15:54:16Z-00002-000',
    ]

    # db = mongodb.load_db()
    # coll = db[setup.MONGO_DB_COL_ARGS]
    # coll_trans = db[setup.MONGO_DB_COL_TRANSLATION]
    # trans = dict()
    # for t in tqdm(coll_trans.find()):
    #     trans[t['arg_id']] = t['_id']

    # =======================================
    # Arguments by ids
    # fancy_print(coll, [4688, 4690, 235, 13584, 13582, 24230, 11422])

    # a = coll.find_one(
    #     {'_id': trans['e7b98175-2019-04-18T14:36:18Z-00002-000']}
    # )
    # print(a)

    # =======================================
    # Short arguments

    # args = find_short(coll, threshold=25, max_amount=200)
    # for arg in args:
    #     print(f"* {arg['id']}: {arg['premises'][0]['text']}")

    # =======================================
    # Numbers and URLs

    # args = find_tokens(coll)

    # train_args_iter = TrainArgsIterator(max_args=10)
    # for a in train_args_iter:
    #     print(a)

    # =======================================
    # NDCG Test

    n_100_q1_gains = [3, 3, 1, 1, 0, 2, 0, 2,
                      3, 3, 0, 0, 3, 3, 1, 0, 3, 2, 3, 2]
    n_100_q2_gains = [3, 0, 0, 0, 0, 3, 0, 3,
                      0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    n_100_q3_gains = [1]
    n_100_q4_gains = [3, 3, 0, 1, 2, 0, 0, 2, 0, 0]
    n_100_q5_gains = [3, 3, 1, 2, 1, 3, 3, 3,
                      2, 3, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0]

    n_500_q1_gains = [3, 3, 1, 2, 1, 2, 0, 0,
                      0, 0, 3, 3, 1, 0, 1, 0, 3, 0, 0, 0]
    n_500_q2_gains = [3, 0, 2, 3, 3, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n_500_q3_gains = [0, 1, 0]
    n_500_q4_gains = [3, 3, 2, 3, 1, 3, 1, 2,
                      0, 1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2]
    n_500_q5_gains = [3, 3, 2, 1, 3, 2, 1, 3,
                      3, 3, 3, 3, 2, 2, 3, 2, 1, 2, 1, 2]

    n_1000_q1_gains = [3, 3, 1, 2, 3, 3, 1,
                       2, 2, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2]
    n_1000_q2_gains = [3, 2, 0, 0, 2, 3, 0,
                       3, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    n_1000_q3_gains = [0, 0, 0, 0, 0, 1, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n_1000_q4_gains = [3, 3, 3, 3, 3, 2, 2,
                       3, 2, 3, 3, 1, 0, 3, 2, 2, 3, 3, 2, 2]
    n_1000_q5_gains = [3, 3, 2, 1, 3, 2, 1,
                       3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2]

    n_1000_q1_inout_gains = [3, 3, 3, 2, 1, 2,
                             1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]

    n_1000_q2_inout_gains = [2, 3, 2, 0, 2, 3,
                             0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0]

    n_1000_q3_inout_gains = [0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    n_1000_q4_inout_gains = [3, 3, 3, 3, 3, 2,
                             2, 3, 2, 3, 3, 1, 3, 2, 3, 2, 2, 1, 3, 3]

    n_1000_q5_inout_gains = [3, 3, 1, 3, 2, 1,
                             3, 3, 3, 3, 2, 3, 2, 3, 2, 2, 1, 1, 3, 1]

    n_100_gains = [n_100_q1_gains, n_100_q2_gains,
                   n_100_q3_gains, n_100_q4_gains, n_100_q5_gains]
    n_500_gains = [n_500_q1_gains, n_500_q2_gains,
                   n_500_q3_gains, n_500_q4_gains, n_500_q5_gains]
    n_1000_gains = [n_1000_q1_gains, n_1000_q2_gains,
                    n_1000_q3_gains, n_1000_q4_gains, n_1000_q5_gains]
    n_1000_inout_gains = [n_1000_q1_inout_gains, n_1000_q2_inout_gains,
                          n_1000_q3_inout_gains, n_1000_q4_inout_gains, n_1000_q5_inout_gains]

    for i, n in enumerate(n_1000_inout_gains):
        try:
            res_NDCG = get_NDCG(n)
            res_PREC = get_precision(n)
        except Exception as e:
            pass

        print(f'{i + 1}) PREC: {res_PREC}, NDCG: {res_NDCG}')
    # print()
    # print(np.sum(res_NDCG))

    # fancy_NDCG = ' & '.join([
    # str(round(n, 2)).replace('.', ',') for n in res_NDCG
    # ])
    # print(fancy_NDCG)
