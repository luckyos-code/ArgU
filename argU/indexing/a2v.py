import argparse
import os
import rootpath
import sys
from tqdm import tqdm
import numpy as np

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.indexing.models import CBOW
    from argU.indexing.models import DESM
    from argU.utils.arguments import TrainArgsIterator
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If force, retrain CBOW model'
    )
    args = parser.parse_args()

    db = load_db()
    coll = db[setup.MONGO_DB_COL_TRAIN]

    if setup.file_exists(setup.CBOW_PATH) and not args.force:
        cbow = CBOW.load()
    else:
        cbow = CBOW()
        cbow.build(
            TrainArgsIterator(coll),
            min_count=5,
            size=100,
            window=7,
        )
        cbow.store()

    desm = DESM(cbow)
    print(desm.model_in.wv.most_similar('Trump'))
    print(desm.model_out.most_similar('Trump'))

    # diff_words = set([])
    # for i in range(10):
    #     v = np.random.random_sample((100,))
    #     for (w, s) in desm.model_out.most_similar(positive=[v], topn=10):
    #         diff_words.add(w)
    # print(diff_words)
    # print(len(diff_words))
    # sys.exit(0)

    embeddings = []
    last_emb = None

    from scipy.spatial.distance import cosine
    for i, arg in enumerate(
        TrainArgsIterator(coll, full_data=True, max_args=-1)
    ):
        arg_emb = desm.arg_to_emb(arg, model_type='out')

        # if last_emb is not None:
        # print()
        # print('\t', 1 - cosine(last_emb, arg_emb))
        # print()
        # last_emb = arg_emb

        # print(arg['text'][:200])
        # print()
        embeddings.append({
            '_id': arg['_id'],
            'emb': arg_emb.tolist(),
        })

    print('Initialize embeddings in MongoDB...')
    coll_emb = db[setup.MONGO_DB_COL_EMBEDDINGS]
    coll_emb.drop()
    coll_emb.insert_many(embeddings)

    print(coll_emb.find_one({}))
