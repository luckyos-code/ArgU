import os
import rootpath
import sys
import csv
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.indexing.models import CBOW
    from argU.indexing.models import DualEmbedding
    from argU.utils.reader import TrainArgsIterator
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)

if __name__ == '__main__':
    cbow = CBOW.load()

    model_in = cbow.model
    dual_embedding_model = DualEmbedding(model_in)
    model_out = dual_embedding_model.model_out

    model = model_in.wv

    print(f"|Vokab| = {len(model.vocab)}")
    words = ['Teacher', 'Tenure']
    arg_ids = set(['c065954f-2019-04-18T14:32:52Z-00001-000',
                   'c065954f-2019-04-18T14:32:52Z-00002-000'])
    args = dict()

    for a_id, a_text in TrainArgsIterator():
        if a_id in arg_ids:
            arg_ids.remove(a_id)
            args[a_id] = a_text

            if len(arg_ids) == 0:
                break
    print(args)
    sys.exit(0)

    for word in words:
        print(f'Wort: {word}')
        if word in model:
            most_sim = model.most_similar(word)
            print(f"Am ähnlichsten zu \"{word}\" -> {most_sim}\n")
        else:
            print(f'Word \"{word}\" ist nicht im Vokabular')
