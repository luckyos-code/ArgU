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
    from argU.utils.reader import Argument
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)

if __name__ == '__main__':
    cbow = CBOW.load()

    model_in = cbow.model
    dual_embedding_model = DualEmbedding(model_in)

    queries = ['Instructor Money']
    arg_ids = set(['c065954f-2019-04-18T14:32:52Z-00001-000'])

    args = dict()
    for a_id, a_text in TrainArgsIterator():
        if a_id in arg_ids:
            arg_ids.remove(a_id)
            args[a_id] = (a_text, Argument.to_vec(
                a_text, model_in, model_in.vector_size)[0]
            )

            if len(arg_ids) == 0:
                break

    processed_queries = dual_embedding_model.get_processed_queries(queries)

    print()
    for arg_id, (arg_text, arg_emb) in args.items():
        print(arg_id)
        print(model_in.wv.similar_by_vector(
            arg_emb, topn=10, restrict_vocab=None)
        )
    print()

    for query_terms, query_matrix in processed_queries:
        for arg_id, (arg_text, arg_emb) in args.items():
            score = dual_embedding_model.desim(query_matrix, arg_emb)
            print(f'Arg: {arg_id}, Query: {query_terms} --> {score}')
        # for word in words:
        #     print(f'Wort: {word}')
        #     if word in model:
        #         most_sim = model.most_similar(word)
        #         print(f"Am ähnlichsten zu \"{word}\" -> {most_sim}\n")
        #     else:
        #         print(f'Word \"{word}\" ist nicht im Vokabular')
