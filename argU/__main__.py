import argparse
import os
import sys

import rootpath

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.utils import queries as Q
    from argU.utils import arguments
    from argU.utils.arguments import Argument
    from argU.indexing.models import CBOW
    from argU.indexing.models import DESM
    from argU.preprocessing.mongodb import load_db
    from argU.preprocessing.mongodb import collection_exists
    from argU.preprocessing.mongodb import new_id_to_args_id_dict
except Exception as e:
    print(e)
    sys.exit(0)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d', '--desm',
    action='store_true',
    help='Reset query results',
)

parser.add_argument(
    '-m', '--merge',
    action='store_true',
    help='Merge Terrier scores with query scores',
)

parser.add_argument(
    '-i', '--input',
    help='Input directory',
    default=setup.ROOT_PATH,
)

parser.add_argument(
    '-o', '--output',
    help='Output directosy',
    default=setup.OUTPUT_PATH,
)

parser.add_argument(
    '-s', '--sentiments',
    help='Sentiments mode',
    default='no',
    choices=['no', 'neutral', 'emotional']
)

argparsed = parser.parse_args()
print(f"Args: {argparsed}")

setup.assert_file_exists(setup.CBOW_PATH)

db = load_db()
coll_args = db[setup.MONGO_DB_COL_ARGS]
coll_emb = db[setup.MONGO_DB_COL_EMBEDDINGS]
coll_res = db[setup.MONGO_DB_COL_RESULTS]
coll_trans = db[setup.MONGO_DB_COL_TRANSLATION]
coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]
print(f'Embedded Args: {coll_emb.count_documents({})}')

queries = Q.read(argparsed.input)

if argparsed.desm:
    desm = DESM(CBOW.load())
    top_args = desm.evaluate_queries(
        desm.queries_to_emb(queries),
        coll_emb,
        top_n=4000,
        max_args=-1,
    )
    desm.store_query_results(coll_res, queries, top_args)

if argparsed.merge:
    assert collection_exists(coll_res)
    assert setup.file_exists(setup.TERRIER_RESULTS_PATH)

    N = 1000
    print(f'N Value: {N}')

    max_queries = -1

    output_dict = dict()
    for i, desm_scores in enumerate(coll_res.find()):
        if i == max_queries:
            break

        query_id = desm_scores['query_id']
        args = desm_scores['args'][:N]

        terrier_data = dict()
        with open(setup.TERRIER_RESULTS_PATH, 'r') as f_in:
            for line in f_in:
                line = line.split()
                if line[0] == query_id:
                    terrier_data[int(line[2])] = line[4]  # Arg
        merged_args = []
        for a in args:
            if a in terrier_data:
                sents = coll_sents.find_one({'_id': a})
                if sents is None:
                    print("BAD================================")
                    sents = {'score': -0.1}
                merged_args.append(
                    (a, float(terrier_data[a]), sents['score'])
                )

        merged_args.sort(key=lambda x: x[1], reverse=True)

        if argparsed.sentiments != 'no':
            merged_args_with_sents = []
            for ma in merged_args:
                dph, sent = ma[1], ma[2]
                if argparsed.sentiments == 'emotional':
                    dph = dph + dph * (abs(sent))
                elif argparsed.sentiments == 'neutral':
                    dph = dph - dph * (abs(sent) / 2)
                merged_args_with_sents.append(
                    (ma[0], dph, sent)
                )
            merged_args = merged_args_with_sents
            merged_args.sort(key=lambda x: x[1], reverse=True)

        print(f'### {query_id} {desm_scores["query_text"]}')

        if len(merged_args) != 0:
            output_dict[query_id] = merged_args
        else:
            output_dict[query_id] = [(
                '10113b57-2019-04-18T17:05:08Z-00001-000',
                0.0,
                0.0,
            )]

    with open(os.path.join(argparsed.output, 'run.txt'), 'w') as f_out:
        method = setup.METHOD_NO
        if argparsed.sentiments == 'emotional':
            method = setup.METHOD_EMOTIONAL
        elif argparsed.sentiments == 'neutral':
            method = setup.METHOD_NEUTRAL
        for (id, args) in output_dict.items():
            for i, (arg_id, score, sent) in enumerate(args):
                trans_id = ''
                try:
                    trans_id = coll_trans.find_one({'_id': arg_id})['arg_id']
                except:
                    trans_id = arg_id

                f_out.write(' '.join([
                    str(id), 'Q0', trans_id, str(i + 1), str(score), method, '\n'
                ]))
