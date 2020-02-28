
import argparse
import os
import sys
import csv
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
    default='None',
    choices=['None', 'neutral', 'emotional']
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
        merged_args = merged_args[:20]

        if argparsed.sentiments != 'None':
            merged_args_with_sents = []
            for ma in merged_args:
                dph, sent = ma[1], ma[2]
                if argparsed.sentiments == 'emotional':
                    dph = dph + dph * (abs(sent) / 2)
                elif argparsed.sentiments == 'neutral':
                    dph = dph - dph * (abs(sent) / 2)
                merged_args_with_sents.append(
                    (ma[0], dph, sent)
                )
            merged_args = merged_args_with_sents
            merged_args.sort(key=lambda x: x[1], reverse=True)

        print(f'### {query_id} {desm_scores["query_text"]}')
        # print('---')
        # arguments.fancy_print(
        #     coll_args,
        #     merged_args_list[:20],
        #     trans_dict=trans_dict,
        #     arg_len=2000,
        # )

        # Sentiment Analysis

        output_dict[query_id] = merged_args

    with open(os.path.join(argparsed.output, 'run.txt'), 'w') as f_out:
    	mehtod = setup.METHOD_NONE
        if argparsed.sentiments == 'emotional':
            mehtod = setup.METHOD_EMOTIONAL
        elif argparsed.sentiments == 'neutral':
            mehtod = setup.METHOD_NEUTRAL
        for (id, args) in output_dict.items():
            for i, (arg_id, score, sent) in enumerate(args):
                f_out.write(' '.join([
                    str(id), 'Q0', coll_trans.find_one(
                        {'_id': arg_id})['arg_id'], str(i + 1),
                    str(score), method, '\n'
                ]))


# for i, res in enumerate(coll_res.find()):
#     if i == 3:
#         break
#     args = arguments.find(coll_args, res['args'])
#     print(res['query_text'])
#     print('=' * 40)
#     for a in args:
#         print('> ', Argument.get_text(a)[:200])
#     print()

    # if args.mode == 'retrieve' or args.mode == 'collect':

    #     # Speichere Argumente in dem passenden Output Format
    #     queries_args = scores.evaluate(threshold=0.5)

    #     with open(setup.RUN_PATH, 'w') as f_out:
    #         for (query_id, query_text, args) in queries_args:
    #             for i, arg in enumerate(args):
    #                 f_out.write(' '.join([
    #                     query_id,
    #                     'Q0',
    #                     arg[0],
    #                     str(i + 1),
    #                     str(arg[1]),
    #                     method,
    #                     '\n',
    #                 ]))
