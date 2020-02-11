# Index CSV
# Die CSV enthält alle Infos für BM25
# Und die Vektor-Embeddings aller Argumente

import argparse
import os
import rootpath
import sys
import json
import csv
import warnings
import traceback
import math
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize

warnings.filterwarnings("error")

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.indexing.models import DualEmbedding
    from argU.utils.beautiful import print_argument_texts
    from argU.utils.reader import TrainArgsIterator
    from argU.utils.reader import Argument
    from argU.utils.utils import path_not_found_exit
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    traceback.print_exc(file=sys.stdout)
    sys.exit(0)


def create(cbow_model, bm25_model, max_args=-1):
    """Erstelle einen neuen Index, der in einer CSV gespeichert wird

    Args:
        cbow_model (:obj:`Word2Vec`): Embedding Modell
        bm25_model (:obj:`BM25Kapi`): BM25 Modell
        max_args (int, optional): Anzahl der Argumente, die indiziert
            werden sollen
    """

    vector_size = cbow_model.vector_size

    with open(setup.INDEX_PATH, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, **setup.INDEX_CONFIG)

        for row in tqdm(TrainArgsIterator(max_args=max_args)):
            arg_id, arg_text = row
            bm25_index = bm25_model.arg_ids.index(arg_id)
            doc_frec = bm25_model.doc_freqs[bm25_index]
            doc_len = bm25_model.doc_len[bm25_index]

            arg_emb, unk = Argument.to_vec(arg_text, cbow_model, vector_size)

            writer.writerow([
                arg_id,
                json.dumps(doc_frec),
                [round(i, 14) for i in arg_emb.tolist()],
                doc_len,
            ])


def collect_arguments(queries, cbow_model, bm25_model, max_args=-1):
    """ Lade den Index und werte die Argumente abhängig vond er Query aus

    Args:
        queries (:obj:`list` of :obj:`str`): Eingaben, nach denen gesucht werden soll
        cbow_model (:obj:`Word2Vec`): Word Embedding Modell
        bm25_model (:obj:`BM25Kapi`): BM25 Index
        alpha (float): Einfluss BM25 und Word Embedding
        max_args (int): Menge der Argumente

    Returns:
        :obj:`tuple` of two :obj:`np.array`: erster array 2 dim. BM25 scores,
            zweiter array 2 dim. Dualsim scores 
    """

    # love titties 4 eva
    # booty squad <333333333
    # ariane staudte <333333

    dual_embedding = DualEmbedding(cbow_model)
    processed_queries = dual_embedding.get_processed_queries(queries)

    bm25_scores = []
    desim_scores = []
    arg_ids = []

    with open(setup.INDEX_PATH, 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in, **setup.INDEX_CONFIG)

        for i, row in tqdm(enumerate(reader)):
            try:
                arg_id, doc_freq, arg_emb, doc_len = [
                    row[0],
                    json.loads(row[1]),
                    np.asarray(json.loads(row[2])),
                    int(row[3])
                ]
            except Exception as e:
                print(e)
                print("\t> Folgende Zeile konnte nicht verarbeitet werden:")
                print("\t>", row)
                continue

            try:
                desim_queries_scores = []
                bm25_queries_scores = []

                for query_terms, query_matrix in processed_queries:

                    desim_query_score = dual_embedding.desim(
                        query_matrix, arg_emb
                    )
                    bm25_query_score = bm25_model.get_single_score(
                        query_terms,
                        doc_freq,
                        doc_len,
                    )

                    desim_queries_scores.append(desim_query_score)
                    bm25_queries_scores.append(bm25_query_score)

                arg_ids.append(arg_id)
                bm25_scores.append(bm25_queries_scores)
                desim_scores.append(desim_queries_scores)

            except RuntimeWarning as e:
                print(arg_id, e)

            if i + 1 == max_args:
                break

        # bm25_scores = np.transpose(np.array(bm25_scores))
        # desim_scores = np.transpose(np.array(desim_scores))
        bm25_scores = np.array(bm25_scores)
        desim_scores = np.array(desim_scores)
        arg_ids = np.asarray(arg_ids)

        print('\nBM25 Scores: ', bm25_scores.shape)
        print('Desim Scores: ', desim_scores.shape)
        print('Arg IDs: ', arg_ids.shape, '\n')
        print(f'Min BM25: {bm25_scores.min()}')
        print(f'Max BM25: {bm25_scores.max()}')
        print(f'Min Desim: {desim_scores.min()}')
        print(f'Max Desim: {desim_scores.max()}\n')

        bm25_norm = normalize(bm25_scores, norm='max', axis=0)
        desim_norm = normalize(desim_scores, norm='max', axis=0)

        # bm25_norm = np.linalg.norm(bm25_scores, axis=0)
        # bm25_norm[bm25_norm == 0] = 0.0001
        # desim_norm = np.linalg.norm(desim_scores, axis=0)
        # desim_norm[desim_norm == 0] = 0.0001

        # bm25_scores = np.transpose(bm25_scores / bm25_norm)
        # desim_scores = np.transpose(desim_scores / desim_norm)

        print(f'norm Min BM25: {bm25_norm.min()}')
        print(f'norm Max BM25: {bm25_norm.max()}')
        print(f'norm Min Desim: {desim_norm.min()}')
        print(f'norm Max Desim: {desim_norm.max()}\n')

        print(f'Shape BM25: {bm25_norm.shape}')
        print(f'Shape Desim: {desim_norm.shape}\n')

        bm25_norm = np.transpose(bm25_norm)
        desim_norm = np.transpose(desim_norm)

        return arg_ids, bm25_norm, desim_norm


def combine_scores(bm25_scores, desim_scores, alpha):
    assert bm25_scores.shape == desim_scores.shape
    bms = alpha * bm25_scores
    desims = (1 - alpha) * desim_scores
    influences = (bms >= desims)
    return bms + desims, influences.flatten().tolist()


def get_top_args(arg_ids, bm25_scores, desim_scores, alpha=0.5, top_n=10):

    top_args_list = []
    final_scores, influences = combine_scores(bm25_scores, desim_scores, alpha)

    print((
        f"Influence Statistics\n"
        f"--------------------\n\n"
        f"BM25 > Desim: {influences.count(True)} Mal.\n"
        f"Desim > BM25: {influences.count(False)} Mal.\n\n"
        # f"Desim und BM25 sollten standardmäßig ähnlich sein (Verhältnis = 1.0)\n"
        # f"Verhältnis: {abs(influences.count(True) / influences.count(False))}\n\n"
    ))

    for bs, ds, fs in zip(bm25_scores, desim_scores, final_scores):
        top_ids = np.argsort(fs)[::-1][:top_n]
        top_args_list.append(
            (
                arg_ids[top_ids],
                fs[top_ids],
                bs[top_ids],
                ds[top_ids],
            )
        )

    return top_args_list


def get_sentiments(top_args):
    with open(
        setup.SENTIMENTS_PATH, 'r', newline='', encoding='utf-8'
    ) as f_in:
        reader = csv.reader(f_in, **setup.SENTIMENTS_CONFIG)
        header = next(reader)

        sentiments = {}
        for line in reader:
            arg, score, magn = line
            sentiments[arg] = (float(score), float(magn))

    query_sentiments = []
    for arg_ids in top_args:
        sentiment_scores = []
        sentiment_magnitudes = []
        for id in arg_ids[0]:
            s = sentiments.get(id, (0, 0))
            sentiment_scores.append(s[0])
            sentiment_magnitudes.append(s[1])
        query_sentiments.append(
            (sentiment_scores, sentiment_magnitudes)
        )

    return query_sentiments

    # results = []
    # for arg_ids, query_sents in zip(top_args, query_sentiments):
    #     abs_sent = [abs(i) for i in query_sents[0]]
    #     pairings = [(i, p[0], p[1])
    #                 for i, p in enumerate(zip(arg_ids[1], abs_sent))]
    #     pairings = sorted(pairings, key=lambda x: (x[2], x[1]), reverse=True)
    #     indices_order = [i[0] for i in pairings]
    #     new_arg_ids = [arg_ids[0][i] for i in indices_order]
    #     new_arg_scores = [(p[1], p[2]) for p in pairings]
    #     results.append(
    #         (new_arg_ids, new_arg_scores)
    #     )
    # return results

# if args.mode != 'read':
#     cbow = CBOW.load(CBOW_MODEL_PATH)
#     bm25_manager = BM25Manager.load(BM25_PATH)

# if args.mode == 'train':
#     create_index(INDEX_PATH, TRAIN_PATH, cbow.model, bm25_manager.index)

# elif args.mode == 'load':

#     queries = [
#         'Donald Trump is bad',
#         'pregnant women abortion stop',
#     ]

#     bm25_scores, desim_scores, arg_ids = analyze_query(
#         queries,
#         INDEX_PATH,
#         cbow.model,
#         bm25_manager.index,
#         max_args=-1
#     )

#     combined_scores = []
#     alpha = 0.5

#     for bm25_query_scores, desim_query_scores in zip(bm25_scores, desim_scores):
#         combined_query_scores = []

#         for b, d in zip(bm25_query_scores, desim_query_scores):
#             combined_query_scores.append(alpha * b + (1 - alpha) * d)

#         combined_scores.append(combined_query_scores)

#     combined_scores = np.asarray(combined_scores)
#     top_n = 10
#     results = []

#     for cs in combined_scores:
#         best_n = np.argsort(cs)[::-1][:top_n]
#         top_args = arg_ids[best_n]
#         top_scores = cs[best_n]

#         results.append((top_args, top_scores))

#     if not os.path.isfile(RESULT_LOG_PATH):
#         result_log_header = ['query', 'top_args', 'scores', 'alpha']
#         with open(RESULT_LOG_PATH, 'w', newline='', encoding='utf-8') as f_out:
#             writer = csv.writer(f_out, delimiter='|', quotechar='"',
#                                 quoting=csv.QUOTE_MINIMAL)
#             writer.writerow(result_log_header)

#     with open(RESULT_LOG_PATH, 'a', newline='', encoding='utf-8') as f_out:
#         writer = csv.writer(f_out, delimiter='|',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)

#         for q, res in zip(queries, results):

#             row = [
#                 q,
#                 res[0].tolist(),
#                 res[1].tolist(),
#                 alpha,
#             ]
#             writer.writerow(row)

# elif args.mode == 'read':
#     with open(RESULT_LOG_PATH, 'r', newline='', encoding='utf-8') as f_in:
#         reader = csv.reader(f_in, delimiter='|', quotechar='"',
#                                 quoting=csv.QUOTE_MINIMAL)

#         header = next(reader)
#         for line in reader:
#             query, best_args = line[0], line[1]
#             print(query)
#             print(best_args)
#             print_argument_texts(best_args, CSV_PATH, print_full_texts=True)
#             print('*' * 60)
