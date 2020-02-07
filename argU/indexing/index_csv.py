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
import math
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from indexing.models import BM25Manager, CBOW, DualEmbedding
from utils.reader import TrainCSVIterator, Argument
from utils.utils import path_not_found_exit
from utils.beautiful import print_argument_texts

warnings.filterwarnings("error")

# RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
# TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')
# CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
# BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.json')
# CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
# INDEX_PATH = os.path.join(RESOURCES_PATH, 'index.csv')
# RESULT_LOG_PATH = os.path.join(RESOURCES_PATH, 'result.log.csv')

# path_not_found_exit([
#     RESOURCES_PATH, TRAIN_PATH, CSV_PATH,
#     BM25_PATH, CBOW_MODEL_PATH, INDEX_PATH,
# ])


# parser = argparse.ArgumentParser()
# parser.add_argument('mode', choices=['load', 'train', 'read'])
# args = parser.parse_args()


def create_index(index_path, train_path, cbow_model, bm25_model, max_rows=-1):
    """Erstelle einen neuen Index, der in einer CSV gespeichert wird

    Args:
        index_path (:obj:`str`): Speicherort Index
        train_path (:obj:`str`): Trainingsdatei für Modelle
        cbow_model (:obj:`Word2Vec`): Embedding Modell
        bm25_model (:obj:`BM25Kapi`): BM25 Modell
        max_rows (int, optional): Anzahl der Argumente, die indiziert
            werden sollen
    """

    vector_size = cbow_model.vector_size

    with open(index_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(
            f_out,
            delimiter='|',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        for row in tqdm(TrainCSVIterator(train_path, max_rows=max_rows)):
            arg_id, arg_text = row
            index = bm25_model.arg_ids.index(arg_id)
            doc_frec = bm25_model.doc_freqs[index]
            doc_len = bm25_model.doc_len[index]
            arg_emb, unk = Argument.to_vec(arg_text, cbow_model, vector_size)

            writer.writerow([
                arg_id,
                json.dumps(doc_frec),
                arg_emb.tolist(),
                unk,
                doc_len,
            ])


def analyze_queries(queries, index_path, cbow_model, bm25_model,
                    max_args=-1):
    """ Lade den Index und werte die Argumente abhängig vond er Query aus

    Args:
        queries (:obj:`list` of :obj:`str`): Eingaben, nach denen gesucht werden soll
        index_csv_path (:obj:`str`): Dateipfad zum Index CSV
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
    dual_embedding.build()

    bm25_scores = []
    desim_scores = []
    arg_ids = []

    with open(index_path, 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in, delimiter='|',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i, row in tqdm(enumerate(reader)):
            arg_id, doc_freq, arg_emb, unk, doc_len = [
                row[0],
                json.loads(row[1]),
                np.asarray(json.loads(row[2])),
                row[3],
                int(row[4])
            ]

            try:
                desim_query_scores = []
                bm25_query_scores = []

                for query in queries:
                    query = query.split()
                    desim_query_scores.append(
                        dual_embedding.desim(query, arg_emb)
                    )
                    bm25_query_scores.append(
                        bm25_model.get_single_score(
                            query,
                            doc_freq,
                            doc_len,
                        )
                    )
                arg_ids.append(arg_id)
                bm25_scores.append(bm25_query_scores)
                desim_scores.append(desim_query_scores)

            except RuntimeWarning as e:
                print(arg_id, e)

            if i + 1 == max_args:
                break

        # bm25_scores = np.transpose(np.array(bm25_scores))
        # desim_scores = np.transpose(np.array(desim_scores))
        bm25_scores = np.array(bm25_scores)
        desim_scores = np.array(desim_scores)
        arg_ids = np.asarray(arg_ids)

        print('BM25 Scores: ', bm25_scores.shape)
        print('Desim Scores: ', desim_scores.shape)
        print('Arg IDs: ', arg_ids.shape)
        print()
        print(f'Min BM25: {bm25_scores.min()}')
        print(f'Max BM25: {bm25_scores.max()}')
        print(f'Min Desim: {desim_scores.min()}')
        print(f'Max Desim: {desim_scores.max()}')

        bm25_norm = np.linalg.norm(bm25_scores, axis=0)
        bm25_norm[bm25_norm == 0] = 0.0001
        desim_norm = np.linalg.norm(desim_scores, axis=0)
        desim_norm[desim_norm == 0] = 0.0001

        bm25_scores = np.transpose(bm25_scores / bm25_norm)
        desim_scores = np.transpose(desim_scores / desim_norm)

        print(f'norm Min BM25: {bm25_scores.min()}')
        print(f'norm Max BM25: {bm25_scores.max()}')
        print(f'norm Min Desim: {desim_scores.min()}')
        print(f'norm Max Desim: {desim_scores.max()}')

        print(f'Shape BM25: {bm25_scores.shape}')
        print(f'Shape Desim: {desim_scores.shape}\n')

        return (bm25_scores, desim_scores, arg_ids)


def combine_scores(bm25_scores, desim_scores, alpha):
    assert bm25_scores.shape == desim_scores.shape
    bms = alpha * bm25_scores
    desims = (1 - alpha) * desim_scores
    # np.set_printoptions(threshold=sys.maxsize)
    # print(bms)
    # print(desims)
    # print(bms < desims)
    # relation = sum(bms > desims) / (bms.shape[0] * bms.shape[1])
    # print(f"Prozentualer Anteil von BMs, die größer als Desim sind: {relation}")
    return bms + desims


def get_top_args(combined_scores, arg_ids, top_n=10):
    results = []
    for query_scores in combined_scores:
        top_ids = np.argsort(query_scores)[::-1][:top_n]
        results.append(
            (arg_ids[top_ids], query_scores[top_ids])
        )
    return results


def sentiment_sort_args(SENTIMENTS_PATH, top_args):
    with open(SENTIMENTS_PATH, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.reader(
            f_in,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
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
            sentiment_scores.append(sentiments.get(id, (0, 0))[0])
            sentiment_magnitudes.append(sentiments.get(id, (0, 0))[1])
        query_sentiments.append(
            (sentiment_scores, sentiment_magnitudes)
        )

    results = []
    for arg_ids, query_sents in zip(top_args, query_sentiments):
        abs_sent = [abs(i) for i in query_sents[0]]
        pairings = [(i, p[0], p[1]) for i, p in enumerate(zip(arg_ids[1], abs_sent))]
        pairings = sorted(pairings, key=lambda x: (x[2], x[1]), reverse=True)
        indices_order = [i[0] for i in pairings]
        new_arg_ids = [arg_ids[0][i] for i in indices_order]
        new_arg_scores = [(p[1], p[2]) for p in pairings]
        results.append(
            (new_arg_ids, new_arg_scores)
        )
    return results

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
