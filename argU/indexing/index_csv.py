# Index CSV
# Die CSV enthält alle Infos für BM25
# Und die Vektor-Embeddings aller Argumente

import argparse
import os
import rootpath
import sys
import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from indexing.models import BM25Manager, CBOW, DualEmbedding
from utils.reader import TrainCSVIterator, Argument
from utils.utils import path_not_found_exit
from utils.beautiful import print_argument_texts

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
TRAIN_PATH = os.path.join(RESOURCES_PATH, 'cbow_train.csv')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
BM25_PATH = os.path.join(RESOURCES_PATH, 'bm25.json')
CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
INDEX_PATH = os.path.join(RESOURCES_PATH, 'index.csv')

path_not_found_exit([
    RESOURCES_PATH, TRAIN_PATH, CSV_PATH,
    BM25_PATH, CBOW_MODEL_PATH, INDEX_PATH,
])


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['load', 'train'])
args = parser.parse_args()


def create_index(csv_path, source_path, cbow_model, bm25_model, max_rows=-1):
    """Erstelle einen neuen Index, der in einer CSV gespeichert wird

    Args:
        csv_path (:obj:`str`): Dateipfad für die CSV
    """

    vector_size = cbow_model.vector_size

    with open(csv_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(
            f_out,
            delimiter='|',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        for row in tqdm(TrainCSVIterator(source_path, max_rows=max_rows)):
            arg_id, arg_text = row
            index = bm25_model.arg_ids.index(arg_id)
            doc_frec = bm25_model.doc_freqs[index]
            doc_len = bm25_model.doc_len[index]
            arg_emb, unk = Argument.to_vec(arg_text, cbow.model, vector_size)

            writer.writerow([
                arg_id,
                json.dumps(doc_frec),
                arg_emb.tolist(),
                unk,
                doc_len,
            ])


def analyze_query(query, index_csv_path, cbow_model, bm25_model,
                  alpha=0.5, max_args=-1, top_n=15):
    """ Lade den Index und werte die Argumente abhängig vond er Query aus

    Args:
        query (:obj:`str`): Eingabe, nach der gescuht werden soll
        index_csv_path (:obj:`str`): Dateipfad zum Index CSV
        cbow_model (:obj:`Word2Vec`): Word Embedding Modell
        bm25_model (:obj:`BM25Kapi`): BM25 Index
        alpha (float): Einfluss BM25 und Word Embedding
        max_args (int): Menge der Argumente
        top_n (int): Wie viele Argumente sollen zurückgegeben werden
    """

    # love titties 4 eva
    # booty squad <333333333
    # ariane staudte <333333

    query = query.split()

    del(bm25_model.doc_freqs)
    del(bm25_model.arg_ids)
    del(bm25_model.doc_len)

    dual_embedding = DualEmbedding(cbow_model)
    dual_embedding.build()

    bm25_scores = []
    desim_scores = []
    arg_ids = []

    with open(INDEX_PATH, 'r', encoding='utf-8', newline='') as f_in:
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
                desim_scores.append(
                    dual_embedding.desim(query, arg_emb)
                )
                bm25_scores.append(
                    bm25_manager.index.get_single_score(
                        query,
                        doc_freq,
                        doc_len,
                    )
                )
                arg_ids.append(
                    arg_id
                )
            except Exception as e:
                print(arg_id, arg_emb)

            if i + 1 == max_args:
                break

        print('Normalisieren...')
        bm25_scores = preprocessing.normalize([bm25_scores])[0]
        desim_scores = preprocessing.normalize([desim_scores])[0]
        final_scores = []

        print('Finale Scores berechnen...')
        for b, d, a in zip(bm25_scores, desim_scores, arg_ids):
            final_scores.append(alpha * b + (1 - alpha) * d)

        print('Top-N Bestimmung...')
        top_n = np.argsort(final_scores)[::-1][:top_n]
        top_args = [arg_ids[i] for i in top_n]
        print(top_args)


cbow = CBOW.load(CBOW_MODEL_PATH)
bm25_manager = BM25Manager.load(BM25_PATH)

if args.mode == 'train':
    create_index(INDEX_PATH, TRAIN_PATH, cbow.model, bm25_manager.index)
elif args.mode == 'load':
    analyze_query(
        'Donald Trump is bad',
        INDEX_PATH,
        cbow.model,
        bm25_manager.index,
        alpha=0.5,
        max_args=-1
    )
