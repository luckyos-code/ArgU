
import argparse
import os
import sys
import csv
import rootpath

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from preprocessing.generate_train_csv import generate_cbow_train_file
from indexing.models import CBOW, BM25Manager
from indexing.index_csv import create_index, analyze_queries, combine_scores, get_top_args, sentiment_sort_args
from utils.reader import TrainCSVIterator, FindArgumentIterator
from utils import queries

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_ARGS_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODELS_TRAIN_PATH = os.path.join(RESOURCES_PATH, 'train.csv')
CBOW_STORE_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')
BM25_STORE_PATH = os.path.join(RESOURCES_PATH, 'bm25.json')
INDEX_STORE_PATH = os.path.join(RESOURCES_PATH, 'index.csv')
SENTIMENTS_PATH = os.path.join(
    ROOT_PATH, 'argU/sentiment/results/argument_sentiments.csv'
)
FOUND_ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, 'results.csv')
QUERIES_PATH = os.path.join(RESOURCES_PATH, 'topics-automatic-runs-task-1.xml')

bm_25_splits = BM25_STORE_PATH.split('.')
BM25_PARTIONAL_PATHS = [
    '.'.join([*bm_25_splits[:-1], 'meta', bm_25_splits[-1]]),
    '.'.join([*bm_25_splits[:-1], 'args', bm_25_splits[-1]]),
]

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['run', 'create'])
parser.add_argument('-c', '--create',
                    choices=['all', 'bm25', 'cbow', 'index'], default='all')
args = parser.parse_args()

if not os.path.isfile(CSV_ARGS_PATH):
    print('Die Argumente müssen als CSV-Format vorliegen...')
    sys.exit(0)

if args.mode == 'create':
    if args.create in ['cbow', 'all']:

        # Erstelle das CBOW Modell
        # Dazu muss vorerst eine Trainings-Datei generiert werden

        if not os.path.isfile(MODELS_TRAIN_PATH):
            print("Erstelle Trainingsdatei für das CBOW-Modell...")
            generate_cbow_train_file(CSV_ARGS_PATH, MODELS_TRAIN_PATH)

        print("Erstelle das CBOW-Modell...")
        cbow = CBOW()
        cbow.build(
            TrainCSVIterator(MODELS_TRAIN_PATH, only_texts=True),
            store_path=CBOW_STORE_PATH,
            min_count=5,
        )
        cbow.store(CBOW_STORE_PATH)
        del(cbow)

    if args.create in ['bm25', 'all']:

        # Erstelle das BM25 Modell
        # Dazu muss vorerst eine Trainings-Datei generiert werden

        if not os.path.isfile(MODELS_TRAIN_PATH):
            print("Erstelle Trainingsdatei für das BM25-Modell...")
            generate_cbow_train_file(CSV_ARGS_PATH, MODELS_TRAIN_PATH)

        print("Erstelle das BM25-Modell...")
        bm25_manager = BM25Manager()
        bm25_manager.build(TrainCSVIterator(MODELS_TRAIN_PATH, max_rows=-1))
        bm25_manager.store(BM25_STORE_PATH)
        del(bm25_manager)

    if args.create in ['index', 'all']:

        # Erstelle den Index aus dem BM25-Modell und dem CBOW-Modell
        # Teste vorher, ob alle Modelle existieren!

        if not os.path.isfile(CBOW_STORE_PATH):
            print("Das CBOW-Modell existiert noch nicht...")
            sys.exit(0)

        for p in BM25_PARTIONAL_PATHS:
            if not os.path.isfile(p):
                print("Das BM25-Modell existiert noch nicht...")
                sys.exit(0)

        print("Erstelle eine Index Datei...")
        cbow = CBOW.load(CBOW_STORE_PATH)
        bm25_manager = BM25Manager.load(BM25_STORE_PATH)
        create_index(
            INDEX_STORE_PATH,
            MODELS_TRAIN_PATH,
            cbow.model,
            bm25_manager.index,
            max_rows=-1,
        )
        del(cbow)
        del(bm25_manager)

if args.mode == 'run':

    # Teste ob die Pfade für CBOW und BM25 existieren

    for p in BM25_PARTIONAL_PATHS:
        if not os.path.isfile(p):
            print("Das BM25-Modell existiert noch nicht...")
            sys.exit(0)

    if not os.path.isfile(CBOW_STORE_PATH):
        print("Das CBOW-Modell existiert noch nicht...")
        sys.exit(0)

    # Teste ob die Sentiment Werte für die Argumente existieren

    if not os.path.isfile(SENTIMENTS_PATH):
        print("Die Sentiments existieren noch nicht...")
        sys.exit(0)

    # Lade BM25 (Meta) und CBOW

    bm25_manager = BM25Manager.load(BM25_STORE_PATH, mode='meta')
    cbow = CBOW.load(CBOW_STORE_PATH)

    query_ids, query_texts = queries.read(QUERIES_PATH, start=0, stop=1)
    query_texts = queries.clean(query_texts)

    print('\n', query_texts, '\n')

    bm25_scores, desim_scores, arg_ids = analyze_queries(
        query_texts,
        INDEX_STORE_PATH,
        cbow.model,
        bm25_manager.index,
        max_args=1000,
    )

    # Bestimme den finalen Score
    # Dazu werden beide Scores paarweise zusammengerechnet
    # Zusätzlich kommen Sentiment Score ins Spiel, der zur Sortierung dient

    alpha = 0.1
    top_args = get_top_args(
        arg_ids,
        bm25_scores,
        desim_scores,
        alpha=alpha,
        top_n=200,
    )

    sentiment_sorted_args = sentiment_sort_args(SENTIMENTS_PATH, top_args)

    # Speichere die gefundenen Argumente für eine Query in einem Log

    if not os.path.isfile(FOUND_ARGUMENTS_PATH):
        result_log_header = ['id', 'query', 'top_args', 'scores', 'alpha']
        with open(FOUND_ARGUMENTS_PATH, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(result_log_header)

    with open(FOUND_ARGUMENTS_PATH, 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(
            f_out,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        for id, query, (arg_ids, arg_scores) in zip(query_ids, query_texts, sentiment_sorted_args):
            line = [id, query, arg_ids, arg_scores, alpha]
            writer.writerow(line)

    # Speichere Argumente in dem passenden Output Format
    # TODO
