
import argparse
import os
import sys
import csv
import rootpath

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.train import generate_train_file
    from argU.indexing.models import CBOW
    from argU.indexing.models import BM25Manager
    from argU.indexing import index
    from argU.utils import queries
    from argU.utils import scores
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)

DEBUG = True

if DEBUG:
    train_args = 20000
    method = 'ulT1DetroitnitzCbowBm25Sentiments'
else:
    train_args = -1
    method = 'method'

# from indexing.index_csv import create_index, analyze_queries, combine_scores, get_top_args, get_sentiments
# from utils.reader import TrainCSVIterator, FindArgumentIterator
# from utils import queries, scores

parser = argparse.ArgumentParser()

parser.add_argument(
    'mode',
    choices=['index', 'retrieve', 'collect'],
    help="Erstelle einen Index oder erhalte Argumente",
)

parser.add_argument(
    '-n', '--max_args',
    default=-1, type=int,
    help='Menge der verwendeten Argumente. -1 = Alle Argumente',
)

parser.add_argument(
    '-a', '--alpha',
    default=0.5, type=float,
    help='Alpha Einfluss BM25 und CBOW',
)

parser.add_argument(
    '-c', '--create',
    choices=['all', 'bm25', 'cbow', 'indexlist'],
    default='all',
)

parser.add_argument(
    '-q', '--query_range',
    nargs='+', type=int,
    default=[0, 1000],
)

parser.add_argument(
    '-d', '--deep_clean',
    action='store_true',
)

parser.add_argument(
    '-v', '--visualize',
    action='store_true',
)

args = parser.parse_args()
print(f"Args: {args}")

# if len(args.query_range) != 2 or args.query_range[0] < 0 or args.query_range[0] >= args.query_range[1]:
#     print("Query Range fehlerhaft...")
#     sys.exit(0)

setup.assert_file_exists(setup.ARGS_ME_CSV_PATH)

if 'index' in args.mode:

    bm25_manager = BM25Manager()
    cbow = CBOW()

    if args.create in ['cbow', 'all']:

        # Erstelle das CBOW Modell

        if not os.path.isfile(setup.TRAIN_ARGS_PATH):
            print("Generate train file...")
            generate_train_file(max_args=train_args)

        print("Generate CBOW model...")
        cbow.build(min_count=5, size=100, window=7)
        cbow.store()

    if args.create in ['bm25', 'all']:

        # Erstelle das BM25 Modell

        if not os.path.isfile(setup.TRAIN_ARGS_PATH):
            print("Generate train file...")
            generate_train_file(max_args=-1)

        print("Erstelle das BM25-Modell...")
        bm25_manager.build(max_args=-1)
        bm25_manager.store()

    if args.create in ['indexlist', 'all']:

        # Erstelle den Index aus dem BM25-Modell und dem CBOW-Modell
        # Teste vorher, ob alle Modelle existieren!

        setup.assert_file_exists(setup.CBOW_PATH)
        setup.assert_file_exists(setup.BM25_META_PATH)
        setup.assert_file_exists(setup.BM25_DOCS_PATH)

        print("Erstelle eine Index Datei...")

        if not cbow.loaded:
            cbow = CBOW.load()
        if not bm25_manager.loaded:
            print("not loaded")
            bm25_manager = BM25Manager.load()

        index.create(
            cbow.model,
            bm25_manager.index,
            max_args=-1,
        )

        if args.deep_clean:
            os.remove(setup.TRAIN_ARGS_PATH)
            os.remove(setup.BM25_DOCS_PATH)

    del(cbow)
    del(bm25_manager)


if args.mode == 'retrieve':

    # Bestimme Argumente für die Testqueries
    # Teste vorher, ob die passenden Dateien existieren

    setup.assert_file_exists(setup.BM25_META_PATH)
    setup.assert_file_exists(setup.CBOW_PATH)
    setup.assert_file_exists(setup.SENTIMENTS_PATH)

    bm25_manager = BM25Manager.load(mode='meta')
    cbow = CBOW.load()

    # Lade die Queries

    query_ids, query_texts = queries.read(
        start=args.query_range[0],
        stop=args.query_range[1],
    )
    query_texts = queries.clean(query_texts)

    for i, qt in enumerate(query_texts):
        print(f"{i+1} > {qt}")
    print()

    # Bestimme die besten Argumente

    arg_ids, bm25_scores, desim_scores = index.collect_arguments(
        query_texts,
        cbow.model,
        bm25_manager.index,
        max_args=args.max_args,
    )

    # Bestimme den finalen Score
    # Dazu werden beide Scores paarweise zusammengerechnet
    # Zusätzlich kommen Sentiment Score ins Spiel, der zur Sortierung dient

    top_args = index.get_top_args(
        arg_ids,
        bm25_scores,
        desim_scores,
        alpha=args.alpha,
        top_n=100,
    )

    sentiments = index.get_sentiments(top_args)

    # Speichere die gefundenen Argumente mit scores in eine Zwischendatei

    scores.collect_scores(
        query_ids,
        query_texts,
        top_args,
        sentiments,
    )

if args.mode == 'retrieve' or args.mode == 'collect':

    # Speichere Argumente in dem passenden Output Format
    queries_args = scores.evaluate(threshold=0.5)

    with open(setup.RUN_PATH, 'w') as f_out:
        for (query_id, query_text, args) in queries_args:
            for i, arg in enumerate(args):
                f_out.write(' '.join([
                    query_id,
                    'Q0',
                    arg[0],
                    str(i + 1),
                    str(arg[1]),
                    method,
                    '\n',
                ]))
