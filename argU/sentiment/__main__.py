from sentiment.analysis import get_nltk_data, run as nltk_run
from sentiment.google import run as google_run
from utils.reader import read_csv, read_csv_header

import os, rootpath, csv, timeit

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources/")
QUERIES_PATH = os.path.join(RESOURCES_PATH, "topics.csv")
QUERIES_AUTOMATIC_PATH = os.path.join(RESOURCES_PATH, "topics-automatic.csv")
ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "args-me.csv")
CLEAN_ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiment_args.csv")


def nltk_queries():
    get_nltk_data()
    nltk_run(read_csv(QUERIES_PATH, max_rows=-1), "queries")


def nltk_arguments():
    get_nltk_data()
    nltk_run(read_csv(ARGUMENTS_PATH, 100), "arguments")


def google_queries():
    queries = []
    for row in read_csv(QUERIES_AUTOMATIC_PATH, max_rows=-1):
        queries.append(row[1])
    for row in read_csv(QUERIES_PATH, max_rows=-1):
        queries.append(row[5] + ".")
    google_run(
        " ".join(queries),
        "queries",
        os.path.join(ROOT_PATH, "argU/sentiment/query_sentiments.csv"),
        "",
    )


def google_argument():
    for num, argument in enumerate(read_csv(CLEAN_ARGUMENTS_PATH, 200), start=1):
        if num <= 200:
            continue
        google_run(
            argument,
            "argument",
            os.path.join(ROOT_PATH, "argU/sentiment/argument_sentiments.csv"),
            os.path.join(ROOT_PATH, "argU/sentiment/sentence_sentiments.csv"),
        )


def google_test_argument(argument):
    google_run(
        argument, "test", "", "",
    )


if __name__ == "__main__":
    duration = timeit.timeit(google_argument, number=1)
    print(duration)
