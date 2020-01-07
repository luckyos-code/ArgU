from sentiment.analysis import get_nltk_data, run as nltk_run
from sentiment.google import run as google_run
from utils.reader import read_csv, read_csv_header

import os, rootpath, csv

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources/")
QUERIES_PATH = os.path.join(RESOURCES_PATH, "topics.csv")
QUERIES_AUTOMATIC_PATH = os.path.join(RESOURCES_PATH, "topics-automatic.csv")
ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "args-me.csv")


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
    for argument in read_csv(ARGUMENTS_PATH, 3):
        google_run(
            argument,
            "argument",
            os.path.join(ROOT_PATH, "argU/sentiment/argument_sentiments.csv"),
            os.path.join(ROOT_PATH, "argU/sentiment/sentence_sentiments.csv"),
        )


if __name__ == "__main__":
    # print(read_csv_header(QUERIES_PATH)) # [0] is query_id, [5] is query_string
    # queryEntry = next(read_csv(QUERIES_PATH, 1))
    # print(queryEntry[5])
    # print(read_csv_header(ARGUMENTS_PATH)) # [2] is source_id, [8] is id, [5] is discussion_argument
    # arguments = []
    # for argumentEntry in read_csv(ARGUMENTS_PATH, 1):
    #    print(argumentEntry[9])
    #    arguments.append(argumentEntry[5])
    # arguments = list(dict.fromkeys(arguments))
    # print(len(arguments))
    # google_queries()
    # google_argument()
    # nltk_arguments()
