from sentiment.analysis import get_nltk_data, run as nltk_run
from sentiment.google import run as google_run
from utils.reader import read_csv, read_csv_header

import os, rootpath, csv, asyncio, time

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources/")
QUERIES_PATH = os.path.join(RESOURCES_PATH, "topics.csv")
QUERIES_AUTOMATIC_PATH = os.path.join(RESOURCES_PATH, "topics-automatic.csv")
ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "args-me.csv")
CLEAN_ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiment_args.csv")
ARGUMENT_SENTIMENTS_PATH = os.path.join(ROOT_PATH, "argU/sentiment/argument_sentiments.csv")
SENTENCE_SENTIMENTS_PATH = os.path.join(ROOT_PATH, "argU/sentiment/sentence_sentiments.csv")


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


def google_test_argument(argument):
    # argument = [doc,stance,text]
    google_run(
        argument, "test", "", "",
    )


def async_google_argument():
    limit = 10000
    count = 0
    for line in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
        count += 1
    while True:
        t0 = time.time()
        tasks = []
        # get analyzed arguments
        csv_args = []
        for arg in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
            csv_args.append(arg[0])
        # get arguments for analysis
        for argument in read_csv(CLEAN_ARGUMENTS_PATH, limit):
            # check if already in csv
            if argument[0] not in csv_args:
                # add new argument as async task
                tasks.append(
                    google_run(
                        argument,
                        "argument",
                        ARGUMENT_SENTIMENTS_PATH,
                        SENTENCE_SENTIMENTS_PATH,
                    )
                )
            if len(tasks) == 600:
                break
        # run async tasks
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(*tasks))
        # give some useful info
        count = 0
        for line in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
            count += 1
        print(f'\nTasks:\t {len(tasks)}')
        print(f'Failed:\t {(len(tasks) + len(csv_args)) - count}')
        print(f'In CSV:\t {count}')
        print(f'Time:\t {time.time() - t0}\n')
        # wait for 600 quota/min limit
        if count < limit:
            print('Waiting before new request...')
            time.sleep(60)
        else:
            break
    loop.close()


if __name__ == "__main__":
    async_google_argument()
