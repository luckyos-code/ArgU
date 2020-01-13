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
ARGUMENT_SENTIMENTS_PATH = os.path.join(
    ROOT_PATH, "argU/sentiment/argument_sentiments.csv"
)
SENTENCE_SENTIMENTS_PATH = os.path.join(
    ROOT_PATH, "argU/sentiment/sentence_sentiments.csv"
)


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


def count_analyzed():
    count = 0
    for arg in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
        count += 1
    return count


def find_duplicates():
    print("\nrunning duplicate check")
    csv_args = []
    duplicates = []
    for arg in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
        if arg[0] in csv_args:
            duplicates.append(arg[0])
        else:
            csv_args.append(arg[0])
    print(f"\nFound:\t {len(duplicates)}")
    print(duplicates)


# def compare_to_csv():
# test for all arguments analyzed


def check_missing():
    print("\nrunning missing check")
    limit = 50000
    tasks = []
    # get analyzed arguments
    csv_args = []
    for num, arg in enumerate(read_csv(ARGUMENT_SENTIMENTS_PATH, -1), start=1):
        csv_args.append(arg[0])
    # get arguments for analysis
    for argument in read_csv(CLEAN_ARGUMENTS_PATH, limit):
        # check if already in csv
        if argument[0] not in csv_args:
            # add new argument as async task
            tasks.append(num)
    # give some useful info
    count = count_analyzed()
    print(f"\nMissed:\t {len(tasks)}")
    print(f"In CSV:\t {count}")
    print(f"Should:\t {limit}")
    print(tasks)


def async_google_argument():
    print("\nrunning analysis")
    limit = 100000
    count = count_analyzed()
    while True:
        t0 = time.time()
        tasks = []
        # get arguments for analysis
        for num, argument in enumerate(read_csv(CLEAN_ARGUMENTS_PATH, limit), start=1):
            # ignore already analyzed
            if num > count:
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
        oldCount = count
        count = count_analyzed()
        failed = (len(tasks) + oldCount) - count
        print(f"\nTasks:\t {len(tasks)}")
        print(f"Failed:\t {failed}")
        print(f"In CSV:\t {count}")
        print(f"Time:\t {time.time() - t0:.2f}\n")
        if failed > 0:
            print("I failed you.. :(")
            break
        if count < limit:
            # wait a minute for 600 quota/min limit
            print("Waiting before new request...")
            time.sleep(61)
        else:
            break
    loop.close()


if __name__ == "__main__":
    async_google_argument()
