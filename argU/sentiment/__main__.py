from sentiment.nltk import get_nltk_data, run as nltk_run
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
    RESOURCES_PATH, "sentiments/argument_sentiments.csv"
)
SENTENCE_SENTIMENTS_PATH = os.path.join(
    RESOURCES_PATH, "sentiments/sentence_sentiments.csv"
)
QUERY_SENTIMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiments/query_sentiments.csv")


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
        " ".join(queries), "queries", QUERY_SENTIMENTS_PATH, "",
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


def count_failed():
    count = 0
    for arg in read_csv("../resources/sentiments/failed.csv", -1):
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
    print(f"Found:\t {len(duplicates)}")
    if len(duplicates) > 0:
        print(duplicates)


def check_missing(limit):
    print("\nrunning missing check")
    tasks = []
    # get analyzed arguments
    csv_args = get_csv_args()
    # get arguments for analysis
    for argument in read_csv(CLEAN_ARGUMENTS_PATH, limit):
        # check if already in csv
        if argument[0] not in csv_args:
            # add new argument as async task
            tasks.append(num)
    # give some useful info
    count = count_analyzed()
    print(f"Missed:\t {len(tasks)}")
    print(f"In CSV:\t {count}")
    print(f"Should:\t {limit}")
    if len(tasks) > 0:
        print(tasks)


def get_csv_args():
    csv_args = []
    for num, arg in enumerate(read_csv(ARGUMENT_SENTIMENTS_PATH, -1), start=1):
        csv_args.append(arg[0])
    return csv_args


# def compare_to_csv():
# test for all arguments analyzed


def run_checks(limit):
    find_duplicates()
    check_missing(limit)


def check_existing(argument, csv_args):
    if argument[0] in csv_args:
        return True
    else:
        return False


def async_google_argument(limit):
    print("\nrunning analysis")
    loop = asyncio.get_event_loop()
    count = count_analyzed()
    countFailed = count_failed()
    failCount = 0
    dummyCount = 0
    # if needed for fix
    # csv_args = get_csv_args()
    while count < limit:
        t0 = time.time()
        tasks = []
        olddummyCount = dummyCount
        # get arguments for analysis
        for num, argument in enumerate(read_csv(CLEAN_ARGUMENTS_PATH, limit), start=1):
            # ignore already analyzed
            if num > count:
                # if needed for fix
                # if check_existing(argument, csv_args) is True:
                #    continue
                # at least 24 words in argument
                if len(argument[2].split()) > 24:
                    # only analyze first 1000 characters
                    argument[2] = argument[2][:1000]
                    # add new argument as async task
                    tasks.append(
                        google_run(
                            argument,
                            "argument",
                            ARGUMENT_SENTIMENTS_PATH,
                            SENTENCE_SENTIMENTS_PATH,
                        )
                    )
                # dummy in csv
                else:
                    with open(
                        ARGUMENT_SENTIMENTS_PATH, mode="a+", newline=""
                    ) as argument_sentiments_csv:
                        argument_sentiment_writer = csv.writer(
                            argument_sentiments_csv,
                            delimiter=",",
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL,
                        )
                        if os.stat(ARGUMENT_SENTIMENTS_PATH).st_size == 0:
                            argument_sentiment_writer.writerow(
                                ["doc", "sentiment_score", "sentiment_magnitude"]
                            )
                        argument_sentiment_writer.writerow(
                            [argument[0], "YYY", "too long",]
                        )
                    dummyCount += 1
            if len(tasks) == 600:
                break
        # run async tasks
        loop.run_until_complete(asyncio.gather(*tasks))
        # give some useful info
        oldCount = count
        count = count_analyzed()
        oldcountFailed = countFailed
        countFailed = count_failed()
        failed = countFailed - oldcountFailed
        print(f"\nTasks:\t {len(tasks)}")
        print(f"Failed:\t {failed}")
        print(f"Dummy:\t {dummyCount - olddummyCount}")
        print(f"In CSV:\t {count}")
        print(f"Time:\t {time.time() - t0:.2f}\n")
        if failed > 0:
            failCount += failed
        if count < limit:
            # wait a minute for 600 quota/min limit
            print("Waiting before new request...")
            time.sleep(62)
    print("Done, limit reached.")
    print(f"Fails: {failCount}")
    print(f"Dummy: {dummyCount}")
    loop.close()


if __name__ == "__main__":
    limit = 102000
    async_google_argument(limit)
    run_checks(limit)
