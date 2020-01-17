from sentiment.nltk import get_nltk_data, run as nltk_run
from sentiment.google import run as google_run
from utils.reader import read_csv, read_csv_header
from more_itertools import unique_everseen

import os, rootpath, csv, asyncio, time

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources/")
QUERIES_PATH = os.path.join(RESOURCES_PATH, "topics.csv")
QUERIES_AUTOMATIC_PATH = os.path.join(RESOURCES_PATH, "topics-automatic.csv")
ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "args-me.csv")
CLEAN_ARGUMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiment_args.csv")
QUERY_SENTIMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiments/query_sentiments.csv")
ARGUMENT_SENTIMENTS_PATH = os.path.join(
    RESOURCES_PATH, "sentiments/argument_sentiments.csv"
)
SENTENCE_SENTIMENTS_PATH = os.path.join(
    RESOURCES_PATH, "sentiments/sentence_sentiments.csv"
)
FAILED_SENTIMENTS_PATH = os.path.join(RESOURCES_PATH, "sentiments/failed.csv")


def google_queries():
    queries = []
    for row in read_csv(QUERIES_AUTOMATIC_PATH, max_rows=-1):
        queries.append(row[1])
    for row in read_csv(QUERIES_PATH, max_rows=-1):
        queries.append(row[5] + ".")
    google_run(
        " ".join(queries),
        "queries",
        QUERY_SENTIMENTS_PATH,
        ARGUMENT_SENTIMENTS_PATH,
        SENTENCE_SENTIMENTS_PATH,
        FAILED_SENTIMENTS_PATH,
    )


def count_analyzed():
    count = 0
    for arg in read_csv(ARGUMENT_SENTIMENTS_PATH, -1):
        count += 1
    return count


def count_failed():
    count = 0
    for arg in read_csv(FAILED_SENTIMENTS_PATH, -1):
        count += 1
    return count


def get_csv_args():
    csv_args = []
    for num, arg in enumerate(read_csv(ARGUMENT_SENTIMENTS_PATH, -1), start=1):
        csv_args.append(arg[0])
    return csv_args


def check_existing(argument, csv_args):
    if argument[0] in csv_args:
        return True
    else:
        return False


def find_duplicates():
    print("\nRunning duplicate check")
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
    print("\nRunning missing check")
    tasks = []
    # get analyzed arguments
    csv_args = get_csv_args()
    # get arguments for analysis
    for argument in read_csv(CLEAN_ARGUMENTS_PATH, limit):
        # check if already in csv
        if argument[0] not in csv_args:
            # add new argument as async task
            tasks.append(argument[0])
    # give some useful info
    print(f"Missed:\t {len(tasks)}")
    print(f"In CSV:\t {len(csv_args)}")
    print(f"Should:\t {limit}")
    if len(tasks) > 0:
        print(tasks)


def google_arguments_limit(limit):
    t_start = time.time()
    loop = asyncio.get_event_loop()
    count = count_analyzed()
    startCount = count
    countFailed = count_failed()
    failCount = 0
    dummyCount = 0
    # if needed for fix
    # csv_args = get_csv_args()
    while count < limit:
        print("\nRunning analysis")
        print("")
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
                            QUERY_SENTIMENTS_PATH,
                            ARGUMENT_SENTIMENTS_PATH,
                            SENTENCE_SENTIMENTS_PATH,
                            FAILED_SENTIMENTS_PATH,
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
            time.sleep(69)
    # done
    print("Done, limit reached.")
    print(f"Added: {count - startCount}")
    print(f"Fails: {failCount}")
    print(f"Dummy: {dummyCount}")
    print(f"Time: {time.time() - t_start:.2f}")
    loop.close()


def add_quota_fails():
    failed_args = []
    for arg in read_csv(FAILED_SENTIMENTS_PATH, -1):
        if arg[2][:3] == '429':
            failed_args.append([arg[0],'',arg[1]])
    count = count_analyzed()
    tasks = len(failed_args)
    for failed in failed_args:
        google_run(
            failed,
            "argument",
            QUERY_SENTIMENTS_PATH,
            ARGUMENT_SENTIMENTS_PATH,
            SENTENCE_SENTIMENTS_PATH,
            FAILED_SENTIMENTS_PATH,
        )
    print(f"Before: {count}")
    print(f"Tasks: {tasks}")
    print(f"Added: {count_analyzed() - count}")


def remove_duplicates(csv):
    with open(csv,'r') as in_file, open('no_dup.csv','w') as out_file:
        out_file.writelines(unique_everseen(in_file))


def remove_dummies():
    return


if __name__ == "__main__":
    # maximum is 387692
    limit = 387692
    google_arguments_limit(limit)
    find_duplicates()
    check_missing(limit)
