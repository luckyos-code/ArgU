from google.cloud import language_v1
from google.cloud.language_v1 import enums
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import csv, os, datetime, asyncio

# define only once
client = language_v1.LanguageServiceClient()
_executor = ThreadPoolExecutor()


async def run(content, mode, queryCSV, argCSV, sentenceCSV, failedCSV):
    """
    Analyzing Sentiment
    """
    # get content
    if mode == "queries":
        text_content = content
    elif mode == "argument":
        # doc, text_content = content[9], content[0]
        doc, text_content = content[0], content[2]
    # set options
    type_ = enums.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}
    encoding_type = enums.EncodingType.UTF8

    # run async analysis
    loop = asyncio.get_running_loop()
    response = False
    try:
        response = await loop.run_in_executor(
            _executor,
            partial(client.analyze_sentiment, document, encoding_type=encoding_type),
        )
    # fail
    except Exception as err:
        # put into failed.csv for later
        with open(failedCSV, mode="a+", newline="") as failed:
            failed_writer = csv.writer(
                failed, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if os.stat(failedCSV).st_size == 0:
                failed_writer.writerow(["qid", "text", "err"])
            failed_writer.writerow([doc, text_content, str(err)])
        # dummy argument to csv
        if mode == "argument":
            with open(argCSV, mode="a+", newline="") as argument_sentiments_csv:
                argument_sentiment_writer = csv.writer(
                    argument_sentiments_csv,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                )
                if os.stat(argCSV).st_size == 0:
                    argument_sentiment_writer.writerow(
                        ["doc", "sentiment_score", "sentiment_magnitude"]
                    )
                argument_sentiment_writer.writerow(
                    [doc, "XXX", "failed",]
                )
    # success
    if mode == "queries" and response is not False:
        # add queries to csv
        with open(queryCSV, mode="w+", newline="") as sentiments_csv:
            query_sentiment_writer = csv.writer(
                sentiments_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if os.stat(queryCSV).st_size == 0:
                query_sentiment_writer.writerow(
                    ["qid", "text", "sentiment_score", "sentiment_magnitude"]
                )
            addition = ""
            number = 1
            for num, sentence in enumerate(response.sentences, start=1):
                # handle weird stuff in data
                if num == 51:
                    number = 1
                    addition = "000"
                if addition == "000" and number == 10:
                    addition = "00"
                if sentence.text.content[-1:] == ".":
                    sentence.text.content = sentence.text.content[:-1]
                query_sentiment_writer.writerow(
                    [
                        addition + str(number),
                        sentence.text.content,
                        "{0:.4f}".format(sentence.sentiment.score),
                        "{0:.4f}".format(sentence.sentiment.magnitude),
                    ]
                )
                number += 1
    elif mode == "argument" and response is not False:
        # add argument to csv
        with open(argCSV, mode="a+", newline="") as argument_sentiments_csv:
            argument_sentiment_writer = csv.writer(
                argument_sentiments_csv,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            if os.stat(argCSV).st_size == 0:
                argument_sentiment_writer.writerow(
                    ["doc", "sentiment_score", "sentiment_magnitude"]
                )
            argument_sentiment_writer.writerow(
                [
                    doc,
                    "{0:.4f}".format(response.document_sentiment.score),
                    "{0:.4f}".format(response.document_sentiment.magnitude),
                ]
            )
            # add sentences to csv
            with open(sentenceCSV, mode="a+", newline="") as sentence_sentiments_csv:
                sentence_sentiment_writer = csv.writer(
                    sentence_sentiments_csv,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                )
                if os.stat(sentenceCSV).st_size == 0:
                    sentence_sentiment_writer.writerow(
                        [
                            "doc",
                            "num",
                            "snippet",
                            "sentiment_score",
                            "sentiment_magnitude",
                        ]
                    )
                for num, sentence in enumerate(response.sentences, start=1):
                    sentence_sentiment_writer.writerow(
                        [
                            doc,
                            num,
                            sentence.text.content.split(" ", 1)[0],
                            "{0:.4f}".format(sentence.sentiment.score),
                            "{0:.4f}".format(sentence.sentiment.magnitude),
                        ]
                    )
