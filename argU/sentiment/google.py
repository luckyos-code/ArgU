from google.cloud import language_v1
from google.cloud.language_v1 import enums
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging, csv, os, datetime, asyncio

# define only once
client = language_v1.LanguageServiceClient()
_executor = ThreadPoolExecutor()


async def run(content, mode, csvpath1, csvpath2):
    """
    Analyzing Sentiment
    """
    # get content
    if mode == "queries":
        text_content = content
    elif mode == "argument":
        # doc, text_content = content[9], content[0]
        doc, text_content = content[0], content[2]
    elif mode == "test":
        doc, text_content = content[0], content[2]

    # set options
    type_ = enums.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}
    encoding_type = enums.EncodingType.UTF8

    # make analysis async
    loop = asyncio.get_running_loop()
    response = False
    try:
        response = await loop.run_in_executor(
            _executor,
            partial(client.analyze_sentiment, document, encoding_type=encoding_type),
        )
    except Exception as err:
        # put into failed.csv for later
        with open(
            "../resources/sentiments/failed.csv", mode="a+", newline=""
        ) as failed:
            failed_writer = csv.writer(
                failed, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if os.stat("../resources/sentiments/failed.csv").st_size == 0:
                failed_writer.writerow(["qid", "text", "err"])
            failed_writer.writerow([doc, text_content, str(err)])
        # dummy argument to csv
        with open(csvpath1, mode="a+", newline="") as argument_sentiments_csv:
            argument_sentiment_writer = csv.writer(
                argument_sentiments_csv,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            if os.stat(csvpath1).st_size == 0:
                argument_sentiment_writer.writerow(
                    ["doc", "sentiment_score", "sentiment_magnitude"]
                )
            argument_sentiment_writer.writerow(
                [doc, "XXX", "failed",]
            )

    if mode == "queries" and response is not False:
        # add queries to csv
        with open(csvpath1, mode="w+", newline="") as sentiments_csv:
            query_sentiment_writer = csv.writer(
                sentiments_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if os.stat(csvpath1).st_size == 0:
                query_sentiment_writer.writerow(
                    ["qid", "text", "sentiment_score", "sentiment_magnitude"]
                )
            addition = ""
            number = 1
            for num, sentence in enumerate(response.sentences, start=1):
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
        with open(csvpath1, mode="a+", newline="") as argument_sentiments_csv:
            argument_sentiment_writer = csv.writer(
                argument_sentiments_csv,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            if os.stat(csvpath1).st_size == 0:
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
            with open(csvpath2, mode="a+", newline="") as sentence_sentiments_csv:
                sentence_sentiment_writer = csv.writer(
                    sentence_sentiments_csv,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                )
                if os.stat(csvpath2).st_size == 0:
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
    elif mode == "test" and response is not False:
        # print for tests
        print(u"Document ID: {}".format(doc))
        print(u"Document sentiment score: {}".format(response.document_sentiment.score))
        print(
            u"Document sentiment magnitude: {}\n".format(
                response.document_sentiment.magnitude
            )
        )

        for num, sentence in enumerate(response.sentences, start=1):
            print(u"Sentence number: {}".format(num))
            print(u"Sentence text: {}".format(sentence.text.content))
            print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
            print(
                u"Sentence sentiment magnitude: {}\n".format(
                    sentence.sentiment.magnitude
                )
            )

    """
    # log
    logging.basicConfig(
        level=logging.DEBUG, filename=logpath, filemode="a+", format="%(message)s"
    )

    logging.info(u"Date: {}\n".format(datetime.datetime.now()))

    logging.info(
        u"Document sentiment score: {}".format(response.document_sentiment.score)
    )
    logging.info(
        u"Document sentiment magnitude: {}\n".format(
            response.document_sentiment.magnitude
        )
    )

    for num, sentence in enumerate(response.sentences, start=1):
        if sentence.text.content[-1:] == ".":
            sentence.text.content = sentence.text.content[:-1]
        logging.info(u"Sentence number: {}".format(num))
        logging.info(u"Sentence text: {}".format(sentence.text.content))
        logging.info(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        logging.info(
            u"Sentence sentiment magnitude: {}\n".format(sentence.sentiment.magnitude)
        )
    """
