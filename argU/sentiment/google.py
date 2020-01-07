from google.cloud import language_v1
from google.cloud.language_v1 import enums
import logging, csv, os, datetime


def getQueryText(rows):
    return " ".join(rows)


def getTitleText(rows):
    return 'There are some facts to what you said but most is speculation. However, you failed to address my other two points, and so the debate is won by default. 2. Religious minorities have flourished under Islam. Muslims are commanded to protect Jews and Christians (the People of the Book) and do them no harm. The Quran says in Sura 109, "To you, your religion. To me, mine." 3. Islam is intolerant of enslaving human beings. The religion eradicated the institution of slavery thanks to the principles set in motion by Muhammad, who was an abolitionist.'
    """
    text = ''
    for row in rows:
        query_text = row [5] + '. '
        text += query_text
    return text[:-1] 
    """


def run(rows, csvpath, logpath, mode):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """
    if mode == "topics":
        text_content = getQueryText(rows)
    elif mode == "titles":
        text_content = getTitleText(rows)

    client = language_v1.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_sentiment(document, encoding_type=encoding_type)

    # print
    # logging.info(u"Document number: {}".format(num))
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
        u"Document sentiment magnitude: {}\n".format(
            response.document_sentiment.magnitude
        )
    )

    for num, sentence in enumerate(response.sentences, start=1):
        if sentence.text.content[-1:] == ".":
            sentence.text.content = sentence.text.content[:-1]
        print(u"Sentence number: {}".format(num))
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(
            u"Sentence sentiment magnitude: {}\n".format(sentence.sentiment.magnitude)
        )
    # csv
    if mode == "topics":
        text_content = getQueryText(rows)
        with open(csvpath, mode="a+", newline="") as sentiments_csv:
            sentiment_writer = csv.writer(
                sentiments_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if os.stat(csvpath).st_size == 0:
                sentiment_writer.writerow(
                    ["num", "text", "sentiment_score", "sentiment_magnitude"]
                )
            for num, sentence in enumerate(response.sentences, start=1):
                if sentence.text.content[-1:] == ".":
                    sentence.text.content = sentence.text.content[:-1]
                sentiment_writer.writerow(
                    [
                        num,
                        sentence.text.content,
                        sentence.sentiment.score,
                        sentence.sentiment.magnitude,
                    ]
                )
    # log
    logging.basicConfig(
        level=logging.DEBUG, filename=logpath, filemode="a+", format="%(message)s"
    )

    logging.info(u"Date: {}\n".format(datetime.datetime.now()))

    # logging.info(u"Document number: {}".format(num))
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