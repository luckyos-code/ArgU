from google.cloud import language_v1
from google.cloud.language_v1 import enums
import logging

def getQueryText(rows):
    text = ''
    for row in rows:
        query_text = row [5] + '. '
        text += query_text
    return text[:-1]

def getTitleText(rows):
    text = ''
    for row in rows:
        query_text = row [5] + '. '
        text += query_text
    return text[:-1]

def run(rows, logpath, mode):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """
    if mode == 'topics':
        text_content = getQueryText(rows)
    elif mode == 'titles':
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
    # Get overall sentiment of the input document
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
        u"Document sentiment magnitude: {}\n".format(
            response.document_sentiment.magnitude
        )
    )
    # Get sentiment for all sentences in the document
    for sentence in response.sentences:
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(u"Sentence sentiment magnitude: {}\n".format(sentence.sentiment.magnitude))
        
    logging.basicConfig(level=logging.DEBUG, filename=logpath, filemode="a+",
                        format="%(message)s")
    logging.info(u"Document sentiment score: {}".format(response.document_sentiment.score))
    logging.info(
        u"Document sentiment magnitude: {}\n".format(
            response.document_sentiment.magnitude
        )
    )
    for sentence in response.sentences:
        logging.info(u"Sentence text: {}".format(sentence.text.content))
        logging.info(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        logging.info(u"Sentence sentiment magnitude: {}\n".format(sentence.sentiment.magnitude))