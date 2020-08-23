import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

from argU import settings
from argU.preprocessing.nlp import PreprocessorPipeline, QueryPipeline, ModelPostPipeline

Query = namedtuple('Query', 'id text')


def get_queries(cbow):
    tree = ET.parse(os.path.join(settings.TOPICS_PATH))
    topics = tree.getroot()

    query_nlp_pipeline = PreprocessorPipeline(ModelPostPipeline(QueryPipeline(cbow)))

    queries = []
    for topic in topics:
        queries.append(
            Query(topic[0].text, query_nlp_pipeline(topic[1].text))
        )

    return queries


def get_mapping_to_arg_id():
    with open(settings.MAPPING_PATH, 'r') as f_in:
        data = json.load(f_in)
    return {int(k): v for (k, v) in data.items()}


def get_arg_id_to_mapping():
    mapping_to_arg_id = get_mapping_to_arg_id()
    return {v: int(k) for (k, v) in mapping_to_arg_id.items()}


def get_mapped_ids_to_sentiments():
    with open(settings.SENTIMENTS_PATH, 'r', encoding='utf8') as f_in:
        reader = csv.reader(f_in, delimiter=',')
        next(reader)

        return _sentiment_mappings(reader)


def _sentiment_mappings(csv_reader):
    sentiments = {}
    mapping = get_arg_id_to_mapping()

    for id, sent, magn in csv_reader:
        sentiments[mapping[id]] = float(sent)

    return sentiments
