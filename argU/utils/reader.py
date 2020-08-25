import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

from argU import settings
from argU.preprocessing.old_nlp import clean_to_nl, clean_to_train, clean_pos_tags

Query = namedtuple('Query', 'id text')


def get_queries(cbow):
    def old_clean(q_text):
        q_text = q_text.replace('?', '')
        q_text = clean_to_nl(q_text)
        q_text = clean_to_train(q_text)
        q_text = clean_pos_tags(q_text)

        return q_text

    tree = ET.parse(os.path.join(settings.TOPICS_PATH))
    topics = tree.getroot()

    queries = []
    for topic in topics:
        queries.append(
            Query(topic[0].text, old_clean(topic[1].text))
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
