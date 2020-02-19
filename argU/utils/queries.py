import os
import sys
import numpy as np
import rootpath
import xml.etree.ElementTree as ET
from collections import namedtuple

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.texts import clean_to_nl
    from argU.preprocessing.texts import clean_to_train
    from argU.preprocessing.texts import clean_pos_tags
except Exception as e:
    print(e)
    sys.exit(0)

Query = namedtuple('Query', 'id text')


def read():
    """Read queries and clean them"""

    queries = []
    tree = ET.parse(setup.TOPICS_PATH)
    topics = tree.getroot()

    for topic in topics:
        queries.append(
            Query(topic[0].text, __clean(topic[1].text))
        )

    return queries


def __clean(q_text):
    q_text = q_text.replace('?', '')
    q_text = clean_to_nl(q_text)
    q_text = clean_to_train(q_text)
    q_text = clean_pos_tags(q_text)

    return q_text


if __name__ == '__main__':
    from argU.indexing.models import CBOW
    from argU.indexing.models import DESM

    cbow = CBOW.load()
    desm = DESM(cbow)
    queries = read()

    print(desm.queries_to_emb(queries))
