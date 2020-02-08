import os
import sys
import rootpath
import xml.etree.ElementTree as ET

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(ROOT_PATH, 'argU'))

from preprocessing.tools import machine_model_clean, sentiment_clean


def read(file_path, start=-1, stop=-1):
    """Erstelle 2 Listen mit IDs und topics der jeweiligen queries"""

    query_ids = []
    query_texts = []

    tree = ET.parse(file_path)
    topics = tree.getroot()

    for topic in topics:
        query_ids.append(topic[0].text)
        query_texts.append(topic[1].text)

    if stop != -1:
        query_ids = query_ids[:stop]
        query_texts = query_texts[:stop]

    if start != -1:
        query_ids = query_ids[start:]
        query_texts = query_texts[start:]

    return query_ids, query_texts


def clean(query_texts):
    for idx, text in enumerate(query_texts):
        query_texts[idx] = machine_model_clean(sentiment_clean(text))
        query_texts[idx] = query_texts[idx].replace('?', '')
    return query_texts


if __name__ == '__main__':
    import os
    import rootpath
    import sys

    RESOURCES_PATH = os.path.join(rootpath.detect(), 'resources/')
    QUERIES_PATH = os.path.join(
        RESOURCES_PATH, 'topics-automatic-runs-task-1.xml')

    queries = read(QUERIES_PATH, start=5, stop=10)
    cleaned_texts = clean(queries[1])
    print(cleaned_texts)
