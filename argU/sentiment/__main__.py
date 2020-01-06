from sentiment.analysis import get_nltk_data, run as nltk_run
from sentiment.google import run as google_run
from utils.reader import read_csv, read_csv_header

import os, rootpath, csv

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
TOPICS_PATH = os.path.join(RESOURCES_PATH, 'topics.csv')
TOPICS_AUTOMATIC_PATH = os.path.join(RESOURCES_PATH, 'topics-automatic.csv')
TITLES_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

def nltk_topics():
    get_nltk_data()
    nltk_run(read_csv(TOPICS_PATH, max_rows=-1), 'topics')
    
def nltk_titles():
    get_nltk_data()
    nltk_run(read_csv(TITLES_PATH, 100), 'titles')

def google_topics():
    topics = []
    for row in read_csv(TOPICS_AUTOMATIC_PATH, max_rows=-1):
        topics.append(row[1])
    for row in read_csv(TOPICS_PATH, max_rows=-1):
        topics.append(row[5] + '.')
    google_run(topics, os.path.join(ROOT_PATH, 'argU/sentiment/topic_sentiments.csv'), os.path.join(ROOT_PATH, 'argU/sentiment/topic_log'), 'topics')
    
def google_titles():
    google_run(read_csv(TITLES_PATH, 100), os.path.join(ROOT_PATH, 'argU/sentiment/title_sentiments.csv'), os.path.join(ROOT_PATH, 'argU/sentiment/title_log'), 'titles')

if __name__ == "__main__":
    #print(read_csv_header(TOPICS_PATH)) # [0] is topic_id, [5] is query_string
    #topicEntry = next(read_csv(TOPICS_PATH, 1))
    #print(topicEntry[5])
    #print(read_csv_header(TITLES_PATH)) # [2] is source_id, [8] is id, [5] is discussion_title
    #titles = []
    #for titleEntry in read_csv(TITLES_PATH, -1):
    #    titles.append(titleEntry[5])
    #titles = list(dict.fromkeys(titles))
    #print(len(titles))
    #google_topics()
    google_titles()
    #nltk_titles()