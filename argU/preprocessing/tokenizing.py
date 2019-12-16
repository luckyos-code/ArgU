import nltk
import os
from nltk.corpus import wordnet
import rootpath

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')

# nltk.download()
def tokenizing_q(query: str):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    stopwords = open(STOPWORDS_PATH, "r").read().split("\n")
    q = query.split(" ")
    aaa = []
    for i in q:
        for char in punctuations:
            i = i.strip(char)
        if i.strip().lower() not in stopwords:
            # print(nltk.pos_tag(i))
            list = []
            list.append(i)
            pos_list = nltk.pos_tag(list)

            tokens = dict()
            synonyms = []
            antonyms = []
            for syn in wordnet.synsets(i):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            tokens['token'] = i
            tokens['pos_tag'] = pos_list[0][1]
            tokens['synonyms'] = synonyms
            if l.antonyms():
                tokens['anton'] = antonyms
            yield tokens
