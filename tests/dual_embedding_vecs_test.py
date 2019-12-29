import os
import rootpath
import test_settings
from gensim.models import KeyedVectors
from utils.reader import ArgumentIterator
from indexing.models import CBOW, Text2Vec, BM25, Argument2Vec

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.arguments.model')

cbow = CBOW()
cbow.load(MODEL_PATH)
model = cbow.model

print(f"Größe des Vokablars: {len(model.wv.vocab)}\n")

word = 'abortion'

w1_sim = model.wv.most_similar(word)
print(model.wv.vocab[word].index)

outv = KeyedVectors(300)
