import os
import rootpath
import argparse

from utils.reader import read_arguments, ArgumentIterator
from indexing.models import CBOW, Argument2Vec

parser = argparse.ArgumentParser()
parser.add_argument('status', default='train', choices=['train', 'load'])
args = parser.parse_args()

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.model')

cbow = CBOW(min_count=3)
if args.status == 'train':
    cbow.build(CSV_PATH, max_args=10000, store_path=MODEL_PATH)
elif args.status == 'load':
    cbow.load(MODEL_PATH)
model = cbow.model

max_args = 1000
query = 'christianity is bad'

a2v = Argument2Vec(model, CSV_PATH, max_args=max_args)
most_simiar_arguments = a2v.most_similar(query)


print()
print(f"Query: {query}")
print('=' * 20)
argument_iter = ArgumentIterator(CSV_PATH, max_args)
for i, argument in enumerate(argument_iter):
    if argument.id in most_simiar_arguments:
        print(f"{i}: Len = {len(argument.text)}, Arg = {argument.text[:15]}")
