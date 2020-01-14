import os
import rootpath
import test_settings
import numpy as np

from utils.reader import ArgumentIterator, FindArgumentIterator, Argument
# from utils.beautiful import print_argument_texts
from indexing.models import Argument2Vec, CBOW, DualEmbedding, BM25Manager, MixtureModel

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
CBOW_MODEL_PATH = os.path.join(RESOURCES_PATH, 'cbow.arguments.model')
A2V_MODEL_PATH = os.path.join(RESOURCES_PATH, 'a2v.model')

cbow = CBOW()
cbow.load(CBOW_MODEL_PATH)

max_args = 1000

# a2v = Argument2Vec.load(A2V_MODEL_PATH)

problematic_arguments = [
    "3d3a3182-2019-04-18T16:27:58Z-00004-000",
    "37c13c72-2019-04-18T15:11:50Z-00000-000",
    "f3f0a387-2019-04-18T12:45:29Z-00002-000",
]

for argument in FindArgumentIterator(CSV_PATH, problematic_arguments):
    vec = Argument.to_vec(argument.text_machine.split(), cbow.model, 300)
    print("-->", vec)