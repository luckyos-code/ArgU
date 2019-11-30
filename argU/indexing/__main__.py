import os
import rootpath
from utils.reader import read_arguments
from indexing.skipgram import SkipGramModel

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

skip_gram_model = SkipGramModel()
skip_gram_model.build_dicts(CSV_PATH, 10000, max_args=10)

for argument in read_arguments(CSV_PATH, 1):
    indices = skip_gram_model.text_to_indices(argument.text)
    print(indices)
