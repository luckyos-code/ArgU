import os
import rootpath
import test_settings
import re
import sys
from tqdm import tqdm

from utils.beautiful import print_argument_texts
from utils.reader import ArgumentIterator, ArgumentCbowIterator
from preprocessing.tools import clean_text, model_text

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
STOPWORDS_PATH = os.path.join(ROOT_PATH, 'stopwords_eng.txt')


# def split_concatenated_words(text):
#     new_splits = []
#     for token in text.split():
#         if not token.isupper() and not token.islower():
#             new_splits.extend(re.findall('[a-zA-Z][^A-Z]*', token))
#         else:
#             new_splits.append(token)

#     return ' '.join(new_splits)


# def get_sub_args(text):
#     p = re.compile("[1-9][\).-]")
#     sub_arg_pos = []
#     sub_args = []

#     for m in p.finditer(text):
#         sub_arg_pos.append(m.start())

#     for i in range(len(sub_arg_pos)):
#         if i + 1 < len(sub_arg_pos):
#             sub_args.append(text[sub_arg_pos[i]:sub_arg_pos[i + 1]])
#         else:
#             sub_args.append(text[sub_arg_pos[i]:])

#     return sub_args


# for arg in ArgumentCbowIterator(CSV_PATH, max_args=100):
# print(' '.join(arg))

for i, arg in enumerate(ArgumentIterator(CSV_PATH, max_args=10)):
    print(f'{i}.1 -> ', arg.text_raw)
    print()
    print(f'{i}.2 -> ', arg.text)
    print()
    print(f'{i}.3 -> ', model_text(arg.text))
    print('\n', '=' * 20, '\n')
