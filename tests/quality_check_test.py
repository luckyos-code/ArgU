import os
import rootpath
import test_settings
import re
import sys
from tqdm import tqdm

from utils.beautiful import print_argument_texts
from utils.reader import ArgumentIterator
from preprocessing.tools import clean_text, model_text

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
STOPWORDS_PATH = os.path.join(RESOURCES_PATH, 'stopwords_eng.txt')


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

# short_args = 0
# arg_count = 0
for arg in ArgumentIterator(CSV_PATH, max_args=100):
    # for term in arg.text.split():
    print("-> ", model_text(arg.text), '\n')
    # arg_count += 1
    # if len(arg.text.split()) <= 25:
    # short_args += 1

    # print(f"Argumente Insgesamt: {arg_count}")
    # print(f"Zu kurze Argumente: {short_args}")
    # print(
    # f"Prozentualer Anteil zu kurzer Argumente: {(short_args*100/arg_count):.2f}%")
