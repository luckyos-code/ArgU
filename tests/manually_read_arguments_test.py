import os
import sys
import rootpath
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

from utils.reader import ArgumentIterator

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

for a in ArgumentIterator(CSV_PATH, 10):
    print(a.id)
    print(a.text_raw)
    print()
