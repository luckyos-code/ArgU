import os
import rootpath
import test_settings

from utils.reader import ArgumentIterator

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

for a in ArgumentIterator(CSV_PATH, 10):
    print(a.id)
    print(a.text_raw)
    print()
