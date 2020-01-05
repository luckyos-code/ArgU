import os
import rootpath
import test_settings

from utils.reader import ArgumentIterator

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

debate_count = dict()
for a in ArgumentIterator(CSV_PATH, 100):
    args = debate_count.get(a.debate_id, list())
    args.append(a.text)
    debate_count[a.debate_id] = args

for (d, c) in debate_count.items():
    if len(c) <= 2:
        print(d)
        for arg in c:
            print(f"\t-> {arg}")
        print()
