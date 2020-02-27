import sys
import os
import rootpath
from tqdm import tqdm
from collections import Counter

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)

db = load_db()
coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]

values = []
for s in tqdm(coll_sents.find()):
    values.append(s['score'])

counts = list(Counter(values).items())
counts.sort(key=lambda tup: tup[0])

for c in counts:
    print(f'{c[0]},{c[1]}')
