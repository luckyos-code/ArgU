import sys
import os
import rootpath
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import settings
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)

db = load_db()
coll_sents_train = db[settings.MONGO_DB_COL_SENTIMENTS_TRAIN]

count = 0
count_all = 0
for arg in tqdm(coll_sents_train.find()):
    if len(arg['text']) > 1000:
        count += 1
    count_all += 1
print(f'Argumente mit mehr als 1000 Zeichen: {count}')
print(f'Argumente insgesamt: {count_all}')
