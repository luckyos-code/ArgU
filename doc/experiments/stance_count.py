import sys
import os
import rootpath
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)

db = load_db()
coll_args = db[setup.MONGO_DB_COL_ARGS]
coll_train = db[setup.MONGO_DB_COL_TRAIN]

max_args = -1

counAll = countPro = countCon = 0

for i, ct in tqdm(enumerate(coll_train.find())):
    if i == max_args:
        break
    arg = coll_args.find_one({'_id': ct['_id']})

    if arg['premises'][0]['stance'] == 'PRO':
        countPro += 1
    else:
        countCon += 1
    counAll += 1

print(f'PRO: {countPro}')
print(f'CON: {countCon}')
print(f'Gesamt: {counAll}')
