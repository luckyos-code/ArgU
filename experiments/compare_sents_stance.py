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
coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]

values = {}
max_args = -1

counAll = countProN = countConN = countProNeg = countConPos = countConNeg = countProPos = 0

for i, ct in tqdm(enumerate(coll_train.find())):
    if i == max_args:
        break
    sent = coll_sents.find_one({'_id': ct['_id']})
    arg = coll_args.find_one({'_id': ct['_id']})
    if sent['score'] < -0.1:
        if arg['premises'][0]['stance'] == 'PRO':
            countProNeg += 1
        else:
        	countConNeg += 1
    elif sent['score'] > 0.1:
        if arg['premises'][0]['stance'] == 'CON':
            countConPos += 1
        else:
        	countProPos += 1
    else:
        if arg['premises'][0]['stance'] == 'PRO':
            countProN += 1
        else:
            countConN += 1
    counAll += 1

print(f'NEW Original stance for neutral sents: {countProN} Pro, {countConN} Con')
print(f'CORRECT Con args for negative sents: {countConNeg}')
print(f'FALSE Pro args for negative sents: {countProNeg}')
print(f'CORRECT Pro args for positive sents: {countProPos}')
print(f'FALSE Con args for positive sents: {countConPos}')
print(f'Gesamt: {counAll}')
