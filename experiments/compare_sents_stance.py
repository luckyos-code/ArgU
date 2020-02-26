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
max_args = 1

for i, ct in tqdm(enumerate(coll_train.find())):
    if i == max_args:
        break
    sent = coll_sents.find_one({'_id': ct['_id']})
    arg = coll_args.find_one({'_id': ct['_id']})
    print(arg)
    values[ct['_id']] = (arg['premises'][0]['stance'], sent['score'])
print(values)

# countProN = countConN = countProNeg = countConPos = 0
# for arg in tqdm(coll_sents.find()):
#     if arg['score'] < -0.1:
#         if values[arg['_id']]['stance'] == 'PRO':
#             countProNeg += 1
#     elif arg['score'] > 0.1:
#         if values[arg['_id']]['stance'] == 'CON':
#             countConPos += 1
#     else:
#         if values[arg['_id']]['stance'] == 'PRO':
#             countProN += 1
#         else:
#             countConN += 1

# print(f'Original stance for neutral sents: {countProN} Pro, {countConN} Con')
# print(f'Pro args for negative sents: {countProNeg}')
# print(f'Con args for positive sents: {countConPos}')
