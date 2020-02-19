import sys
import os
import rootpath
import re
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.mongodb import load_db
    from argU.utils import queries as Q
except Exception as e:
    raise e

db = load_db()
coll = db[setup.MONGO_DB_COL_TRAIN]

print('Train to .trec')
with open(setup.TREC_PATH, 'w', encoding='utf-8') as f_out:
    for arg in tqdm(coll.find()):
        out = (
            f'<DOC>\n'
            f'<DOCNO>{arg["_id"]}</DOCNO>\n'
            f'{arg["text"]}\n'
            f'</DOC>\n'

        )
        f_out.write(out)

print('Topics to .trec')
queries = Q.read()

with open(setup.TREC_PATH_TOPICS, 'w', encoding='utf-8') as f_out:
    for query in queries:
        f_out.write((
            f'<top>\n'
            f'<num>{query.id}</num><title>\n'
            f'{query.text}\n'
            f'</title>\n'
            f'</top>\n'
        ))
