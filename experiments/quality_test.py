# This file's only purpuse is to check if sentiments and quality of args correlate


import csv
import json
import os
import sys
import rootpath

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.mongodb import load_db
    from argU.utils.arguments import find
    from argU.utils.arguments import Argument
except Exception as e:
    print(e)
    sys.exit(0)

q1 = ("Teachers Get Tenure", [
    "c065954f-2019-04-18T14:32:52Z-00001-000",
    "c065954f-2019-04-18T14:32:52Z-00003-000",
    "24e47090-2019-04-18T19:22:46Z-00004-000",
    "51530f3f-2019-04-18T18:15:02Z-00004-000",
    "ff0947ec-2019-04-18T12:23:12Z-00000-000",
    "1a76ed9f-2019-04-18T16:07:27Z-00005-000",
    "ff0947ec-2019-04-18T12:23:12Z-00001-000",
    "eef749e-2019-04-18T18:41:19Z-00004-000",
])

q4 = ("Corporal Punishment Used Schools", [
    "4712ec0a-2019-04-18T12:53:28Z-00007-000",
    "2fc6200f-2019-04-18T17:01:39Z-00003-000",
    "c6b2791c-2019-04-18T14:59:08Z-00002-000",
    "c6b2791c-2019-04-18T14:59:08Z-00004-000",
    "c6b278de-2019-04-18T15:01:18Z-00003-000",
    "29b5e1ff-2019-04-18T17:57:40Z-00005-000",
    "cb52628f-2019-04-18T11:53:57Z-00001-000",
    "57c3ac9d-2019-04-18T19:10:29Z-00000-000",
    "e7b98175-2019-04-18T14:36:18Z-00002-000",
    "c93845a0-2019-04-18T15:10:52Z-00001-000",
])

q5 = ("Social Security Privatized", [
    "2d6f4e75-2019-04-15T20:22:43Z-00009-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00007-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00015-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00008-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00014-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00010-000",
    "2d6f4e75-2019-04-15T20:22:43Z-00012-000",
    "41ee0719-2019-04-18T14:19:05Z-00005-000",
    "dac7811d-2019-04-18T20:00:32Z-00002-000",
    "dac7811d-2019-04-18T20:00:32Z-00004-000",
])

queries = [q1, q4, q5]

db = load_db()
coll_trans = db[setup.MONGO_DB_COL_TRANSLATION]
coll_args = db[setup.MONGO_DB_COL_ARGS]
coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]

for q in queries:
    print()
    print(f'{q[0]}\n=================================\n')
    ids = {
        ct['_id']: ct['arg_id'] for ct in coll_trans.find({'arg_id': {'$in': q[1]}})
    }

    args = find(coll_args, list(ids.keys()))
    args = {ids[a['_id']]: a for a in args}

    for i, arg_id in enumerate(q[1]):
        arg = args[arg_id]
        print(f'{i + 1} {arg_id} ({arg["_id"]})')
        # print(f'\n{Argument.get_text(arg)}\n')
        print(coll_sents.find_one({'_id': arg['_id']}))

    print()
