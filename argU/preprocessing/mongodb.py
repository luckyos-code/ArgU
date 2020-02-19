from pymongo import MongoClient
import csv
import json
import os
import rootpath
import sys
from tqdm import tqdm

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.texts import clean_to_train
    from argU.preprocessing.texts import clean_to_nl
except Exception as e:
    print(e)
    sys.exit(0)


def load_db():
    """Erhalte ein DB Objekt"""

    print('Connect to MongoDB...')
    client = MongoClient(setup.MONGO_DB_URL)
    return client[setup.MONGO_DB_NAME]


def collection_exists(coll):
    return coll.count_documents({}) > 0


def init_db(coll_args, coll_translation, directory):
    """Initialisiere die Argumente aus der Ursprungsdatei"""

    trans = []
    with open(os.path.join(directory, 'args-me.json'), 'r') as f_in:
        data = json.load(f_in)
        arguments = data['arguments']

    for i, arg in tqdm(enumerate(arguments)):
        trans.append({
            '_id': i,
            'arg_id': arguments[i].pop('id')
        })
        arguments[i]['_id'] = i

    print(f'Insert {len(arguments)} arguments into mongo db...')
    coll_args.insert_many(arguments)
    coll_translation.insert_many(trans)


def new_id_to_args_id_dict(coll_trans):
    trans_dict = dict()
    for arg in tqdm(coll_trans.find()):
        trans_dict[arg['_id']] = arg['arg_id']
    return trans_dict


def init_sents(coll_trans, coll_sents):
    trans_dict = new_id_to_args_id_dict(coll_trans)

    with open(
        setup.SENTIMENTS_PATH, 'r', encoding='utf-8', newline=''
    ) as f_in:
        reader = csv.reader(f_in, **setup.SENTIMENTS_CONFIG)
        header = next(reader)

        sents_data = []
        for line in tqdm(reader):
            sents_data.append({
                '_id': trans_dict[line[0]],
                'score': float(line[1]),
                'magnitude': float(line[2])
            })
        del trans_dict
        coll_sents.insert_many(sents_data)


def init_train(source_coll, destiny_coll, min_arg_length=25, max_args=-1):
    """Erstelle die Trainingsdaten-Komponente für CBOW und BM25"""

    data = []
    for i, arg in tqdm(enumerate(source_coll.find())):
        if i == max_args:
            break

        nl_text = clean_to_nl(arg['premises'][0]['text'])
        if len(nl_text.split()) >= min_arg_length:
            data.append({
                '_id': arg['_id'],
                'text': clean_to_train(nl_text),
            })

    destiny_coll.insert_many(data)


def store(data, coll_name, override=False):
    db = load_db()
    coll = db[coll_name]

    if override:
        coll.delete_many()

    if type(data) is list:
        coll.insert_many(data)
    else:
        coll.insert_one(data)


def init_emb_backup(db):
    coll_emb = db[setup.MONGO_DB_COL_EMBEDDINGS]
    coll_emb_back = db[setup.MONGO_DB_COL_EMBEDDINGS_BACKUP]

    coll_emb_back.drop()

    print('Create embedding backup...')
    pipeline = [{"$match": {}}, {"$out": setup.MONGO_DB_COL_EMBEDDINGS_BACKUP}]
    coll_emb.aggregate(pipeline)

    print(coll_emb.find_one({}))
    print(coll_emb_back.find_one({}))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='Input Args path',
        default=setup.RESOURCES_PATH,
    )
    args = parser.parse_args()

    db = load_db()

    coll_args = db[setup.MONGO_DB_COL_ARGS]
    coll_train = db[setup.MONGO_DB_COL_TRAIN]
    coll_trans = db[setup.MONGO_DB_COL_TRANSLATION]
    coll_sents = db[setup.MONGO_DB_COL_SENTIMENTS]

    # print('Delete Collections...')
    # coll_args.drop()
    # coll_train.drop()
    # coll_trans.drop()
    # coll_sents.drop()

    if(not collection_exists(coll_args) or not collection_exists(coll_trans)):
        print('Init args collection...')
        init_db(coll_args, coll_trans, args.input)

    if(not collection_exists(coll_sents)):
        print('Init sentiment collection...')
        init_sents(coll_trans, coll_sents)

    if(not collection_exists(coll_train)):
        print('Init train collection...')
        init_train(coll_args, coll_train, max_args=-1)

    print('Argument Collection:', coll_args.count_documents({}))
    print('Training Collection:', coll_train.count_documents({}))
    print('Translation Collection:', coll_trans.count_documents({}))
    print('Sentiments Collection:', coll_sents.count_documents({}))
    print()

    print(coll_args.find_one({}))
    print(coll_train.find_one({}))
    print(coll_trans.find_one({}))
    print(coll_sents.find_one({}))
