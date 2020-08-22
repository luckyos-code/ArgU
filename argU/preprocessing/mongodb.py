import csv

from tqdm import tqdm


def collection_exists(coll):
    return coll.count_documents({}) > 0


# def init_db(coll_args, coll_translation, directory):
#     """Initialisiere die Argumente aus der Ursprungsdatei"""
#
#     trans = []
#     with open(os.path.join(directory, 'args-me.json'), 'r') as f_in:
#         data = json.load(f_in)
#         arguments = data['arguments']
#
#     for i, arg in tqdm(enumerate(arguments)):
#         trans.append({
#             '_id': i,
#             'arg_id': arguments[i].pop('id')
#         })
#         arguments[i]['_id'] = i
#
#     print(f'Insert {len(arguments)} arguments into mongo db...')
#     coll_args.insert_many(arguments)
#     coll_translation.insert_many(trans)


# def new_id_to_args_id_dict(coll_trans):
#     trans_dict = dict()
#     for arg in tqdm(coll_trans.find()):
#         trans_dict[arg['arg_id']] = arg['_id']
#     return trans_dict


def init_sents(coll_trans, coll_sents):
    trans_dict = new_id_to_args_id_dict(coll_trans)

    with open(
            settings.SENTIMENTS_PATH, 'r', encoding='utf-8', newline=''
    ) as f_in:
        reader = csv.reader(f_in, **settings.SENTIMENTS_CONFIG)
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


def init_sents_train(source_coll, destiny_coll, min_arg_length=25, max_args=-1):
    data = []
    for i, arg in tqdm(enumerate(source_coll.find())):
        if i == max_args:
            break

        text = arg['premises'][0]['text']
        nl_text = clean_to_nl(text)
        if len(nl_text.split()) >= min_arg_length:
            data.append({
                '_id': arg['_id'],
                'text': clean_to_sentiment(text),
            })

    destiny_coll.insert_many(data)


def init_emb_backup(db):
    coll_emb = db[settings.MONGO_DB_COL_EMBEDDINGS]
    coll_emb_back = db[settings.MONGO_DB_COL_EMBEDDINGS_BACKUP]

    coll_emb_back.drop()

    print('Create embedding backup...')
    pipeline = [{"$match": {}}, {"$out": settings.MONGO_DB_COL_EMBEDDINGS_BACKUP}]
    coll_emb.aggregate(pipeline)

    print(coll_emb.find_one({}))
    print(coll_emb_back.find_one({}))


def init_res_backup(db):
    coll = db[settings.MONGO_DB_COL_RESULTS]
    coll_back = db[settings.MONGO_DB_COL_RESULTS_BACKUP]

    coll_back.drop()

    print('Create results backup...')
    pipeline = [{"$match": {}}, {"$out": settings.MONGO_DB_COL_RESULTS_BACKUP}]
    coll.aggregate(pipeline)

    print(coll.find_one({}))
    print(coll_back.find_one({}))


def load_emb_backup(db):
    coll_emb = db[settings.MONGO_DB_COL_EMBEDDINGS]
    coll_emb_back = db[settings.MONGO_DB_COL_EMBEDDINGS_BACKUP]

    coll_emb.drop()

    print('Load embedding backup...')
    pipeline = [{"$match": {}}, {"$out": settings.MONGO_DB_COL_EMBEDDINGS}]
    coll_emb_back.aggregate(pipeline)

    print(coll_emb.find_one({}))
    print(coll_emb_back.find_one({}))


def load_res_backup(db):
    coll = db[settings.MONGO_DB_COL_RESULTS]
    coll_back = db[settings.MONGO_DB_COL_RESULTS_BACKUP]

    coll.drop()

    print('Load results backup...')
    pipeline = [{"$match": {}}, {"$out": settings.MONGO_DB_COL_RESULTS}]
    coll_back.aggregate(pipeline)

    print(coll.find_one({}))
    print(coll_back.find_one({}))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='Input Args path',
        default=settings.RESOURCES_PATH,
    )
    args = parser.parse_args()

    db = load_db()

    coll_args = db[settings.MONGO_DB_COL_ARGS]
    coll_train = db[settings.MONGO_DB_COL_TRAIN]
    coll_trans = db[settings.MONGO_DB_COL_TRANSLATION]
    coll_sents = db[settings.MONGO_DB_COL_SENTIMENTS]
    coll_sents_train = db[settings.MONGO_DB_COL_SENTIMENTS_TRAIN]

    # print('Delete Collections...')
    # coll_args.drop()
    # coll_train.drop()
    # coll_trans.drop()
    coll_sents.drop()

    if (not collection_exists(coll_args) or not collection_exists(coll_trans)):
        print('Init args collection...')
        init_db(coll_args, coll_trans, args.input)

    if (not collection_exists(coll_sents)):
        print('Init sentiment collection...')
        init_sents(coll_trans, coll_sents)

    if (not collection_exists(coll_train)):
        print('Init train collection...')
        init_train(coll_args, coll_train, max_args=-1)

    if (not collection_exists(coll_sents_train)):
        print('Init sentiments train collection...')
        init_sents_train(coll_args, coll_sents_train, max_args=-1)

    print('Argument Collection:', coll_args.count_documents({}))
    print('Training Collection:', coll_train.count_documents({}))
    print('Translation Collection:', coll_trans.count_documents({}))
    print('Sentiments Collection:', coll_sents.count_documents({}))
    print('Sentiments Train Collection:', coll_sents_train.count_documents({}))
    print()

    print(coll_args.find_one({}))
    print(coll_train.find_one({}))
    print(coll_trans.find_one({}))
    print(coll_sents.find_one({}))
    print(coll_sents_train.find_one({}))
    print()

    coll_emb = db[settings.MONGO_DB_COL_EMBEDDINGS]
    coll_emb_back = db[settings.MONGO_DB_COL_EMBEDDINGS_BACKUP]

    # print(coll_emb.find_one({}))
    # print(coll_emb_back.find_one({}))
