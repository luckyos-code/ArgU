import csv
import json

import pymongo
from tqdm import tqdm

from argU import settings
from argU.preprocessing.nlp import api_nlp_pipeline, model_nlp_pipeline


class MongoDB:
    ARGS_COLL = 'args'
    DESM_IN_COLL = 'desm_in_in'
    DESM_OUT_COLL = 'desm_in_out'

    def __init__(self, *, overwrite=False, max_args=-1):
        self.client = pymongo.MongoClient(settings.MONGO_DB_URL)
        self.db = self.client[settings.MONGO_DB_NAME]
        self.overwrite = overwrite
        self.max_args = max_args
        self._init()

    def __str__(self):
        first_argument = self.args_coll.find().next()
        return f"""
            MongoDB Status
            --------------
            Number of Arguments stored: {self.args_coll.count()}

            First Argument (ID: {first_argument['_id']})
            ---------------------------------------------
            Text >> "{first_argument['premises'][0]['text'][:130]} ..."
            Model text >> "{first_argument['premises'][0]['model_text'][:130]} ..."
            API Text >> "{first_argument['premises'][0]['api_text'][:130]} ..."
            In emb >> {first_argument['premises'][0].get('in_emb', [])[:5]} ...
            Out emb >> {first_argument['premises'][0].get('out_emb', [])[:5]} ...
        """

    @property
    def args_coll(self):
        return self.db[MongoDB.ARGS_COLL]

    @property
    def desm_in_coll(self):
        return self.db[MongoDB.DESM_IN_COLL]

    @property
    def desm_out_coll(self):
        return self.db[MongoDB.DESM_OUT_COLL]

    @property
    def _where_embeddings_not_exist(self):
        return {
            "$and": [
                {"premises.0.emb_in": {"$exists": False}},
                {"premises.0.emb_out": {"$exists": False}}
            ]
        }

    @property
    def _where_new_texts_not_exist(self):
        return {
            "$and": [
                {"premises.0.model_text": {"$exists": False}},
                {"premises.0.apil_text": {"$exists": False}}
            ]
        }

    def model_texts_iterator(self):
        return self._TextsIterator(args_coll=self.args_coll, key='model_text')

    def api_texts_iterator(self):
        return self._TextsIterator(args_coll=self.args_coll, key='api_text')

    def add_args_embeddings(self, *, in_emb_model, out_emb_model):
        for arg in tqdm(self.args_coll.find(self._where_embeddings_not_exist), total=self.args_coll.count()):
            arg = self._add_arg_embeddings(arg, in_emb_model=in_emb_model, out_emb_model=out_emb_model)
            self._store_arg(arg, ignore_len_check=True)

    def add_args_sentiments(self):
        sentiments_dict = self._read_sentiments()
        for arg in tqdm(self.args_coll.find()):
            self._add_arg_sentiment(arg, sentiments_dict)

    def get_arg_by_id(self, id):
        return self.args_coll.find_one({'id': id})

    def create_desm_collection(self, *, desm, args_topn=100):
        coll = self._init_desm_collection(desm)

        json_data = []
        for i, (query_id, top_args) in enumerate(desm.query_results_iterator(args_topn=args_topn)):
            query_data = []
            for arg in top_args:
                query_data.append(arg)
            json_data.append({
                query_id: query_data
            })

        coll.insert_many(json_data)

    def _init_desm_collection(self, desm):
        coll = self.desm_in_coll if desm.emb_type == 'in_emb' else self.desm_out_coll
        coll.drop()
        return coll

    class _TextsIterator:
        def __init__(self, *, args_coll, key):
            self.args_coll = args_coll
            self.key = key

        def __iter__(self):
            for arg in self.args_coll.find():
                yield arg['premises'][0][self.key].split()

    def _add_arg_sentiment(self, arg, sentiments):
        if arg['id'] in sentiments:
            self.args_coll.update_one(
                {'_id': arg['_id']},
                {"$set": {"sentiment": sentiments[arg['id']]}},
            )

    def _read_sentiments(self):
        sentiments = {}
        with open(settings.SENTIMENTS_PATH, 'r', encoding='utf-8') as f_in:
            reader = self._get_reader(f_in)

            for row in tqdm(reader):
                id, sentiment, magnitude = row
                sentiments[id] = float(sentiment)
        return sentiments

    def _get_reader(self, f_in):
        reader = csv.reader(f_in, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        next(reader)
        return reader

    def _init(self):
        if self._must_overwrite():
            self._insert_args()
            self._add_args_texts()
            self._remove_irrelevant_args()

    def _remove_irrelevant_args(self):
        self.args_coll.remove(self._where_new_texts_not_exist)

    def _insert_args(self):
        with open(settings.ARGS_ME_JSON_PATH, 'r', encoding='utf8') as f_in:
            data = json.load(f_in)
        data = self._get_args_list(data)

        self.args_coll.drop()
        self.args_coll.insert_many(data)

    def _get_args_list(self, data):
        if self.max_args != -1:
            return data['arguments'][:self.max_args]
        return data['arguments']

    def _add_args_texts(self):
        for arg in tqdm(self.args_coll.find(), total=self.args_coll.count()):
            arg = self._add_arg_texts(arg)
            self._store_arg(arg)

    def _add_arg_texts(self, arg):
        arg['premises'][0]['model_text'] = model_nlp_pipeline(arg['premises'][0]['text'])
        arg['premises'][0]['api_text'] = api_nlp_pipeline(arg['premises'][0]['text'])
        return arg

    def _add_arg_embeddings(self, arg, *, in_emb_model, out_emb_model):
        arg['premises'][0]['in_emb'] = in_emb_model.text_to_emb(arg['premises'][0]['model_text']).tolist()
        arg['premises'][0]['out_emb'] = out_emb_model.text_to_emb(arg['premises'][0]['model_text']).tolist()
        return arg

    def _store_arg(self, arg, *, ignore_len_check=False):
        if ignore_len_check or self._has_min_length(arg, 25):
            self._update_one_arg_premise(arg)

    def _update_one_arg_premise(self, arg):
        self.args_coll.update_one(
            {'_id': arg['_id']},
            {"$set": {"premises": arg['premises']}},
        )

    def _has_min_length(self, arg, length):
        """This function uses the `api_text`, which is very close to the original `text`, but formatting errors
            were corrected. The original `text` has too much noise for this operation.
        """

        return len(arg['premises'][0]['api_text'].split()) >= length

    def _must_overwrite(self):
        return self.overwrite or self.args_coll.count() == 0
