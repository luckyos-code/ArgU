import csv
import json

import pymongo
from tqdm import tqdm

from argU import settings
from argU.preprocessing.nlp import model_nlp_pipeline, api_nlp_pipeline


class MongoDB:
    ARGS_COLL = 'args'
    DESM_IN_COLL = 'desm_in_in'
    DESM_OUT_COLL = 'desm_in_out'

    def __init__(self):
        self.client = pymongo.MongoClient(settings.MONGO_DB_URL)
        self.db = self.client[settings.MONGO_DB_NAME]

    def __str__(self):
        first_argument = self.args_coll.find().next()
        return f"""
            MongoDB Status
            --------------
            Number of Arguments stored: {self.args_coll.count()}

            First Argument (ID: {first_argument['_id']})
            ---------------------------------------------
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

    def reinitialize(self, *, max_args=-1):
        self._store_args(max_args)
        self._optimize_args()

    def model_texts_iterator(self):
        return self._TextsIterator(args_coll=self.args_coll, key='model_text')

    def api_texts_iterator(self):
        return self._TextsIterator(args_coll=self.args_coll, key='api_text')

    def add_args_embeddings(self, *, in_emb_model, out_emb_model):
        for arg in tqdm(self.args_coll.find(self._where_embeddings_not_exist), total=self.args_coll.count()):
            arg = self._add_arg_embeddings(arg, in_emb_model=in_emb_model, out_emb_model=out_emb_model)
            self._store_arg(arg, ignore_len_check=True)

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

    def _store_args(self, max_args=-1, data=None):
        if data is None:
            with open(settings.ARGS_ME_JSON_PATH, 'r', encoding='utf8') as f_in:
                data = json.load(f_in)
            data = self._get_args_list(data, max_args)

        self.args_coll.drop()
        self.args_coll.insert_many(data)

    def _get_args_list(self, data, max_args):
        if max_args != -1:
            return data['arguments'][:max_args]
        return data['arguments']

    def _optimize_args(self):
        new_args = self._get_optimized_args()
        self._store_args(data=new_args)

    def _get_optimized_args(self):
        new_args = []
        for arg in tqdm(self.args_coll.find(), total=self.args_coll.count()):
            new_arg = self._create_opt_arg(arg)
            self._append_arg(new_arg, new_args)
        return new_args

    def _append_arg(self, arg, arg_list):
        if self._has_min_length(arg, 25):
            arg_list.append(arg)

    def _create_opt_arg(self, arg):
        return {
            'id': arg['id'],
            'premises': [{
                'model_text': model_nlp_pipeline(arg['premises'][0]['text']),
                'api_text': api_nlp_pipeline(arg['premises'][0]['text']),
                'stance': arg['premises'][0]['stance']
            }],
            'conclusion': arg['conclusion']
        }

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
