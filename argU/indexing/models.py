import time

import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from scipy.spatial import distance
from tqdm import tqdm

from argU import settings
from argU.database.mongodb import MongoDB
from argU.preprocessing.nlp import token_variants
from argU.utils.reader import get_queries


class EpochLogger(CallbackAny2Vec):
    def __init__(self, store_path=None, epochs_store=1):
        self.epoch = 1
        self.epochs_store = epochs_store
        self.last_time = None
        self.store_path = store_path

    def on_epoch_begin(self, model):
        self.last_time = time.time()

    def on_epoch_end(self, model):
        print(
            f"Epoch {self.epoch} - Time Duration: {(time.time() - self.last_time):.2f}s"
        )
        self.last_time = time.time()
        if self.epoch % self.epochs_store == 0 and self.store_path is not None:
            model.save(self.store_path)
            print("Model stored ...")
        self.epoch += 1


class CBOW:
    """Continuous Bag of Words Model"""

    def __init__(self):
        self.model = None
        self.default_emb = None

    def __str__(self):
        if self.model is None:
            return 'Model not initialized ...'
        else:
            return f"""
                Continuous Bag of Words Model
                -----------------------------
                
                Embedding length: {self.model.vector_size}
                Vocabulary size: {len(self.model.wv.vocab)}
                Example 'drugs' similarities: {self.model.most_similar(positive=['drugs'], topn=5)} ...
                Default embedding (First 5): {self.default_emb[:5]} ...
                Most sim. to default embedding: {self.model.wv.most_similar(positive=[self.default_emb], topn=5)} ...
            """

    @property
    def loaded(self):
        return self.model is not None

    @property
    def out_model(self):
        model_out = KeyedVectors(self.model.vector_size)
        model_out.vocab = self.model.wv.vocab
        model_out.index2word = self.model.wv.index2word
        model_out.vectors = self.model.trainables.syn1neg
        return model_out

    def switch_to_out_embedding(self):
        self.model = self.out_model

    def train(self, iterable, min_count=5, size=300, window=3):
        self.model = Word2Vec(
            iterable,
            size=size,
            window=window,
            min_count=min_count,
            workers=4,
            callbacks=[EpochLogger(store_path=settings.CBOW_PATH)]
        )
        self._init_default_emb()

    def store(self):
        self.model.save(settings.CBOW_PATH)

    def _init_default_emb(self):
        assert self.model is not None, 'Model has not been initialized yet!'

        emb = np.zeros((self.model.vector_size,))
        top_n = 100

        for token in self.model.wv.index2entity[:top_n]:
            emb += self.model.wv[token]

        self.default_emb = emb / top_n

    @staticmethod
    def load():
        try:
            cbow = CBOW()
            cbow.model = Word2Vec.load(settings.CBOW_PATH)
            cbow._init_default_emb()
            return cbow
        except Exception as e:
            print("Model could not be loaded ...")
            raise e


class InEmbedding:
    def __init__(self, *, cbow):
        self.model = self._init_model(cbow)
        self.default_emb = cbow.default_emb

    def _init_model(self, cbow):
        return cbow.model

    def text_to_emb(self, text):
        emb_matrix = self.text_to_emb_matrix(text)
        vec = self._emb_matrix_to_vec(emb_matrix)

        return vec

    def text_to_emb_matrix(self, text):
        tokens = text.split()
        emb_matrix = self._default_emb_matrix(tokens)

        for i, token in enumerate(tokens):
            emb_matrix[i] = self._token_to_emb(token)

        return emb_matrix

    def _default_emb_matrix(self, tokens):
        return np.zeros((len(tokens), self.model.vector_size))

    def _emb_matrix_to_vec(self, emb_matrix):
        return np.sum(emb_matrix, axis=0) / (emb_matrix.shape[0])

    def _token_to_emb(self, token):
        for tv in token_variants(token):
            if self._emb_exists(tv):
                return self._create_norm_emb(self.model.wv[tv])
        return self._create_norm_emb(self.default_emb)

    def _emb_exists(self, token):
        return token in self.model.wv.vocab

    def _create_norm_emb(self, emb):
        return emb / np.linalg.norm(emb)


class OutEmbedding(InEmbedding):
    def __init__(self, *, cbow):
        super().__init__(cbow=cbow)

    def _init_model(self, cbow):
        return cbow.out_model


class Desm:
    """
    Args:
        _cos_sim_matrix (:obj:`np.array`): rows = queries, columns = arguments, cell = cos similarity
    """

    def __init__(self, *, emb_type):
        self.emb_type = emb_type
        self._queries_emb_matrices = None
        self._cos_sim_matrix = None
        self._queries_dict = {}
        self._query_ids = []
        self._arg_ids = []
        self._init()

    def __str__(self):
        return f"""
            DESM Model similarities
            -----------------------
            
            Cos-Sim Matrix: {self._cos_sim_matrix.shape}
            Queries: {len(self.query_ids)}
            Arguments: {len(self.arg_ids)}
        """

    @property
    def query_ids(self):
        return self._query_ids

    @property
    def arg_ids(self):
        return self._arg_ids

    @property
    def cos_sim_matrix(self):
        return self._cos_sim_matrix

    def query_results_iterator(self, *, args_topn=100):
        for query_id, query_arguments_cos_sims in zip(self._query_ids, self._cos_sim_matrix):
            yield query_id, self._query_topn_args(query_arguments_cos_sims, args_topn)

    def print_examples(self, *, queries_num, args_topn):
        mongo_db = MongoDB()
        for i, (query_id, top_query_args) in enumerate(self.query_results_iterator(args_topn=args_topn)):
            if i == queries_num:
                break
            print(f'{query_id}) "{self._queries_dict[query_id]}"')
            for arg in top_query_args:
                id = arg['arg_id']
                arg = mongo_db.get_arg_by_id(id)
                print(f'\t{id}: {arg["premises"][0]["model_text"][:140]}')

    def _query_topn_args(self, cos_sims, args_topn):
        topn_ranked_idxs = self._topn_ranked_idxs_by_cos_sim(cos_sims, args_topn)

        topn_arg_ids = self._get_ranked_arg_ids(topn_ranked_idxs)
        topn_cos_sims = self._get_ranked_cos_sims(cos_sims, topn_ranked_idxs)

        return self._args_cos_sims_list(topn_arg_ids, topn_cos_sims)

    def _args_cos_sims_list(self, arg_ids, cos_sims):
        args_cos_sims_list = []

        for rank, (arg_id, cos_sim) in enumerate(zip(arg_ids, cos_sims)):
            args_cos_sims_list.append({
                'arg_id': arg_id,
                'rank': rank,
                'cos_sim': cos_sim,
            })

        return args_cos_sims_list

    def _topn_ranked_idxs_by_cos_sim(self, cos_sims, args_topn):
        return np.argsort(cos_sims)[::-1][:args_topn]

    def _get_ranked_cos_sims(self, cos_sims, ranked_ids):
        return cos_sims[ranked_ids]

    def _get_ranked_arg_ids(self, ranked_ids):
        return [self._arg_ids[idx] for idx in ranked_ids]

    def _init(self):
        self._init_queries()
        self._init_cos_sim_matrix()
        self._transpose_cos_sim_matrix()

    def _transpose_cos_sim_matrix(self):
        self._cos_sim_matrix = np.transpose(self._cos_sim_matrix)

    def _init_queries(self):
        cbow = CBOW.load()
        queries = get_queries(cbow)

        self._queries_dict = self._create_queries_dict(queries)
        self._queries_emb_matrices = self._create_queries_emb_matrices(
            queries=queries,
            emb_model=InEmbedding(cbow=cbow),
        )

    def _create_queries_dict(self, queries):
        queries_dict = {}
        for query in queries:
            queries_dict[query.id] = query.text
        return queries_dict

    def _create_queries_emb_matrices(self, *, queries, emb_model):
        queries_emb_matrices = []

        for query in queries:
            self._query_ids.append(query.id)
            emb_matrix = emb_model.text_to_emb_matrix(query.text)
            queries_emb_matrices.append(emb_matrix)

        return queries_emb_matrices

    def _init_cos_sim_matrix(self):
        cos_sim_matrix = []

        for arg in tqdm(MongoDB().args_coll.find().limit(100)):
            self._arg_ids.append(arg['id'])
            cos_sim_matrix.append(self._get_arg_queries_cos_sims(arg))

        self._cos_sim_matrix = cos_sim_matrix

    def _get_arg_queries_cos_sims(self, arg):
        arg_emb = self._get_arg_emb(arg)
        arg_queries_cos_sims = [self._arg_query_cos_sim(arg_emb, qem) for qem in self._queries_emb_matrices]

        return np.array(arg_queries_cos_sims)

    def _get_arg_emb(self, arg):
        return np.array(arg['premises'][0][self.emb_type])

    def _arg_query_cos_sim(self, arg_emb, query_matrix):
        """
        Args:
            query_matrix (:obj:`numpy.array`): embedded query as matrix
            arg_emb (:obj:`numpy.array`): embedded argument as vector

        Returns:
            float: dual embedding similarity
        """
        cos_sims = 1 - distance.cdist(query_matrix, np.expand_dims(arg_emb, axis=0), 'cosine')
        cos_sims_sum = sum(cos_sims) / len(cos_sims)
        result = 0.0 if np.isnan(cos_sims_sum[0]) else cos_sims_sum[0]

        return result
