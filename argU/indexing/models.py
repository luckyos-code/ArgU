
import csv
import json
import os
import sys
import rootpath
import time
from tqdm import tqdm

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from scipy.spatial import distance

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

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
            print("Modell gespeichert...")
        self.epoch += 1


class CBOW:
    """Continuous Bag of Words Model"""

    def __init__(self):
        self.model = None

    @property
    def loaded(self):
        return self.model is not None

    def build(self, iterable, min_count=5, size=300, window=3):
        self.model = Word2Vec(
            iterable,
            size=size,
            window=window,
            min_count=min_count,
            workers=4,
            callbacks=[EpochLogger(store_path=setup.CBOW_PATH)]
        )

    def store(self):
        """ Store CBOW model """

        tick = time.time()
        self.model.save(setup.CBOW_PATH)
        time_spent = time.time() - tick
        print(f"Time to store CBOW: {time_spent:.2f}s")

    @staticmethod
    def load():

        print("Load CBOW...")
        tick = time.time()
        cbow = CBOW()
        cbow.model = Word2Vec.load(setup.CBOW_PATH)
        time_spent = time.time() - tick

        print(f"Time needed: {time_spent:.2f}s")
        return cbow


diff_words = set([])


class DESM:
    def __init__(self, cbow):
        self.vector_size = cbow.model.vector_size
        self.model_in = cbow.model
        self.model_out = self.__init_model_out()

    def __init_model_out(self):
        """Create KeyVectors for w_OUT"""

        model_out = KeyedVectors(self.vector_size)
        model_out.vocab = self.model_in.wv.vocab
        model_out.index2word = self.model_in.wv.index2word
        model_out.vectors = self.model_in.trainables.syn1neg
        return model_out

    def __get_term_variants(self, term):
        return [
            term,
            term[0].lower() + term[1:],
            term.lower(),
        ]

    def arg_to_emb(self, arg_train, model_type='in'):
        if model_type == 'in':
            wv = self.model_in.wv
        elif model_type == 'out':
            wv = self.model_out
        else:
            assert False, 'Wrong model type. Choose in or out...'

        text = arg_train['text'].split()
        emb_matrix = np.zeros(
            (len(text), wv.vector_size)
        )

        unk = 0
        for i, term in enumerate(text):
            term_vars = self.__get_term_variants(term)
            for tv in term_vars:
                if tv in wv.vocab:
                    emb = wv[tv]
                    emb_matrix[i] = emb / np.linalg.norm(emb)
                    break
                if i == len(term_vars):
                    unk += 1

        vec = np.sum(emb_matrix, axis=0) / (emb_matrix.shape[0])

        # print(wv.most_similar(positive=emb_matrix, topn=40))
        # print(f'Unk. count: {unk}')

        # for (w, s) in wv.most_similar(positive=[vec], topn=10):
        # diff_words.add(w)
        # print(diff_words)
        # print(len(diff_words))
        return vec

    def queries_to_emb(self, queries, model_type='in'):
        if model_type == 'in':
            model = self.model_in
        elif model_type == 'out':
            model = self.model_out
        else:
            assert False, 'Wrong model type. Choose in or out...'

        query_embs = []
        unk_all = 0
        for query in queries:
            terms = query.text.split()
            emb_matrix = np.zeros((len(terms), model.vector_size))
            unk = 0
            for i, term in enumerate(terms):
                term_vars = self.__get_term_variants(term)
                for tv in term_vars:
                    if tv in model.wv.vocab:
                        emb = model.wv.word_vec(tv)
                        emb_matrix[i] = emb
                        break
                    if i == len(term_vars):
                        unk += 1

            print((
                f'[{query.id}] {query.text}  {emb_matrix.shape} -> '
                f'{unk} von {len(query.text.split())} Wörtern unbekannt'
            ))
            # for i, emb in enumerate(emb_matrix):
            # most_sim = model.wv.most_similar(positive=[emb], topn=4)
            # print(f'\t{terms[i]} -> {most_sim}')
            print()
            unk_all += unk
            query_embs.append(emb_matrix)
        print(f'Number of queries: {len(query_embs)}')
        print(f'Number of unknown words: {unk_all}')
        return query_embs

    def evaluate_queries(self, query_matrices, coll_emb, top_n=500, max_args=-1):
        resulting_scores = []
        arg_ids = []
        for i, arg in tqdm(enumerate(coll_emb.find())):
            if i == max_args:
                break
            arg_ids.append(arg['_id'])
            resulting_scores.append(np.array(
                [self.__get_scores(qm, arg['emb']) for qm in query_matrices]
            ))

        arg_ids = np.array(arg_ids)

        top_args = []
        for query_scores in np.transpose(resulting_scores):
            top_ids = np.argsort(query_scores)[::-1][:top_n]
            top_args.append(arg_ids[top_ids])

        return top_args

    def store_query_results(self, coll, queries, args):
        assert len(queries) == len(args)

        coll.drop()
        for q, args in zip(queries, args):
            coll.insert_one({
                'query_id': q.id,
                'query_text': q.text,
                'args': args.tolist(),
            })
        print(coll.find_one({}))

    def __get_scores(self, query_matrix, arg_emb):
        """ Dual Embedding similarity for query and args

        Args:
            query_matrix (:obj:`numpy.array`): embedded queries
            arg_emb (:obj:`numpy.array`): argument embedding

        Returns:
            float: dual embedding similarity
        """

        arg_emb = np.array(arg_emb)
        cos_sims = 1 - distance.cdist(
            query_matrix,
            np.expand_dims(arg_emb, axis=0),
            'cosine'
        )

        cos_sum = sum(cos_sims) / len(cos_sims)
        if np.isnan(cos_sum[0]):
            return 0.0
        else:
            return cos_sum[0]
