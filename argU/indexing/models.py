
import csv
import json
import os
import sys
import rootpath
import time
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sklearn import preprocessing

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.utils.reader import Argument
    from argU.utils.reader import TrainArgsIterator
    from argU.indexing.rank_bm25 import BM25Okapi
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

    def build(self, min_count=5, size=300, window=3):
        self.model = Word2Vec(
            TrainArgsIterator(only_texts=True),
            size=size,
            window=window,
            min_count=min_count,
            workers=4,
            callbacks=[EpochLogger(store_path=setup.CBOW_PATH)]
        )

    def store(self):
        tick = time.time()
        self.model.save(setup.CBOW_PATH)
        time_spent = time.time() - tick

        print(f"CBOW: Benötigte zeit zum Speichern: {time_spent:.2f}s")

    @staticmethod
    def load():

        print("Lade CBOW Modell...")
        tick = time.time()
        cbow = CBOW()
        cbow.model = Word2Vec.load(setup.CBOW_PATH)
        time_spent = time.time() - tick

        print(f"CBOW: Benötigte zeit zum Laden: {time_spent:.2f}s")
        return cbow


class BM25Manager:
    """BM25Kapi Manager

    Args:
        index (:obj:`BM25Kapi`): BM25 Algorithmus
    """

    def __init__(self):
        self.index = None

    @property
    def loaded(self):
        return self.index is not None

    def build(self, max_args=-1):
        """Erstelle ein neues BM25Kapi Objekt. Texte und IDs
            werden im Manager zusätzlich gespeichert.
        """

        self.index = BM25Okapi(TrainArgsIterator(max_args=max_args))

    @staticmethod
    def load(mode=None):
        """Lade einen als JSON gespeicherten `BM25Manager`

        Args:
            mode (:obj:`str`, optional): Entweder None, 'meta' oder 'args'. Manchmal braucht
                man nicht das ganze Modell
        """

        print("Lade BM25 Modell...")
        tick = time.time()

        bm25_manager = BM25Manager()
        bm25_manager.index = BM25Okapi()

        if mode is None or mode == 'meta':
            with open(setup.BM25_META_PATH, 'r') as f_in:
                meta_data = json.load(f_in)

                bm25_manager.index.idf = meta_data['idf']
                bm25_manager.index.corpus_size = meta_data['corpus_size']
                bm25_manager.index.k1 = meta_data['k1']
                bm25_manager.index.b = meta_data['b']
                bm25_manager.index.avgdl = meta_data['avgdl']

        if mode is None or mode == 'args':
            with open(
                setup.BM25_DOCS_PATH, 'r', newline='', encoding='utf-8'
            ) as f_in:
                reader = csv.reader(f_in, setup.BM25_DOCS_CONFIG)

                for a_id, dl, df in tqdm(reader):
                    bm25_manager.index.arg_ids.append(a_id)
                    bm25_manager.index.doc_len.append(int(dl))
                    bm25_manager.index.doc_freqs.append(json.loads(df))

        time_spent = time.time() - tick

        if mode is None or mode == 'args':
            print(
                f"BM25: Benötigte zeit zum Laden von "
                f"{len(bm25_manager.index.arg_ids)} "
                f"Zeit: {time_spent:.2f}s"
            )
        else:
            print('Metadaten für BM25Kapi wurden geladen...')
            print(f"Zeit: {time_spent:.2f}s")

        return bm25_manager

    def store(self):
        """Speichere diesen `BM25Manager`

        Args:
            path (str): Dateipfad
        """

        tick = time.time()

        meta_payload = {
            'idf': self.index.idf,
            'corpus_size': self.index.corpus_size,
            'k1': self.index.k1,
            'b': self.index.b,
            'avgdl': self.index.avgdl,
        }

        with open(
            setup.BM25_DOCS_PATH, 'w', newline='', encoding='utf-8'
        ) as f_out:
            writer = csv.writer(f_out, setup.BM25_DOCS_CONFIG)
            for i, (a_id, dl, df) in tqdm(enumerate(zip(
                self.index.arg_ids, self.index.doc_len, self.index.doc_freqs
            ))):
                writer.writerow([
                    a_id, dl, json.dumps(df)
                ])

        time_spent = time.time() - tick
        print((
            f"BM25: Benötigte zeit zum Speichern von "
            f"{len(self.index.arg_ids)} Argumenten: {time_spent:.2f}s"
        ))


class DualEmbedding:
    def __init__(self, model_in):
        self.model_in = model_in
        self.model_out = self.__init_model_out()
        self.vector_size = model_in.trainables.layer1_size

    def __init_model_out(self):
        """Erstelle KeyVectors für w_IN und w_OUT"""

        self.vector_size = self.model_in.trainables.layer1_size

        model_out = KeyedVectors(self.vector_size)
        model_out.vocab = self.model_in.wv.vocab
        model_out.index2word = self.model_in.wv.index2word
        model_out.vectors = self.model_in.trainables.syn1neg

    def get_processed_queries(self, queries, q_type='in'):
        model = self.model_in if q_type == 'in' else self.model_out
        processed_queries = []

        for j, query in enumerate(queries):
            terms = query.split()
            matrix = np.zeros((len(terms), self.vector_size), dtype=float)
            unk = 0
            for i, term in enumerate(terms):
                if term in model.wv:
                    matrix[i] = model.wv[term]
                else:
                    unk += 1
            print(f"Query {j}: {unk} von {len(terms)} Wörtern sind unbekannt")
            processed_queries.append(
                (terms, matrix)
            )
        return processed_queries

    def desim(self, query_matrix, arg_emb):
        """ Berechnung der Ähnlichkeit zwischen Query
            und Argument für IN und OUT Embeddings

        Args:
            query_matrix (np.array): Vorverarbeitete Wörter der Query
            arg_emb (numpy.array): Argument Text Embedding

        Returns:
            float: dual embedding similarity
        """

        cos_sims = distance.cdist(
            query_matrix,
            np.expand_dims(arg_emb, axis=0),
            'cosine'
        )
        cos_sum = sum(cos_sims) / len(cos_sims)
        if np.isnan(cos_sum[0]):
            return 0.0
        else:
            return cos_sum[0]


class Argument2Vec:
    def __init__(self, cbow):
        self.cbow = cbow
        self.av = dict()

    def build(self, train_csv_iterator):
        if self.cbow is None:
            print('Model kann nicht erstellt werden...')
            return False

        vector_size = self.cbow.model.vector_size

        for row in tqdm(train_csv_iterator):
            arg_id = row[0]
            arg_text = row[1]

            arg_emb, unk = Argument.to_vec(
                arg_text, self.cbow.model, vector_size)
            self.av[arg_id] = (arg_emb, unk)

    # def most_similar(self, query, topn=5):
    #     query_embeddings = [self.w2v_model.wv[word] for word in query.split()]
    #     if topn == -1:
    #         topn = len(self.tv)

    #     ids = []
    #     similarities = []

    #     for (id, arg_emb) in self.tv.items():
    #         cos_sim = 0
    #         for q_emb in query_embeddings:
    #             cos_sim += distance.cosine(np.transpose(q_emb), arg_emb)

    #         cos_sim *= (1 / len(query_embeddings))
    #         similarities.append(cos_sim)
    #         ids.append(id)

    #     similarities = np.asarray(similarities)
    #     best_indices = np.argpartition(similarities, -topn)[-topn:]
    #     best_indices = best_indices[np.argsort(similarities[best_indices])]

    #     best_texts = []
    #     for i in best_indices:
    #         best_texts.append(ids[i])

    #     return best_texts, np.sort(similarities)[::-1]

    @staticmethod
    def load(path):
        """Falls es einen Index gibt, der gespeichert wurde, kann dieser hier
        geladen werden. Dabei wird `bm25_index` definiert.

        Args:
            path (str): Pfad zur Datei
        """

        tick = time.time()

        a2v = Argument2Vec(None)
        with open(path, 'r') as f_in:
            a2v.av = json.load(f_in)
            for key, (v1, v2) in a2v.av.items():
                if v1 is None:
                    print(key, v1, v2)
                else:
                    a2v.av[key] = (np.asarray(v1), v2)

        time_spent = time.time() - tick
        print(
            f"A2V: Benötigte zeit zum Laden von {len(a2v.av)} Argumenten: {time_spent:.2f}s")

        return a2v

    def store(self, path):
        """Speichere den Index, falls dieser existiert, in eine Datei

        Args:
            path (str): Dateipfad
        """

        tick = time.time()

        for key, (v1, v2) in self.av.items():
            try:
                self.av[key] = (v1.tolist(), v2)
            except Exception as e:
                print(key, v1, v2)
                print(e)
        with open(path, 'w') as f_out:
            json.dump(self.av, f_out)

        time_spent = time.time() - tick
        print(
            f"A2V: Benötigte zeit zum Speichern von {len(self.av)} Argumenten: {time_spent:.2f}s")


class MixtureModel:
    def __init__(self, a2v, bm25_manager):
        self.bm25_manager = bm25_manager
        self.a2v = a2v

    def mmsims(self, query, alpha=0.75):
        scores = self.bm25_manager.norm_scores(query)
        dual_embedding = DualEmbedding(self.a2v.w2v_model)
        dual_embedding.build()

        for (id, (emb, _)) in self.a2v.av.items():
            de_sim = dual_embedding.desim(query, emb)
            bm25_sim = scores[id]
            mm_sim = alpha * de_sim + (1 - alpha) * bm25_sim
            scores[id] = (bm25_sim, de_sim, mm_sim)

        keys_to_remove = [k for k in scores if isinstance(scores[k], float)]
        for k in keys_to_remove:
            del scores[k]  # Für diese existiert kein Dual Embedding

        return scores
