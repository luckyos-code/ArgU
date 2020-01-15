
import json
import time
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sklearn import preprocessing

from utils.reader import Argument
from indexing.rank_bm25 import BM25Okapi


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, store_path=None, epochs_store=1):
        self.epoch = 1
        self.epochs_store = epochs_store
        self.last_time = None
        self.store_path = store_path

    def on_epoch_begin(self, model):
        self.last_time = time()

    def on_epoch_end(self, model):
        print(
            f"Epoch {self.epoch} - Time Duration: {(time() - self.last_time):.2f}s"
        )
        self.last_time = time()
        if self.epoch % self.epochs_store == 0 and self.store_path is not None:
            model.save(self.store_path)
            print("Modell gespeichert...")
        self.epoch += 1


class CBOW:
    def __init__(self):
        self.window = 3
        self.size = 300
        self.model = None

    def build(self, iterable, min_count=5, store_path=None):
        self.model = Word2Vec(
            iterable,
            size=self.size,
            window=self.window,
            min_count=min_count,
            workers=4,
            callbacks=[EpochLogger(store_path=store_path)]
        )

    def store(self, path):
        tick = time.time()

        self.model.save(path)

        time_spent = time.time() - tick
        print(f"CBOW: Benötigte zeit zum Speichern: {time_spent:.2f}s")

    @staticmethod
    def load(path):

        print("Lade CBOW Modell...")
        tick = time.time()

        cbow = CBOW()
        cbow.model = Word2Vec.load(path)

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

    def build(self, train_csv_iterator):
        """Erstelle ein neues BM25Kapi Objekt. Texte und IDs
            werden im Manager zusätzlich gespeichert.

        Args:
            train_csv_iterator (:obj:`TrainCSVIterator`): Iterator
                für vorverarbeitete Argumente
        """

        self.index = BM25Okapi(train_csv_iterator)

    # def norm_scores(self, query):
    #     """Bestimme die normalisierten Scores für alle Argumente

    #     Args:
    #         query (:obj:`str`): Eingabe

    #     Returns:
    #         dict (:obj:`str`: float): Argument IDs und dazugehörige
    #             normalisierte und absteigend sortierte Scores.
    #     """

    #     query_splitted = query.split()
    #     scores = self.index.get_scores(query_splitted)
    #     scores = preprocessing.normalize([scores])

    #     scores = {arg: score for (arg, score) in zip(
    #         self.index.arg_ids, scores[0])}
    #     scores = {k: v for k, v in sorted(
    #         scores.items(), key=lambda item: item[1], reverse=True)}

    #     return scores

    @staticmethod
    def load(path):
        """Lade einen als JSON gespeicherten `BM25Manager`

        Args:
            path (:obj:`str`): Dateipfad
        """
        
        print("Lade BM25 Modell...")

        with open(path, 'r') as f_in:

            tick = time.time()
            data = json.load(f_in)

            bm25_manager = BM25Manager()

            bm25_manager.index = BM25Okapi()
            bm25_manager.index.idf = data['idf']
            bm25_manager.index.corpus_size = data['corpus_size']
            bm25_manager.index.doc_len = data['doc_len']
            bm25_manager.index.doc_freqs = data['doc_freqs']
            bm25_manager.index.k1 = data['k1']
            bm25_manager.index.b = data['b']
            bm25_manager.index.avgdl = data['avgdl']
            bm25_manager.index.arg_ids = data['arg_ids']

            time_spent = time.time() - tick
            print(
                f"BM25: Benötigte zeit zum Laden von {len(bm25_manager.index.arg_ids)} Argumenten: {time_spent:.2f}s")

            return bm25_manager

    def store(self, path):
        """Speichere diesen `BM25Manager`

        Args:
            path (str): Dateipfad
        """

        tick = time.time()
        payload = {
            'idf': self.index.idf,
            'corpus_size': self.index.corpus_size,
            'doc_len': self.index.doc_len,
            'doc_freqs': self.index.doc_freqs,
            'k1': self.index.k1,
            'b': self.index.b,
            'avgdl': self.index.avgdl,
            'arg_ids': self.index.arg_ids
        }

        with open(path, 'w') as f_out:
            json.dump(payload, f_out)

        time_spent = time.time() - tick
        print(
            f"BM25: Benötigte zeit zum Speichern von {len(self.index.arg_ids)} Argumenten: {time_spent:.2f}s")

    # def get_top_n_ids(self, query, top_n=10):
    #     """Suche für die gegebene Query die besten `top_n` Ergebnisse

    #     Args:
    #         query (:obj:`str`): Anfrage
    #         top_n (int): Die Top N Ergebnisse

    #     Returns:
    #         list: Gefundene Indizes sortiert nach Relevanz
    #     """

    #     return self.index.get_top_n(query.split(), self.argument_ids, n=top_n)


class DualEmbedding:
    def __init__(self, model_in):
        self.model_in = model_in
        self.model_out = None
        self.vector_size = -1

    def build(self):
        """Erstelle KeyVectors für w_IN und w_OUT"""

        self.vector_size = self.model_in.trainables.layer1_size

        self.model_out = KeyedVectors(self.vector_size)
        self.model_out.vocab = self.model_in.wv.vocab
        self.model_out.index2word = self.model_in.wv.index2word
        self.model_out.vectors = self.model_in.trainables.syn1neg

    def desim(self, query, arg_emb, q_type='in'):
        """ Berechnung der Ähnlichkeit zwischen Query und Argument für IN und OUT Embeddings

        Args:
            query (list): Vorverarbeitete Wörter der Query
            arg_emb (numpy.array): Argument Text Embedding
            q_type (str): Query Type, entweder 'in' oder 'out'
            arg_type (str): Argument Type, entweder 'in' oder 'out'

        Returns:
            float: dual embedding similarity
        """

        norm_factor = 1 / len(query)
        cos_sum = 0

        query_model = self.model_in if q_type == 'in' else self.model_out

        for q_term in query:
            try:
                cos_sum += 1 - distance.cosine(
                    np.transpose(query_model.wv[q_term]),
                    arg_emb,
                )
            except KeyError as ke:
                pass
                # print(f'{q_term} -> kein Embedding')

        return norm_factor * cos_sum


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
