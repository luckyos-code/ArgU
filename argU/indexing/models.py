<<<<<<< HEAD
from utils.reader import read_arguments
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
from numpy import linalg as LA
from utils.reader import ArgumentTextIterator, ArgumentIterator
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from time import time
from scipy.spatial import distance
from tqdm import tqdm
from scipy import spatial


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
        assert self.model is not None
        self.model.save(path)
        print("Modell gespeichert...")

    def load(self, path):
        self.model = Word2Vec.load(path)


class BM25Manager:
    """Index zur Bestimmung relevanter Argumente für eine Query"""

    def __init__(self):
        self.index = None
        self.argument_texts = []
        self.argument_ids = []

    def build(self, path, max_args=-1):
        """Erstelle einen neuen BM25Kapi

        Args:
            path (str): CSV path
            max_args (int): Menge der Argumente. -1 = alle
        """

        for argument in ArgumentIterator(path, max_args):
            self.argument_texts.append(argument.text)
            self.argument_ids.append(argument.id)

        self.index = BM25Okapi(self.argument_texts)

    @staticmethod
    def load(path):
        """Falls es einen Index gibt, der gespeichert wurde, kann dieser hier
        geladen werden. Dabei wird `bm25_index` definiert.

        Args:
            path (str): Pfad zur Datei
        """

        with open(path, 'rb') as f_in:
            return pickle.load(f_in)

    def store(self, path):
        """Speichere den Index, falls dieser existiert, in eine Datei

        Args:
            path (str): Dateipfad
        """

        assert self.index is not None
        with open(path, 'wb') as f_out:
            pickle.dump(self, f_out)

    def get_top_n_ids(self, query, top_n=10):
        """Suche für die gegebene Query die top `top_n` Ergebnisse

        Args:
            query (str): Anfrage
            top_n (int): Die Top N Ergebnisse

        Returns:
            list: Gefundene Indizes sortiert nach Relevanz
        """

        assert self.index is not None
        return self.index.get_top_n(query.split(), self.argument_ids, n=top_n)

    def norm_scores(self, query):
        """Bestimme die Scores bezüglich aller Argumente

        Args:
            query (str): Eingabe

        Returns:
            dict: Argument IDs und dazugehörige, normalisierte Scores
        """

        scores = self.index.get_scores(query.split())
        scores = scores / LA.norm(scores)
        return {arg: score for (arg, score) in zip(self.argument_ids, scores)}


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
        query = query.split()

        for q_term in query:
            cos_sum += 1 - distance.cosine(
                np.transpose(query_model.wv[q_term]),
                arg_emb,
            )

        return norm_factor * cos_sum


class Argument2Vec:
    def __init__(self, w2v_model, arguments_path):
        self.w2v_model = w2v_model
        self.arguments_path = arguments_path

        self.av = dict()

    def _is_valid(self, argument):
        valid = len(argument.text) > 5
        return valid

    def build(self, max_args=-1):
        vector_size = self.w2v_model.vector_size

        for argument in tqdm(ArgumentIterator(
            self.arguments_path, max_args)
        ):
            if self._is_valid(argument):
                arg_emb, unk_words = argument.get_vec(
                    self.w2v_model, vector_size
                )
                self.av[argument.id] = (arg_emb, unk_words)

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

        with open(path, 'rb') as f_in:
            return pickle.load(f_in)

    def store(self, path):
        """Speichere den Index, falls dieser existiert, in eine Datei

        Args:
            path (str): Dateipfad
        """

        with open(path, 'wb') as f_out:
            pickle.dump(self, f_out)


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
=======
from utils.reader import read_arguments
from collections import defaultdict, Counter
import numpy as np
from utils.reader import ArgumentTextsIterator, ArgumentIterator
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from time import time
from tqdm import tqdm
from scipy import spatial


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, store_path=None, epochs_store=1):
        self.epoch = 1
        self.store_path = store_path
        self.epochs_store = epochs_store
        self.last_time = None

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
    def __init__(self, min_count=3):
        self.window = 3
        self.size = 300
        self.min_count = min_count
        self.model = None

    def build(self, path, max_args=100, store_path=None):
        arguments_iter = ArgumentTextsIterator(path, max_args)
        self.model = Word2Vec(
            arguments_iter,
            size=self.size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            callbacks=[EpochLogger(store_path=store_path)]
        )

    def store(self, path):
        assert self.model is not None
        self.model.save(path)
        print("Modell gespeichert...")

    def load(self, path):
        self.model = Word2Vec.load(path)


class Argument2Vec:
    def __init__(self, w2v_model, path, max_args=100):
        """Modell, das aus Argumenten einen Vektor generiert

        Arguments:
            arguments_iterator (Argument, iterable): Vorverarbeitete Argumente
            w2v_model (Word2Vec): Moodell für Word-Embeddings
        """

        self.arguments_iterator = ArgumentIterator(path, max_args)
        self.w2v_model = w2v_model
        self.av = dict()
        self._build()

    def load(self, path):
        pass

    def save(self, path):
        pass

    def most_similar(self, query, topn=5):
        query_embeddings = [self.w2v_model.wv[word] for word in query.split()]

        arguments = []
        similarities = []

        for (argument_id, argument_embedding) in self.av.items():
            sim = 0
            for query_embedding in query_embeddings:
                a = np.dot(np.transpose(query_embedding), argument_embedding)
                b = np.linalg.norm(query_embedding) * np.linalg.norm(
                    argument_embedding
                )
                sim += a / b

            sim *= (1 / len(query_embeddings))

            similarities.append(sim)
            arguments.append(argument_id)

        similarities = np.asarray(similarities)
        best_indices = np.argpartition(similarities, -topn)[-topn:]
        best_indices = best_indices[np.argsort(similarities[best_indices])]

        best_arguments = []
        for i in best_indices:
            best_arguments.append(arguments[i])

        return best_arguments

    def _centeroid(self, data):
        length, dim = data.shape
        return np.array([np.sum(data[:, i])/length for i in range(dim)])

    def _build(self):
        vector_size = self.w2v_model.vector_size
        for argument in tqdm(self.arguments_iterator):
            if argument.text and len(argument.text) > 5:
                embedding_matrix = np.zeros((len(argument.text), vector_size))
                for i, word in enumerate(argument.text):
                    try:
                        embedding_matrix[i] = self.w2v_model.wv[word]
                    except Exception as e:
                        pass
                centeroid = self._centeroid(embedding_matrix)
                self.av[argument.id] = centeroid
>>>>>>> rank_test
