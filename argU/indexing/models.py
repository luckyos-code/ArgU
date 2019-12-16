from utils.reader import read_arguments
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
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

    def build(self, iterable, min_count=3, store_path=None):
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


class BM25:
    """Index zur Bestimmung relevanter Argumente für eine Query"""

    def __init__(self):
        self.index = None
        self.argument_texts = []
        self.argument_ids = []

    def build(self, path, max_args=100):
        for argument in ArgumentIterator(path, max_args):
            self.argument_texts.append(argument.text)
            self.argument_ids.append(argument.id)

        self.index = BM25Okapi(self.argument_texts)

    def load(self, path):
        """Falls es einen Index gibt, der gespeichert wurde, kann dieser hier
        geladen werden. Dabei wird `bm25_index` definiert.

        Args:
            path (str): Pfad zur Datei
        """

        with open(path, 'rb') as f_in:
            self.index = pickle.load(f_in)

    def store(self, path):
        """Speichere den Index, falls dieser existiert, in eine Datei

        Args:
            path (str): Dateipfad
        """

        assert self.index is not None
        with open(index_path, 'wb') as f_out:
            pickle.dump(self.index, f_out)

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


class Text2Vec:
    def __init__(self, w2v_model, iterable_texts):
        """Modell, das aus beliebigen Texten einen Vektor generiert

        Arguments:
            iterable_texts (Text, iterable): Vorverarbeitete Argumente
            w2v_model (Word2Vec): Moodell für Word-Embeddings
        """

        self.w2v_model = w2v_model
        self.iterable_texts = iterable_texts

        self.tv = dict()
        self._build()

    def most_similar(self, query, topn=5):
        query_embeddings = [self.w2v_model.wv[word] for word in query.split()]
        if topn == -1:
            topn = len(self.tv)

        ids = []
        similarities = []

        for (id, text_embedding) in self.tv.items():
            sim = 0
            for query_embedding in query_embeddings:
                a = np.dot(np.transpose(query_embedding), text_embedding)
                b = np.linalg.norm(query_embedding) * np.linalg.norm(
                    text_embedding
                )
                sim += a / b

            sim *= (1 / len(query_embeddings))
            similarities.append(sim)
            ids.append(id)

        similarities = np.asarray(similarities)
        best_indices = np.argpartition(similarities, -topn)[-topn:]
        best_indices = best_indices[np.argsort(similarities[best_indices])]

        best_texts = []
        for i in best_indices:
            best_texts.append(ids[i])

        return best_texts, np.sort(similarities)[::-1]

    def _is_valid(self, input_text):
        return True

    def _build(self):
        vector_size = self.w2v_model.vector_size

        for iterable_text in tqdm(self.iterable_texts):
            if self._is_valid(iterable_text):
                embedding_matrix = np.zeros(
                    (len(iterable_text.text), vector_size)
                )

                unknown_word_count = 0
                unknown_words = []
                for i, word in enumerate(iterable_text.text):
                    try:
                        embedding_vec = self.w2v_model.wv[word]
                        embedding_vec = np.true_divide(
                            embedding_vec, np.linalg.norm(embedding_vec)
                        )
                        embedding_matrix[i] = embedding_vec
                    except Exception as e:
                        unknown_word_count += 1
                        unknown_words.append(word)
                centroid = np.sum(embedding_matrix, axis=0) / (
                    embedding_matrix.shape[0]
                )

                # print(
                #     f"UNK: {unknown_word_count} = {(unknown_word_count / len(argument.text)):.4f}"
                # )
                self.tv[iterable_text.id] = centroid


class Argument2Vec(Text2Vec):
    def __init__(self, w2v_model, iterable_texts):
        Text2Vec.__init__(self, w2v_model, iterable_texts)

    def _is_valid(self, argument):
        valid = len(argument.text) > 5
        return valid
