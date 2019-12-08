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

    def most_similar(self, word, topn=5):
        word_embedding = self.w2v_model.wv[word]
        arguments = []
        similarities = []

        for (argument_id, argument_embedding) in self.av.items():
            sim = 1 - spatial.distance.cosine(
                word_embedding, argument_embedding
            )
            similarities.append(sim)
            arguments.append(argument_id)

        similarities = np.asarray(similarities)
        best_indices = np.argpartition(similarities, -topn)[-topn:]
        best_indices = best_indices[np.argsort(similarities[best_indices])]

        best_arguments = []
        for i in best_indices:
            best_arguments.append(arguments[i])

        return best_arguments

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
                argument_vector = embedding_matrix.sum(
                    axis=0) / len(argument.text)
                self.av[argument.id] = argument_vector
