from utils.reader import read_arguments
from collections import defaultdict, Counter
import numpy as np
from utils.reader import ArgumentTextsIterator
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from time import time


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
