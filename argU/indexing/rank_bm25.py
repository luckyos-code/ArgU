# Überarbeitetes Package, um BM25 auszuführen
# Ursprung: https://pypi.org/project/rank-bm25/

import math
from tqdm import tqdm
import numpy as np

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, iterable_corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.arg_ids = []

        if iterable_corpus is not None:
            nd = self._initialize(iterable_corpus)
            self._calc_idf(nd)

    def _initialize(self, iterable_corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for (id, document) in tqdm(iterable_corpus):
            self.corpus_size += 1
            self.arg_ids.append(id)
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(
            documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, iterable_corpus=None, tokenizer=None):
        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25
        super().__init__(iterable_corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in tqdm(nd.items()):
            idf = math.log(self.corpus_size - freq + 0.5) - \
                math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in tqdm(negative_idfs):
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_single_score(self, query, doc_freq, doc_len):
        """Gleicher Algorithmus, jedoch wwerden Argumente schrittweise verarbeitet"""

        score = 0
        for q in query:
            q_freq = doc_freq.get(q, 0)
            score += self.idf.get(q, 0) * (
                (q_freq * (self.k1 + 1)) / (
                    q_freq + self.k1 * (
                        1 - self.b + self.b * doc_len / self.avgdl
                    )
                )
            )
        return score
