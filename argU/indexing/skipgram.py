from utils.reader import read_arguments
from collections import defaultdict, Counter
from tqdm import tqdm


class SkipGramModel:
    def __init__(self):
        self.dictionary = None
        self.reversed_dictionary = None

    def build_dicts(self, path, n_words, max_args=1000):
        """Process raw inputs into a dataset."""

        word_counts = defaultdict(int)
        for argument in tqdm(read_arguments(path, max_args)):
            for w in argument.text:
                word_counts[w] += 1

        word_counts = Counter(word_counts).most_common(n_words-1)
        word_counts.append(('UNK', -1))

        self.dictionary = dict()
        for word, _ in word_counts:
            self.dictionary[word] = len(self.dictionary)

        self.reversed_dictionary = dict(
            zip(self.dictionary.values(), self.dictionary.keys())
        )

    def text_to_indices(self, text):
        assert self.dictionary is not None

        indices = []
        for word in text:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = self.dictionary['UNK']
            indices.append(index)
        return indices
