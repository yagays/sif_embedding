# https://www.kaggle.com/procode/sif-embeddings-got-69-accuracy
# https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py

import logging

import itertools
from collections import Counter

import MeCab
import numpy as np
from pyflann import FLANN
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger("sif")


def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split(" ")


def map_word_frequency(document):
    return Counter(itertools.chain(*document))


class SifEmbedding():
    def __init__(self, embedding_file, a=1e-3, tokenize=tokenize):
        self.word2vec_file = embedding_file
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_file, binary=True)
        self.embedding_dim = self.word2vec.vector_size
        self.tokenize = tokenize
        self.a = a
        self.sentence_list = []
        self.sentence_list_tokenized = []
        self.word_counts = Counter()
        self.sentence_embedding = np.array([])
        self.flann = FLANN()

    def _weighted_bow(self, sentence):
        vs = np.zeros(self.embedding_dim)
        sentence_length = 0

        for word in sentence:
            a_value = self.a / (self.a + self.word_counts[word])  # smooth inverse frequency, SIF
            try:
                vs = np.add(vs, np.multiply(a_value, self.word2vec[word]))  # vs += sif * word_vector
                sentence_length += 1
            except Exception:
                logger.debug(f"Embedding Vector: {word} not found")

        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)

        return vs

    def _fit_svd(self, X):
        svd = TruncatedSVD(n_components=1, n_iter=100, random_state=0)
        svd.fit(X)
        self.u = svd.components_
        return self

    def _transform_svd(self, X):
        vs = X - X.dot(self.u.transpose()) * self.u
        return vs

    def _fit_transform_svd(self, X):
        return self._fit_svd(X)._transform_svd(X)

    def fit(self, sentence_list):
        for sentence in sentence_list:
            self.sentence_list.append(sentence)
            self.sentence_list_tokenized.append(self.tokenize(sentence))

        self.word_counts = map_word_frequency(self.sentence_list_tokenized)

        # Alg.1 step 1
        sentence_vec = []
        for sentence in self.sentence_list_tokenized:
            sentence_vec.append(self._weighted_bow(sentence))

        # Alg.1 step 2
        self.sentence_embedding = self._fit_transform_svd(np.array(sentence_vec))

        # make index for similarity search
        self.flann.build_index(self.sentence_embedding)

    def infer_vector(self, sentence):
        return self._transform_svd(self._weighted_bow(self.tokenize(sentence)))

    def predict(self, sentence, topn=1):
        vs = self.infer_vector(sentence)
        result, dists = self.flann.nn_index(vs, num_neighbors=topn)

        if topn != 1:
            result = result[0]
            dists = dists[0]

        output = []
        for i, index in enumerate(result.tolist()):
            text = self.sentence_list[index]
            sim = dists[i]
            output.append([text, sim])
        return output
