
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pyflann import FLANN

from src.sif import tokenize
from ld_corpus import load_ld_corpus
from ld_corpus import topics


class W2VAverageEmbedding():
    def __init__(self, embedding_file, tokenize=tokenize):
        self.word2vec_file = embedding_file
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_file, binary=True)
        self.embedding_dim = self.word2vec.vector_size
        self.tokenize = tokenize
        self.sentence_list = []
        self.sentence_list_tokenized = []
        self.sentence_embedding = np.array([])
        self.flann = FLANN()

    def _average_bow(self, sentence):
        vs = np.zeros(self.embedding_dim)
        sentence_length = 0

        for word in sentence:
            try:
                vs = np.add(vs, self.word2vec[word])
                sentence_length += 1
            except Exception:
                pass
                # print(f"Embedding Vector: {word} not found")

        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)

        return vs

    def fit(self, sentence_list):
        for sentence in sentence_list:
            self.sentence_list.append(sentence)
            self.sentence_list_tokenized.append(self.tokenize(sentence))

        # Alg.1 step 1
        sentence_vec = []
        for sentence in self.sentence_list_tokenized:
            sentence_vec.append(self._average_bow(sentence))

        self.sentence_embedding = np.array(sentence_vec)

        # make index for similarity search
        self.flann.build_index(self.sentence_embedding)

    def infer_vector(self, sentence):
        return self._average_bow(self.tokenize(sentence))

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


def predict_labels(text, sif, docs, labels, n=1):
    most_similar_docs = sif.predict(text, n)
    most_similar_labels = []
    for m in most_similar_docs:
        most_similar_labels.append(labels[docs.index(m[0])])
    return max(most_similar_labels, key=most_similar_labels.count)


docs, labels = load_ld_corpus()
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.1, random_state=42)

# X_train_tokenized = [tokenize(x) for x in X_train]
# model = Word2Vec(size=200, min_count=10)
# model.build_vocab(X_train_tokenized)
# model.train(X_train_tokenized, total_examples=model.corpus_count, epochs=10)
# model.wv.save_word2vec_format("model/ld_word2vec_200.model", binary=True)

w2v = W2VAverageEmbedding("model/ld_word2vec_200.model")
w2v.fit(X_train)

y_predict = [predict_labels(x, w2v, X_train, y_train, 5) for x in X_test]

print(classification_report(y_test,
                            y_predict,
                            target_names=topics.keys()))
print(confusion_matrix(y_test, y_predict))
