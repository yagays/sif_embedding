from collections import Counter

import MeCab
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from ld_corpus import load_ld_corpus
from ld_corpus import topics


def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split(" ")


def predict_labels(text, model):
    infer_vec = model.infer_vector(tokenize(text))
    result = model.docvecs.most_similar([infer_vec])
    return result[0][0]


docs, labels = load_ld_corpus()
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.2, random_state=42)

documents = [TaggedDocument(tokenize(x), tags=[y]) for x, y in zip(X_train, y_train)]
d2v = Doc2Vec(documents=documents, min_count=10, vector_size=200, workers=4, epochs=10)

y_predict = [predict_labels(x, d2v) for x in X_test]

# Evaluation
print(classification_report(y_test,
                            y_predict,
                            target_names=topics.keys()))
print(confusion_matrix(y_test, y_predict))
