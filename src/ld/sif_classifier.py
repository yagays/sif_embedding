from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec

from src.sif import SifEmbedding
from src.sif import tokenize
from ld_corpus import load_ld_corpus
from ld_corpus import topics


def predict_labels(text, sif, docs, labels, n=1):
    most_similar_docs = sif.predict(text, n)
    most_similar_labels = []
    for m in most_similar_docs:
        most_similar_labels.append(labels[docs.index(m[0])])
    return max(most_similar_labels, key=most_similar_labels.count)


docs, labels = load_ld_corpus()
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.1, random_state=42)

X_train_tokenized = [tokenize(x) for x in X_train]
model = Word2Vec(size=200, min_count=10)
model.build_vocab(X_train_tokenized)
model.train(X_train_tokenized, total_examples=model.corpus_count, epochs=10)
model.wv.save_word2vec_format("model/ld_word2vec_200.model", binary=True)

sif = SifEmbedding("model/ld_word2vec_200.model")
sif.fit(X_train)

y_predict = [predict_labels(x, sif, X_train, y_train, 5) for x in X_test]

print(classification_report(y_test,
                            y_predict,
                            target_names=topics.keys()))
print(confusion_matrix(y_test, y_predict))
