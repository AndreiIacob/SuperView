from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pickle
import detectNLP
import data_helpers
import gensim
from GloVe_tensorflow import *
from os.path import exists, join

#can be "GloVe", "pretrainedWord2vec" or "category_trainedWord2vec"
wordvecType = "category_trainedWord2vec"

#can be "KMeans" or "DBSCAN"
clusteringType = "DBSCAN"

X = []
words = []

category = "Food"
with open("data/Las_Vegas-" + category + ".p", "rb") as f:
    reviews, business_id2business, user_id2user = pickle.load(f)
del user_id2user
del business_id2business

word2count = dict()
corpus = []
for review in reviews:
    corpus.append(review["words"])
    for word in set(review["words"]):
        if word not in word2count:
            word2count[word] = 0
        word2count[word] += 1

del reviews

if wordvecType == "category_trainedWord2vec":
    model = gensim.models.word2vec.Word2Vec(corpus,size=300, iter=100)
    word2vec = model.wv
elif wordvecType == "GloVe":
    if exists("data/GloVe/" + category + ".bin"):
        with open("data/GloVe/" + category + ".bin", 'rb') as vector_f:
            word2vec = pickle.load(f)
    else:
        model = GloVeModel(embedding_size=300, context_size=10)
        model.fit_to_corpus(corpus)
        print("Training")
        model.train(num_epochs=100)
        word2vec = dict()

        for text in corpus:
            for word in text:
                if word not in word2vec:
                    try:
                        word2vec[word] = model.embedding_for(word)
                    except Exception:
                        continue

        with open("data/GloVe/" + category + ".bin", 'wb+') as vector_f:
            pickle.dump(word2vec, vector_f)

elif wordvecType == "pretrainedWord2vec":
    word2vec = dict()
    for word in word2count:
        word2vec[word] = detectNLP.get_average_wordvec(word)
else:
    raise Exception("Invalid wordvecType")

cnt = 0
for word in word2count:
    try:
        wordVec = word2vec[word]
        if not np.isnan(wordVec).any():
            X.append(wordVec)
            words.append(word)
    except Exception:
        continue

if clusteringType=="KMeans":
    classifier = KMeans(n_clusters=10)
elif clusteringType=="DBSCAN":
    classifier = DBSCAN(min_samples=100)
else:
    raise Exception("Invalid clusteringType")

predictions = classifier.fit_predict(X)
word2cluster = dict()

rows = [["Word","Cluster"]]
for i, word in enumerate(words):
    clusterNum = predictions[i]
    word2cluster[word] = clusterNum
    rows.append([word, clusterNum])

filename = "data/" + wordvecType + "-" + clusteringType + "-" + category+"_word2cluster.csv"
data_helpers.write_csv(filename, rows)
data_helpers.write_csv_to_xlsx(filename)
