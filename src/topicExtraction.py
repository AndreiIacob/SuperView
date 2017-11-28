from data_helpers import *
from SentimentAnalysis import *
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_topics(business_id , n_features = 1000, n_components = 10, n_top_words = 20):
    try:
        reviews, selected_business_name = get_business_data(business_id)
    except Exception:
        reviews, selected_business_name = get_business_data_local( business_id)

    stars2texts = [[]] * 5
    for review in reviews:
        if "sentences" not in review:
            review["sentences"] = get_sentence_sentiment_coreNLP(review["text"], core_nlp=core_nlp)
        if len(review["sentences"]) == 0:
            continue
        star = float(review["stars"]) - 1
        average_sentiment = 0.0
        for sentence_sentiment in review["sentences"]:
            sentence = sentence_sentiment[0]
            sentiment = sentence_sentiment[1]
            average_sentiment += sentiment
        average_sentiment /= len(review["sentences"])
        average_sentiment += 1
        for sentence_sentiment in review["sentences"]:
            sentence = sentence_sentiment[0]
            sentiment = sentence_sentiment[1] + 1

            sentence_star = int(round((float(sentiment) / average_sentiment) * star))

            if sentence_star > 4:
                sentence_star = 4

            stars2texts[sentence_star].append(sentence)

    topics = dict()
    print()
    print(selected_business_name)
    for j in range(4):
        if j == 0:
            data_samples = stars2texts[0]
            prefix = "Very negative"
        elif j == 1:
            data_samples = stars2texts[0] + stars2texts[1]
            prefix = "Negative"
        elif j == 2:
            data_samples = stars2texts[3] + stars2texts[4]
            prefix = "positive"
        elif j == 3:
            data_samples = stars2texts[4]
            prefix = "very positive"

        topics[prefix] = dict()
        n_samples = len(data_samples)

        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(data_samples)
        print("done in %0.3fs." % (time() - t0))

        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english')
        t0 = time()
        tf = tf_vectorizer.fit_transform(data_samples)
        print("done in %0.3fs." % (time() - t0))
        print()

        # Fit the NMF model
        print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        t0 = time()
        nmf = NMF(n_components=n_components, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\n" + prefix + " " + selected_business_name + " Topics in NMF model (Frobenius norm):")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        topics[prefix]["FroNMF"] = []
        for topic_idx, topic in enumerate(nmf.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([tfidf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics[prefix]["FroNMF"].append([tfidf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

        # Fit the NMF model
        print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
              "tf-idf features, n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        t0 = time()
        nmf = NMF(n_components=n_components, random_state=1,
                  beta_loss="kullback-leibler", solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        print(
            "\n" + prefix + " " + selected_business_name + " Topics in NMF model (generalized Kullback-Leibler divergence):")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        topics[prefix]["KLNMF"] = []
        for topic_idx, topic in enumerate(nmf.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([tfidf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics[prefix]["KLNMF"].append([tfidf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        print("\n" + prefix + " " + selected_business_name + " Topics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()

        topics[prefix]["tfLDA"] = []
        for topic_idx, topic in enumerate(lda.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([tf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics[prefix]["tfLDA"].append([tf_feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
    return topics

def get_business_data_local(business_id):
    categories = ["Hotels","Food"]
    for category in categories:
        with open("data/Las_Vegas-" + category + ".p", "rb") as f:
            all_reviews, business_id2business, user_id2user = pickle.load(f)
        if business_id in business_id2business:
            break

    reviews = []
    business_name = business_id2business[business_id]["name"]
    for review in all_reviews:
        if review["business_id"] == business_id:
            reviews.append(review)

    return reviews, business_name

def get_business_data(business_id):
    with open("database/config.json", "r") as f:
        config = json.load(f)

    client = MongoClient('mongodb://%s:%s@' % (config["username"], config["password"]) + config["url"] + '')
    db = client[config["db_name"]]
    collection = db['reviews']
    reviews = collection.find({"business_id": business_id})
    collection = db['businesses']
    business = collection.find_one({'_id': business_id})
    business_name = business["name"]
    return reviews, business_name
