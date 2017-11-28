from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pycorenlp import StanfordCoreNLP
from helper_functions import *
import pickle
import logging

from time import time
from multiprocessing.dummy import Pool as ThreadPool
from AOP import *

import unittest

'''
install:
    pip install wget
    pip install pycorenlp
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    unzip stanford-corenlp-full-2016-10-31.zip
    source:    http://stackoverflow.com/questions/32879532/stanford-nlp-for-python

start server cmd:

    cd C:\Windows\System32\corenlp-python\stanford-corenlp-full-2016-10-31
    java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001
'''


core_nlp = StanfordCoreNLP('http://localhost:9001')
def get_sentiment_coreNLP(text, print_result=False, core_nlp= core_nlp):
    res = core_nlp.annotate(text,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 100,
                       })

    sentiments = []
    for s in res["sentences"]:
        score = float(s["sentimentValue"])/4
        sentiments.append(score)
    if len(sentiments) == 1:
        return sentiments[0]
    if print_result:
        for s in res["sentences"]:
            print ("%d: '%s': %s %s" % (
                s["index"],
                " ".join([t["word"] for t in s["tokens"]]),
                s["sentimentValue"], s["sentiment"]))
    return sentiments

def get_sentence_sentiment_coreNLP(text, core_nlp= core_nlp):
    res = core_nlp.annotate(text,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 10000,
                       })

    sentence_sentiments = []
    print(res["sentences"])
    try:
        for s in res["sentences"]:
            sentence = " ".join([t["word"] for t in s["tokens"]])
            sentence_sentiments.append([sentence, int(s["sentimentValue"])])
    except Exception:
        logging.warn('failed to get sentiment value for text: %s' % text)
        pass
    return sentence_sentiments

def get_average_sentiment_coreNLP(text, core_nlp= core_nlp):
    try:
        res = core_nlp.annotate(text,
                           properties={
                               'annotators': 'tokenize,ssplit,sentiment',
                               'outputFormat': 'json',
                               'timeout': 100000,
                           })
        sentiments = []

        for s in res["sentences"]:
            sentence = []
            for token in s["tokens"]:
                sentence.append(token['word'])
            score = float(s["sentimentValue"])/4
            sentiments.append(score)
        return sum(sentiments) / len(sentiments)
    except Exception:
        return 0.5

def get_senitment_textblob(text):
    testimonial = TextBlob(text)
    return (testimonial.sentiment.polarity+1) / 2

def get_sentiment_vader(text):
    vaderAnalyzer = SentimentIntensityAnalyzer()
    scores = vaderAnalyzer.polarity_scores(text)
    #dictkeys: neg(negative),  neu(neutral) and pos(positive)
    return scores["neu"] * 0.5 + scores["pos"]

def get_dict_acc( dictionary):
    overall = [0.0, 0]
    for key in dictionary:
        overall[0] += dictionary[key][0]
        overall[1] += dictionary[key][1]
    return round(overall[0]/overall[1],4)

@timing
def compare_sentiment_analysis(reviews, business_id2business, output_file = "data/compare_sentiment_analysis.txt"):
    textblob2acc = dict()
    vader2acc = dict()
    corenlp2acc = dict()
    textblob2error = dict()
    vader2error = dict()
    corenlp2error = dict()
    reviews.reverse()
    time1 = time()

    for review in reviews:
        if review["business_id"] not in textblob2acc:
            textblob2acc[review["business_id"]] = [0.0, 0]
            vader2acc[review["business_id"]]    = [0.0, 0]
            corenlp2acc[review["business_id"]]  = [0.0, 0]
            textblob2error[review["business_id"]] = [0.0, 0]
            vader2error[review["business_id"]]    = [0.0, 0]
            corenlp2error[review["business_id"]]  = [0.0, 0]

        textblob2acc[review["business_id"]][1] += 1
        vader2acc[review["business_id"]][1] += 1
        corenlp2acc[review["business_id"]][1] += 1
        textblob2error[review["business_id"]][1] += 1
        vader2error[review["business_id"]][1] += 1
        corenlp2error[review["business_id"]][1] += 1

        score = (float(review["stars"]) - 1.0) / 4
        text = review["text"]
        textblob_score = get_senitment_textblob(text)
        if abs(score - textblob_score) <= 0.125:
            textblob2acc[review["business_id"]][0] += 1

        textblob2error[review["business_id"]][0] += abs(score - textblob_score)

        vader_score = get_sentiment_vader(text)
        if abs(score - vader_score) <= 0.125:
            vader2acc[review["business_id"]][0] += 1

        vader2error[review["business_id"]][0] += abs(score - vader_score)

        core_nlp_score = get_average_sentiment_coreNLP(text)
        if abs(score - core_nlp_score) <= 0.125:
            corenlp2acc[review["business_id"]][0] += 1

        corenlp2error[review["business_id"]][0] += abs(score - core_nlp_score)

        if (time() - time1) > 3600:
            break


    with open(output_file, "a") as f:
        accs = [textblob2acc, vader2acc, corenlp2acc,textblob2error,vader2error,corenlp2error]
        print_str = "Business Name\tTextblob Accuracy\tVader Accuracy\tCoreNLP Accuracy\tTextblob Error\tVader Error\tCoreNLP Error"
        print(print_str)
        f.write(print_str + "\n")
        for business_id in textblob2acc:
            if business_id not in business_id2business:
                continue
            print_str = business_id2business[business_id]["name"]
            for acc in accs:
                print_str+= "\t" + str(round(acc[business_id][0]/acc[business_id][1],4))
            print(print_str)
            f.write(print_str + "\n")
        print_str = "Overall"
        for acc in accs:
            print_str += "\t" + str(get_dict_acc(acc))
        print(print_str)
        f.write(print_str + "\n")

def add_sentences_to_reviews(reviews, port = 9000):
    core_nlp = StanfordCoreNLP('http://localhost:' + str(reviews[1]))
    try:
        for review in reviews[0]:
            if "sentences" not in review:
                review["sentences"] = get_sentence_sentiment_coreNLP(review["text"], core_nlp = core_nlp)
            elif review["sentences"] is []:
                review["sentences"] = get_sentence_sentiment_coreNLP(review["text"], core_nlp = core_nlp)
    except Exception:
        pass
    return reviews

@timing
def add_sentences_to_reviews_multithreaded(reviews,numberOfThreads = 4):
    step = len(reviews) // numberOfThreads
    split_reviews = [(reviews[step * i:step * (i + 1)], i + 9001) for i in range(numberOfThreads)]
    pool = ThreadPool(numberOfThreads)
    results = pool.map(add_sentences_to_reviews, split_reviews)
    return results



def coreNLP_sentiment_by_word(text, average=False):
    words = process_words(text, "en", stem=True)
    ratings = [0] * 5
    for word in words:
        try:
            res = core_nlp.annotate(word,
                                    properties={
                                        'annotators': 'sentiment',
                                        'outputFormat': 'json',
                                        'timeout': 100,
                                    })
            for s in res["sentences"]:
                ratings[int(s["sentimentValue"])] += 1
        except Exception:
            pass

    if average:
        total = np.sum(ratings)
        if total == 0:
            return 2
        rating_average = 0.0
        for i in range(len(ratings)):
            rating_average += i * ratings[i]
        return rating_average/total
    else:
        return np.argmax(ratings)

class TddSentimentAnalysis(unittest.TestCase):
    def test_positive_sentences_classified_incorectly(self):
        sentences = [('''My favorite part about this property is the casino ,
     especially the poker room ,
     which is located right in front of the hotel ,
     and it 's big enough to accommodate enough people than almost all of the other poker rooms on the strip .''',3),
                     ("I heart this poker room and the casino here is HUGE !",4),
                     ("I stayed at the Aria after getting a great deal for a 2 night stay .",3),
                     ('''Our room looked out on the Vdara and not the strip ,
     but still provided a great view .''',3),
                     ("bathrooms - awesome .",4)]

        errors = [0.0] * 3
        for sentence in sentences:
            text = sentence[0]
            rating = sentence[1]
            textblob_score = get_senitment_textblob(text) * 4
            vader_score    = get_sentiment_vader(text) * 4
            word_coreNLP   = coreNLP_sentiment_by_word(text,average=True)
            errors[0] += abs(rating-textblob_score)
            errors[1] += abs(rating - vader_score)
            errors[2] += abs(rating - word_coreNLP)

        self.assertLess(np.min(errors),len(sentences)*0.7)

    def test_sentences_classified_corectly(self):
        sentences = [("pool area - meh .",2),
                     ("4 stars because the evening front desk was n't as helpful and the morning shift .",1),
                     ("Look maybe these things are small , but to have to stop and call the front the desk is annoying .",1),
                     ("Overall experience was very good .",4),
                     ("Aria has a tram that will take you over to the Bellagio .",3)]

        errors = [0.0] * 3
        for sentence in sentences:
            text = sentence[0]
            rating = sentence[1]
            textblob_score = get_senitment_textblob(text) * 4
            vader_score    = get_sentiment_vader(text) * 4
            word_coreNLP   = coreNLP_sentiment_by_word(text,average=True)
            errors[0] += abs(rating-textblob_score)
            errors[1] += abs(rating - vader_score)
            errors[2] += abs(rating - word_coreNLP)
        self.assertLess(np.min(errors), len(sentences) * 0.7)


if __name__ == '__main__':
    categories = ["Hotels"]
    for category in categories:
        print(category)
        with open("data/Las_Vegas-" + category + ".p", "rb") as f:
            all_reviews, business_id2business, user_id2user = pickle.load(f)
        compare_sentiment_analysis(all_reviews, business_id2business,"data/"+category+"_compare_sentiment_analysis.txt")
    unittest.main()




