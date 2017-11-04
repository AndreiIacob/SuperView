import pandas as pd
import ijson
import json, re, csv
from nltk.corpus import wordnet as wn
from googletrans import Translator
import string
import unicodedata
import nltk.stem.snowball as snowball
import nltk
import numpy as np
import tensorflow as tf
import pickle
from xlsxwriter.workbook import Workbook
from nltk.probability import ELEProbDist
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from helper_functions import *


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def train(labeled_featuresets, label_freqdist, estimator=ELEProbDist):
    ...
    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)
    ...
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    ...
    return NaiveBayesClassifier(label_probdist, feature_probdist)

def word_feats(words):
    return dict([(word, True) for word in words])

def classify_using_NaiveBayes(reviews):
    negative_reviews = []
    positive_reviews = []
    for review in reviews:
        if review['stars'] >3:
            positive_reviews.append(process_words(review['text'], language="en"))
        elif review['stars'] <3:
            negative_reviews.append(process_words(review['text'], language="en"))

    negfeats = [(word_feats(negative_review), 'neg') for negative_review in negative_reviews]
    posfeats = [(word_feats(positive_review), 'pos') for positive_review in positive_reviews]

    trainfeats = negfeats + posfeats
    testfeats = negfeats + posfeats
    print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

    classifier = NaiveBayesClassifier.train(trainfeats)
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features()

#business dict_keys(['business_id', 'name', 'neighborhood', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'])
#review dict_keys(['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool'])
#tip dict_keys(['text', 'date', 'likes', 'business_id', 'user_id'])
#photos dict_keys(['photo_id', 'business_id', 'caption', 'label'])
#checkin dict_keys(['time', 'business_id'])
#user dict_keys(['user_id', 'name', 'review_count', 'yelping_since', 'friends', 'useful', 'funny', 'cool', 'fans', 'elite', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'])

city2category2count    = dict()
city2business_count    = dict()
business2review_count  = dict()
business_id2categories = dict()
category2reviews       = dict()
business2reviews_count = dict()
category2reviews_count = dict()
business_id2name       = dict()
business_id2city       = dict()
category2business2reviews = dict()

reviews = []

with open(r"dataset\business.json", 'r', encoding='utf-8') as f:
    for line in f:
        business = json.loads(line)
        city = business["city"]
        if city not in city2category2count:
            city2category2count[city] = dict()
            city2business_count[city] = 0
        city2business_count[city] += 1
        categories  = business["categories"]
        business_id = business["business_id"]
        name = business["name"]
        business_id2name[business_id] = name
        business_id2city[business_id] = city
        business_id2categories[business_id] = categories
        business2reviews_count[business_id] = [0] * 5
        for category in categories:
            if category not in category2business2reviews:
                category2business2reviews[category] = dict()
            category2business2reviews[category][business_id] = []

business2word_count = dict()

texts = []
reviews = []
with open(r"dataset\review.json", 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line)
        texts.append(review["text"])
        reviews.append(review)
        business_id = review["business_id"]
        stars = review["stars"]
        business2reviews_count[business_id][stars - 1] += 1
        for category in business_id2categories[business_id]:
            category2business2reviews[category][business_id].append(review)


category = 'Greek'

print(category + "\n")

word2rating = dict()
for business in category2business2reviews[category]:
    business_word2rating = dict()
    for review in category2business2reviews[category][business]:
        words = set(process_words(review["text"], "en"))
        stars = review["stars"]
        for word in words:
            if word not in business_word2rating:
                business_word2rating[word] = [0.0, 0]
                if word not in word2rating:
                    word2rating[word] = [0.0, 0]
            business_word2rating[word][0] += stars
            business_word2rating[word][1] += 1
            word2rating[word][0] += stars
            word2rating[word][1] += 1

    word_score_instances = []
    for word in business_word2rating:
        business_word2rating[word][0] = business_word2rating[word][0]/business_word2rating[word][1]
        if business_word2rating[word][1] > 10:
            word_score_instances.append([word] + business_word2rating[word])

    print("\n" + business_id2name[business] + "\n")

    word_score_instances.sort(key = lambda x: abs(x[1]-3), reverse=True)
    for entry in word_score_instances[:100]:
        print(entry[0] + " - rating:" + str(entry[1]) + "  instances:" + str(entry[2]))
    word_score_instances.sort(key = lambda x: x[2], reverse=True)
    for entry in word_score_instances[:100]:
        print(entry[0] + " - rating:" + str(entry[1]) + "  instances:" + str(entry[2]))

word_score_instances = []
for word in word2rating:
    word2rating[word][0] = word2rating[word][0]/word2rating[word][1]
    if word2rating[word][1] > 50:
        word_score_instances.append([word] + word2rating[word])

print("\n" + category + "\n")
word_score_instances.sort(key = lambda x: abs(x[1]-3), reverse=True)
for entry in word_score_instances[:100]:
    print(entry[0] + " - rating:" + str(entry[1]) + "  instances:" + str(entry[2]))
word_score_instances.sort(key = lambda x: x[2], reverse=True)
for entry in word_score_instances[:100]:
    print(entry[0] + " - rating:" + str(entry[1]) + "  instances:" + str(entry[2]))

print("\n\n")

category_reviews = []
for business in category2business2reviews[category]:
    business_word2rating = dict()
    category_reviews += category2business2reviews[category][business]

    print("\n" + business_id2name[business] + "\n")
    classify_using_NaiveBayes(category2business2reviews[category][business])

classify_using_NaiveBayes(category_reviews)