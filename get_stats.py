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
from helper_functions import *

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
word2rating            = dict()
category2reviews       = dict()
business2reviews_count = dict()
category2reviews_count = dict()
business_id2name       = dict()
business_id2city       = dict()

reviews = []
avg_reviews_for_users = [0.0, 0]
with open(r"dataset\user.json", 'r', encoding='utf-8') as f:
    for line in f:
        j = json.loads(line)
        avg_reviews_for_users[0] += j["review_count"]
        avg_reviews_for_users[1] += 1

print("Average review count for users:",round(avg_reviews_for_users[0]/avg_reviews_for_users[1],5))

with open(r"dataset\business.json", 'r', encoding='utf-8') as f:
    for line in f:
        j = json.loads(line)
        reviews.append(j)
        city = j["city"]
        if city not in city2category2count:
            city2category2count[city] = dict()
            city2business_count[city] = 0
        city2business_count[city] += 1
        categories  = j["categories"]
        business_id = j["business_id"]
        name = j["name"]
        business_id2name[business_id] = name
        business_id2city[business_id] = city
        business_id2categories[business_id] = categories
        business2reviews_count[business_id] = [0] * 5
        for category in categories:
            if category not in category2reviews_count:
                category2reviews_count[category] = [0] * 5

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
            if category not in category2reviews:
                category2reviews[category] = []
            category2reviews[category].append(review)
            category2reviews_count[category][stars - 1] += 1


common_header = []
for i in range(1,6):
    common_header.append(str(i) + " Star Reviews")

common_header.append("Positive Reviews")
common_header.append("Negative Reviews")
common_header.append("All Reviews Count")
common_header.append("Average Rating")

city2reviews_count = dict()
city2businesses    = dict()
for business in business2reviews_count:
    city = business_id2city[business]
    if city not in city2reviews_count:
        city2reviews_count[city] = [0] * 5
        city2businesses[city]    = 0
    city2businesses[city]    += 1
    for k in range(5):
        city2reviews_count[city][k] += business2reviews_count[business][k]

category_rows    = [["Category"] + common_header]
business_id_rows = [["Business Name","Business ID","Number of Categories","Categories","City"] + common_header]
city_rows        = [["City","Businesses" ] + common_header]

for category in category2reviews_count:
    star_counts = category2reviews_count[category]
    row = [category]
    score_sum = 0.0
    for i in range(5):
        score_sum += star_counts[i] * (i+1)
        row.append(star_counts[i])
    row.append(star_counts[3] + star_counts[4])
    row.append(star_counts[0] + star_counts[1])
    number_of_reviews = sum(star_counts)
    row.append(number_of_reviews)
    row.append(score_sum/number_of_reviews)
    category_rows.append(row)

write_csv_and_xlsx("stats/category_stats", category_rows)

for business in business2reviews_count:
    city = business_id2city[business]
    business_name = business_id2name[business]
    categories = business_id2categories[business]
    star_counts = business2reviews_count[business]
    row = [business, business_name, len(categories), categories, city]
    score_sum = 0.0
    for i in range(5):
        score_sum += star_counts[i] * (i+1)
        row.append(star_counts[i])
    row.append(star_counts[3] + star_counts[4])
    row.append(star_counts[0] + star_counts[1])
    number_of_reviews = sum(star_counts)
    row.append(number_of_reviews)
    if number_of_reviews>0:
        row.append(score_sum/number_of_reviews)
    else:
        row.append(0)
    business_id_rows.append(row)

write_csv_and_xlsx("stats/business_stats", business_id_rows)

for city in city2reviews_count:
    star_counts = city2reviews_count[city]
    row = [city, city2businesses[city]]
    score_sum = 0.0
    for i in range(5):
        score_sum += star_counts[i] * (i+1)
        row.append(star_counts[i])
    row.append(star_counts[3] + star_counts[4])
    row.append(star_counts[0] + star_counts[1])
    number_of_reviews = sum(star_counts)
    row.append(number_of_reviews)
    if number_of_reviews>0:
        row.append(score_sum/number_of_reviews)
    else:
        row.append(0)
    city_rows.append(row)

write_csv_and_xlsx("stats/city_stats", city_rows)

import math

split_texts = []
number_of_splits = 100
numuber_of_reviews_per_split = math.ceil(float(len(texts))/number_of_splits)
for i in range(number_of_splits):
    text_chunk = texts[numuber_of_reviews_per_split*i:numuber_of_reviews_per_split*(i+1)]
    text_chunk = process_words(text_chunk, "en", stem=True)
    texts = texts[:numuber_of_reviews_per_split*i] + text_chunk + texts[numuber_of_reviews_per_split*(i+1):]

i = 0
while i < len(texts):
    if texts[i] == []:
        del texts[i]
        del reviews[i]
    else:
        i += 1

for text in texts:
    for word in text:
        if word not in word2rating:
            word2rating[word] = [0.0, 0]
        word2rating[word][1] += 1

frequncy_boundries = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
frequency_cnts = [0] * len(frequncy_boundries)

for text in texts:
    for word in text:
        for k, frequncy_boundry in enumerate(frequncy_boundries):
            if word2rating[word][1] >= frequncy_boundry:
                frequency_cnts[k] += 1

for k, frequncy_boundry in enumerate(frequncy_boundries):
    print("Number of words that appear a minimum of", frequncy_boundry,":",frequency_cnts[k])

MIN_DOCUMENT_LENGTH = 10000000
MAX_DOCUMENT_LENGTH = 0
word_total = 0.0
for text in texts:
    length = len(text)
    if length > MAX_DOCUMENT_LENGTH:
        MAX_DOCUMENT_LENGTH = length
    if length < MIN_DOCUMENT_LENGTH:
        MIN_DOCUMENT_LENGTH = length
    word_total += length
print("Average number of words per review:",word_total/len(texts))
print("Min number of words:", MIN_DOCUMENT_LENGTH)
print("Max number of words:", MAX_DOCUMENT_LENGTH)

