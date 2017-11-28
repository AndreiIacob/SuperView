import numpy as np
import re
import itertools
import csv
import nltk.stem.snowball as snowball
import nltk
from nltk.corpus import wordnet as wn
from googletrans import Translator
import operator
import tensorflow as tf
import string
import heapq
import codecs
import collections
import unicodedata
import glob
import pickle
import matplotlib.pyplot as plt
from xlsxwriter.workbook import Workbook

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(), !?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r", ", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class batcher:
    def __init__(self, x, y):
        self.data = list(zip(x, y))
        np.random.shuffle(self.data)
        self.position = 0

    def next_batch(self, batch_size):
        aux = self.position+batch_size
        if aux > len(self.data):
            np.random.shuffle(self.data)
            self.position = 0
            aux = batch_size
        batch = self.data[self.position: aux]
        self.position+=batch_size
        batch_x, batch_y = zip(*batch)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

class one_hot_batcher:
    def __init__(self, x, y, x_depth, y_depth):
        self.data = list(zip(x, y))
        self.x_depth = x_depth
        self.y_depth = y_depth
        np.random.shuffle(self.data)
        self.position = 0

    def next_batch(self, batch_size):
        aux = self.position+batch_size
        if aux > len(self.data):
            np.random.shuffle(self.data)
            self.position = 0
            aux = batch_size
        batch = self.data[self.position: aux]
        self.position+=batch_size
        x, y = zip(*batch)
        return one_hot(x, self.x_depth), one_hot(y, self.y_depth)

class embeding_batcher:
    def __init__(self, x, y, word2vec, size, y_depth, word2vec_size=300):
        self.data = list(zip(x, y))
        self.word2vec = word2vec
        self.size = size
        self.y_depth = y_depth
        np.random.shuffle(self.data)
        self.position = 0
        self.word2vec_size = word2vec_size

    def next_batch(self, batch_size):
        aux = self.position+batch_size
        if aux > len(self.data):
            np.random.shuffle(self.data)
            self.position = 0
            aux = batch_size
        batch = self.data[self.position: aux]
        self.position+=batch_size
        x, y = zip(*batch)
        return text2vec(x, self.word2vec, self.size, self.word2vec_size), y

class equal_batcher:
    def __init__(self, x, y, embedding = None, one_hot_depth_x = None, one_hot_depth_y = None, word2vec = None, size = None, word2vec_dimentions = None):
        if isinstance(embedding, str):
            embedding = re.sub("_", " ", embedding)
        self.embedding = embedding
        if embedding == "one hot":
            self.one_hot_depth_x = one_hot_depth_x
            self.one_hot_depth_y = one_hot_depth_y
        elif embedding == "word2vec":
            self.word2vec = word2vec
            self.size = size
            self.word2vec_dimentions = word2vec_dimentions

        if isinstance(y[0], collections.Hashable):
            self.uses_dict = True
            self.data = dict()
            self.position = dict()
            for i in range(len(x)):
                x_val = x[i]
                y_val = y[i]
                if y_val not in self.data:
                    self.data[y_val] = []
                    self.position[y_val] = 0
                self.data[y_val].append(x_val)
            self.y = []
            for y_val in self.data:
                np.random.shuffle(self.data[y_val])
                self.y.append(y_val)
        else:
            self.uses_dict = False
            self.data = [[x[0]]]
            self.position = [0]
            self.y = [y[0]]

            for i in range(1, len(x)):
                x_val = x[i]
                y_val = y[i]

                if not (np.any([(y_val == x).all() for x in self.y])) :
                    j = len(self.y)
                    self.y.append(y_val)
                    self.position.append(0)
                    self.data.append([])
                else:
                    j = [np.array_equal(y_val, x) for x in self.y].index(True)

                self.data[j].append(x_val)
            for vals in self.data:
                np.random.shuffle(vals)


    def next_batch(self, batch_size):
        get_from_each = batch_size//len(self.data)

        batch_y = []
        batch_x = []

        if self.uses_dict:
            for y_val in self.data:
                batch_y += [y_val]*get_from_each
                aux = self.position[y_val] + get_from_each
                if aux>len(self.data[y_val]):
                    np.random.shuffle(self.data[y_val])
                    self.position[y_val] = 0
                    aux = get_from_each
                batch_x += self.data[y_val][self.position[y_val]:aux]
                self.position[y_val] = aux
                if self.position[y_val] >= len(self.data[y_val]):
                    self.position[y_val] = 0

                if len(batch_x) == 89:
                    print(y_val)

            leftover = batch_size - len(batch_x)

            for rand_num in np.random.choice(len(self.data), leftover, replace=False):
                y_val = self.y[rand_num]
                batch_y.append(y_val)
                batch_x.append(self.data[y_val][self.position[y_val]])
                self.position[y_val] +=1
                if self.position[y_val]>=len(self.data[y_val]):
                    self.position[y_val] = 0
        else:
            for i in range(len(self.data)):
                batch_y += [self.y[i]] * get_from_each
                aux = self.position[i] + get_from_each
                if aux > len(self.data[i]):
                    np.random.shuffle(self.data[i])
                    self.position[i] = 0
                    aux = get_from_each
                batch_x += self.data[i][self.position[i]:aux]
                self.position[i] = aux
                if self.position[i] >= len(self.data[i]):
                    self.position[i] = 0

            leftover = batch_size - len(batch_x)

            for rand_num in np.random.choice(len(self.data), leftover, replace=False):
                y_val = self.y[rand_num]
                batch_y.append(y_val)
                batch_x.append(self.data[rand_num][self.position[rand_num]])
                self.position[rand_num] += 1
                if self.position[rand_num] >= len(self.data[rand_num]):
                    self.position[rand_num] = 0

        if self.embedding is 'one hot':
            if self.one_hot_depth_x is not None:
                batch_x = one_hot(batch_x, self.one_hot_depth_x)
            if self.one_hot_depth_y is not None:
                batch_y = one_hot(batch_x, self.one_hot_depth_y)
        elif self.embedding == "word2vec":
            batch_x = text2vec(batch_x, self.word2vec, self.size, self.word2vec_dimentions)

        return np.array(batch_x), np.array(batch_y)

def fillter_low_count(texts, labels, texts2=None, labels2=None, min_frequency = 100):
    count = dict()
    for label in labels:
        if label not in count:
            count[label] = 0
        count[label]+=1

    if labels2 is not None:
        for label in labels2:
            if label not in count:
                count[label] = 0
            count[label] += 1

    check_whole_set = False
    for label in labels:
        if count[label]<min_frequency:
            check_whole_set = True
            break

    if check_whole_set:
        i = 0
        while i<len(texts):
            if count[labels[i]]<min_frequency:
                del texts[i]
                del labels[i]
            else:
                i+=1

        if labels2 is not None and texts2 is not None:
            i = 0
            while i < len(texts2):
                if count[labels2[i]] < min_frequency:
                    del texts2[i]
                    del labels2[i]
                else:
                    i += 1

    if labels2 is not None:
        return texts, labels, texts2, labels2

    return texts, labels


def text2vec(text, word2vec, size, word2vec_size=300, process = False, language=None):
    nums = []
    if process:
        text = process_words(text, language)

    if not isinstance(text[0], list):
        for word in text:
            if word in word2vec:
                nums.append(word2vec[word])
            else:
                nums.append([-1000]*word2vec_size)
        return normalize_size(nums, size, word2vec_size=word2vec_size)
    else:
        err = 0
        for subtext in text:
            words = []
            unknown = 0
            for word in subtext:
                if word in word2vec:
                    words.append(word2vec[word])
                else:
                    words.append([-1000]*word2vec_size)
                    unknown += 1
            if (len(words) - unknown) == 0:
                print(subtext)
                err += 1
            nums.append(normalize_size(words, size, word2vec_size=word2vec_size))

        if err > 0:
            FAIL = '\033[91m'
            if err == 1:
                print(FAIL + "Warning: " + str(err) + " text contains no words from vocabulary" + bcolors.ENDC)
            else:
                print(FAIL + "Warning: " +str(err)+ " texts contain no words from vocabulary" + bcolors.ENDC)
    return nums


def convert_label_to_num(labels):
    label_to_num = {}
    num_to_label = {}
    nums = []
    index = 0
    for label in labels:
        label = label.strip()
        if label in label_to_num:
            nums.append(label_to_num[label])
        else:
            label_to_num[label]=index
            index+=1
            nums.append(label_to_num[label])
    num_to_label = dict(zip(label_to_num.values(), label_to_num.keys()))
    return np.array(nums), label_to_num, num_to_label

def write_csv_to_xlsx(csv_file):
        workbook = Workbook(csv_file[:-4] + '.xlsx')
        worksheet = workbook.add_worksheet()
        with open(csv_file, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet.write(r, c, col)
        workbook.close()


def normalize_size(v, size, word2vec_size=None):
    if word2vec_size is None or word2vec_size == 0:
        if len(v)<size:
            return np.concatenate((np.zeros(size-len(v)), v))
        elif len(v)>size:
            return v[:size]
        return v
    else:
        if len(v)<size:
            return np.vstack((np.zeros((size-len(v), word2vec_size)), v))
        elif len(v)>size:
            return v[:size]
        return v

def read_csv(file, encoding = 'utf-16' ):
    with codecs.open(file, 'r', encoding=encoding) as f:
       reader = csv.reader(f, delimiter=',', quotechar='"', dialect='excel')
       texts = []
       labels1 = []
       labels2 = []
       languages = []
       for row in reader:
           if row[2] != '' and row[3] != '' :
               texts.append(row[0])
               labels1.append(row[2])
               labels2.append(row[3])
               if len(row)>4:
                    languages.append(row[5])

    return texts, labels1, labels2, languages


def write_csv(filename, rows):
    with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', 
                               quotechar='"')
        for row in rows:
            csvwriter.writerow(row)

def to_embeding_format(texts, texts2, word_dict):

    result = []
    word2id = dict()
    for word in word_dict:
        size = len(word_dict[word])
        break

    id2number = np.zeros((len(word_dict)+1,size),dtype=np.float64)

    for word in word_dict:
        word2id[word] = len(word2id)+1
        id2number[len(word2id)] = word_dict[word]

    #transfrom words into ids
    for text in texts:
        sent = []
        for word in text:
            sent.append(word2id[word])
        result.append(sent)
    result = np.array(result)

    result2 = []
    for text in texts2:
        sent = []
        for word in text:
            sent.append(word2id[word])
            result2.append(sent)
    result2 = np.array(result2)

    id2word = zip(word2id.values(),word2id.keys())

    return result, result2, id2number, word2id, id2word


def build_dataset(texts, number_of_words=None, document_size=None, process=True, language=None, min_count=0 ):
    if process:
        words = process_words(texts, language)
    else:
        words = [nltk.word_tokenize(text) for text in texts]

    words_flat = [word for sublist in words for word in sublist]
    count = collections.Counter(words_flat).most_common()
    dictionary = dict()
    for word, cnt in count:
        if cnt >= min_count:
            if number_of_words is not None and len(dictionary) > number_of_words:
                dictionary[word] = number_of_words+1
            else:
                dictionary[word] = len(dictionary)+1
                if number_of_words is not None and len(dictionary) > number_of_words:
                    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    if number_of_words is None or number_of_words>=len(dictionary):
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    else:
        reverse_dictionary[number_of_words+1] = '_unknown'
    nums = []
    max_document_length = 0
    number_of_words = len(dictionary)+1
    for text in words:
        if len(text)>max_document_length:
            max_document_length = len(text)
        sent = []
        for word in text:
            if word in dictionary:
                sent.append(dictionary[word])
            else:
                sent.append(len(dictionary)+1)
        nums.append(sent)

    if document_size is None:
        document_size = max_document_length

    for i in range(len(nums)):
        nums[i] = normalize_size(nums[i], document_size)

    return np.array(nums, dtype=np.int32), dictionary, reverse_dictionary, number_of_words

def text2nums(text, word2num, document_size, unknown_token=None, language=None, process=True):
    if unknown_token is None:
        unknown_token = len(word2num)
    if process:
        words = process_words(text, language)
    else:
        words = nltk.word_tokenize(text)

    nums = []
    for word in words:
        if word in word2num:
            nums.append(word2num[word])
        else:
            nums.append(unknown_token)

    return normalize_size(nums, document_size)


def fillter_dict_frequency(label2_label1_text_dict, axis = 2, min_frequency = 100 ):
    if axis == 2:
        for label1 in label2_label1_text_dict.keys():
            label_dict = label2_label1_text_dict[label1]
            keys = list(label_dict.keys())
            for label2 in keys:
                if len(label_dict[label2])<min_frequency:
                    del label2_label1_text_dict[label1][label2]

    if axis == 1:
        keys = list(label2_label1_text_dict.keys())
        for label1 in keys:
            size = 0
            label_dict = label2_label1_text_dict[label1]

            for label2 in label_dict:
                size+=len(label_dict[label2])
            if size<min_frequency:
                del label2_label1_text_dict[label1]

    return label2_label1_text_dict

def build_dict(texts, labels1, labels2, languages = None, other = None, filter_language = 'en'):
    dictionary = dict()
    label1_set = set(labels1)
    for label in label1_set:
        dictionary[label] = dict()
    for i in range(len(texts)):
        if languages is not None:
            lang = languages[i]
        if languages is None or lang == filter_language:
            label1 = labels1[i]
            label2 = labels2[i]
            text = texts[i]

            if label2 not in dictionary[label1]:
                dictionary[label1][label2] = []
            if other is None:
                dictionary[label1][label2].append(text)
            else:
                dictionary[label1][label2].append((text, other[i]))

    return dictionary

def build_confusion_matrix(labels, predicted_labels, label2num = None):
    if label2num == None:
        label2num = dict()
        for label in labels:
            if label not in label2num:
                label2num[label] = len(label2num)
    confusion = np.zeros((len(label2num), len(label2num)))
    for i in range(len(labels)):
        label = labels[i]
        predicted_label = predicted_labels[i]
        confusion[label2num[label]][label2num[predicted_label]] +=1
    return confusion, label2num


def dict_to_lists(dictionary):
    x = []
    y = []
    for label in dictionary:
        l = dictionary[label]
        x.append(l)
        y += [label]* len(l)
    return x, y

def to_percentage(x):
    return str(round(x*100, 2))+"%"

def to_num(x):
    return round(float(x[:-1])/100,5)

def get_top(a, k=5):
    if a is np.ndarray:
        return heapq.nlargest(k, range(len(a)), a.take)
    else:
        return heapq.nlargest(k, range(len(a)), a.__getitem__)


it_stopwords = []
#read italian stopwords
#with open("data/stopwords.txt", 'r', encoding='utf-8') as f:
#    it_stopwords = f.readlines()
#    it_stopwords = [word.strip() for word in it_stopwords]
alphabet = list("abcdefghijklmnopqrstuvwxyz")
en_months = ['jan','january','feb','february','mar','march','apr','april','may','jun','june','jul','julie','aug','august','sept','sep','september','oct','october','nov','november','dec','december']
it_months = ['lunedi','martedi','mercoledi','giovedi','venerdi','sabato','domenica','finesettimana','gennaio','gen','febbraio','marzo','aprile','maggio','mag','giugno','giu','luglio','lug','agosto','ago','settembre','set','ottobre','ott','novembre','dicembre','dic']

stopwords = { "en":set(["","a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep	keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "'s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure	t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z"] + en_months + alphabet) ,
              "ro":set(["","a", "abia", "acea", "aceasta", "această", "aceea", "aceeasi", "acei", "aceia", "acel", "acela", "acelasi", "acele", "acelea", "acest", "acesta", "aceste", "acestea", "acestei", "acestia", "acestui", "aceşti", "aceştia", "acolo", "acord", "acum", "adica", "ai", "aia", "aibă", "aici", "aiurea", "al", "ala", "alaturi", "ale", "alea", "alt", "alta", "altceva", "altcineva", "alte", "altfel", "alti", "altii", "altul", "am", "anume", "apoi", "ar", "are", "as", "asa", "asemenea", "asta", "astazi", "astea", "astfel", "astăzi", "asupra", "atare", "atat", "atata", "atatea", "atatia", "ati", "atit", "atita", "atitea", "atitia", "atunci", "au", "avea", "avem", "aveţi", "avut", "azi", "aş", "aşadar", "aţi", "b", "ba", "bine", "bucur", "bună", "c", "ca", "cam", "cand", "capat", "care", "careia", "carora", "caruia", "cat", "catre", "caut", "ce", "cea", "ceea", "cei", "ceilalti", "cel", "cele", "celor", "ceva", "chiar", "ci", "cinci", "cind", "cine", "cineva", "cit", "cita", "cite", "citeva", "citi", "citiva", "conform", "contra", "cu", "cui", "cum", "cumva", "curând", "curînd", "când", "cât", "câte", "câtva", "câţi", "cînd", "cît", "cîte", "cîtva", "cîţi", "că", "căci", "cărei", "căror", "cărui", "către", "d", "da", "daca", "dacă", "dar", "dat", "datorită", "dată", "dau", "de", "deasupra", "deci", "decit", "degraba", "deja", "deoarece", "departe", "desi", "despre", "deşi", "din", "dinaintea", "dintr", "dintr-", "dintre", "doar", "doi", "doilea", "două", "drept", "dupa", "după", "dă", "e", "ea", "ei", "el", "ele", "era", "eram", "este", "eu", "exact", "eşti", "f", "face", "fara", "fata", "fel", "fi", "fie", "fiecare", "fii", "fim", "fiu", "fiţi", "foarte", "fost", "frumos", "fără", "g", "geaba", "graţie", "h", "halbă", "i", "ia", "iar", "ieri", "ii", "il", "imi", "in", "inainte", "inapoi", "inca", "incit", "insa", "intr", "intre", "isi", "iti", "j", "k", "l", "la", "le", "li", "lor", "lui", "lângă", "lîngă", "m", "ma", "mai", "mare", "mea", "mei", "mele", "mereu", "meu", "mi", "mie", "mine", "mod", "mult", "multa", "multe", "multi", "multă", "mulţi", "mulţumesc", "mâine", "mîine", "mă", "n", "ne", "nevoie", "ni", "nici", "niciodata", "nicăieri", "nimeni", "nimeri", "nimic", "niste", "nişte", "noastre", "noastră", "noi", "noroc", "nostri", "nostru", "nou", "noua", "nouă", "noştri", "nu", "numai", "o", "opt", "or", "ori", "oricare", "orice", "oricine", "oricum", "oricând", "oricât", "oricînd", "oricît", "oriunde", "p", "pai", "parca", "patra", "patru", "patrulea", "pe", "pentru", "peste", "pic", "pina", "plus", "poate", "pot", "prea", "prima", "primul", "prin", "printr-", "putini", "puţin", "puţina", "puţină", "până", "pînă", "r", "rog", "s", "sa", "sa-mi", "sa-ti", "sai", "sale", "sau", "se", "si", "sint", "sintem", "spate", "spre", "sub", "sunt", "suntem", "sunteţi", "sus", "sută", "sînt", "sîntem", "sînteţi", "să", "săi", "său", "t", "ta", "tale", "te", "ti", "timp", "tine", "toata", "toate", "toată", "tocmai", "tot", "toti", "totul", "totusi", "totuşi", "toţi", "trei", "treia", "treilea", "tu", "tuturor", "tăi", "tău", "u", "ul", "ului", "un", "una", "unde", "undeva", "unei", "uneia", "unele", "uneori", "unii", "unor", "unora", "unu", "unui", "unuia", "unul", "v", "va", "vi", "voastre", "voastră", "voi", "vom", "vor", "vostru", "vouă", "voştri", "vreme", "vreo", "vreun", "vă", "x", "z", "zece", "zero", "zi", "zice", "îi", "îl", "îmi", "împotriva", "în", "înainte", "înaintea", "încotro", "încât", "încît", "între", "întrucât", "întrucît", "îţi", "ăla", "ălea", "ăsta", "ăstea", "ăştia", "şapte", "şase", "şi", "ştiu", "ţi", "ţie"] + en_months + alphabet),
              "it":set(it_stopwords+["", "a", "adesso", "ai", "al", "alla", "allo", "allora", "altre", "altri", "altro", "anche", "ancora", "avere", "aveva", "avevano", "ben", "buono", "che", "chi", "cinque", "comprare", "con", "consecutivi", "consecutivo", "cosa", "cui", "da", "de", "del", "della", "dello", "dentro", "deve", "devo", "di", "doppio", "due", "e", "ecco", "fare", "fine", "fino", "fra", "gente", "giu", "ha", "hai", "hanno", "ho", "il", "indietro	invece", "io", "la", "lavoro", "le", "lei", "lo", "loro", "lui", "lungo", "ma", "me", "meglio", "molta", "molti", "molto", "nei", "nella", "no", "noi", "nome", "nostro", "nove", "nuovi", "nuovo", "o", "oltre", "ora", "otto", "peggio", "pero", "persone", "piu", "poco", "primo", "promesso", "qua", "quarto", "quasi", "quattro", "quello", "questo", "qui", "quindi", "quinto", "rispetto", "sara", "secondo", "sei", "sembra	sembrava", "senza", "sette", "sia", "siamo", "siete", "solo", "sono", "sopra", "soprattutto", "sotto", "stati", "stato", "stesso", "su", "subito", "sul", "sulla", "tanto", "te", "tempo", "terzo", "tra", "tre", "triplo", "ultimo", "un", "una", "uno", "va", "vai", "voi", "volte", "vostro"] + en_months + it_months + alphabet),
              }

def replace_diactitics(text):
    text = re.sub("[ăĂâÂ]", "a", text)
    text = re.sub("[îÎ]", "i", text)
    text = re.sub("[șȘ]", "s", text)
    text = re.sub("[țȚ]", "t", text)
    return text

printable = set(string.printable)
def remove_nonprintable(text):
    return list(filter(lambda x: x in printable, text))

def remove_numbers(text):
    return re.sub("[0-9., -]", "", text)

def keep_only_letters(text):
    return re.sub("[^a-z]", "", text)

def keep_only_letters_and_spaces(text):
    return re.sub("[^a-z ]", " ", text)

def process_words(text, language=None, stem=True, to_ascii=True, character_level=False):
    if language is None:
        translator = Translator()
        if isinstance(text, list):
            language = translator.detect(text)[0].lang

    if language == "ro":
        if isinstance(text, list):
            text = [replace_diactitics(subtext) for subtext in text]
        else:
            text = replace_diactitics(text)

    if isinstance(text, list):
        if to_ascii:
            text = [ unicodedata.normalize('NFKD', subtext).encode('ascii','ignore').decode("ascii") for subtext in text]
        text = [subtext.lower() for subtext in text]
    else:
        if to_ascii:
            text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode("ascii")
        text = text.lower()

    if language == "ro":
        stemmer = snowball.RomanianStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    elif language == "it":
        stemmer = snowball.ItalianStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    elif language == "en":
        stemmer = snowball.EnglishStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []
    else:
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    stopw = []
    if language in stopwords:
        stopw = stopwords[language]

    if isinstance(text, list):
        for i in range(len(words)):
            sent = words[i]
            sentence = []
            if stem:
                for word in sent:
                    word = keep_only_letters(word)

                    if word not in stopw:
                        if character_level:
                            sentence += list(word)
                        else:
                            sentence.append(stemmer.stem(word))
            else:
                for word in sent:
                    word = keep_only_letters(word)
                    if word not in stopw:
                        if character_level:
                            sentence += list(word)
                        else:
                            sentence.append(word)
            procced_text.append(sentence)
    else:
        for word in words:
            word = keep_only_letters(word)
            if word not in stopw:
                if character_level:
                    procced_text += list(word)
                else:
                    procced_text.append(word)

    return procced_text

char2number = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}

def process_chars(text, length, language = 'it'):
    if isinstance(text, list):
        result = np.zeros((len(text), length), dtype=np.int)
        for i in range(len(text)):
            text[i] = keep_only_letters_and_spaces(text[i])
            words = nltk.word_tokenize(text[i])
            j = 0
            while j<len(words):
                if words[j] in stopwords[language]:
                    del words[j]
                else:
                    j+=1
            text[i] = ' '.join(words)

            for j in range(min(len(text[i]),length)):
                result[i][j] = char2number[text[i][j]]
    else:
        text = keep_only_letters_and_spaces(text)
        result = np.zeros( length, dtype=np.int)
        words = nltk.word_tokenize(text)
        j = 0
        while j < len(words):
            if words[j] in stopwords[language]:
                del words[j]
            else:
                j += 1
        text = ' '.join(words)

        for j in range(min(len(text), length)):
            result[j] = char2number[text[j]]
    return result

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check_if_text_in_language(words, language = 'ita'):
    if not isinstance(words, list):
        words = words.split()
    for word in words:
        if not is_number(word):
            synsets = wn.synsets(word, lang=language)
            if len(synsets) == 0:
                return False
    return True

def process_for_named_entity(text, language, to_ascii = True, stem=False, shorten = False):
    if language is None:
        translator = Translator()
        if isinstance(text, list):
            language = translator.detect(text)[0].lang

    if language == "ro":
        if isinstance(text, list):
            text = [replace_diactitics(subtext) for subtext in text]
        else:
            text = replace_diactitics(text)

    if isinstance(text, list):
        if to_ascii:
            text = [ unicodedata.normalize('NFKD', subtext).encode('ascii','ignore').decode("ascii") for subtext in text]
        text = [subtext.lower() for subtext in text]
    else:
        if to_ascii:
            text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode("ascii")
        text = text.lower()

    if language == "ro":
        stemmer = snowball.RomanianStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    elif language == "it":
        stemmer = snowball.ItalianStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    elif language == "en":
        stemmer = snowball.EnglishStemmer()
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []
    else:
        if isinstance(text, list):
            words = [nltk.word_tokenize(subtext) for subtext in text]
        else:
            words = nltk.word_tokenize(text)
        procced_text = []

    if isinstance(text, list):
        for i in range(len(words)):
            sent = words[i]
            sentence = []
            if stem:
                for word in sent:
                    word = re.sub("[^a-z0-9]", "", word)
                    if word != '':
                        sentence.append(stemmer.stem(word))
            else:
                for word in sent:
                    word = re.sub("[^a-z0-9]", "", word)
                    if word != '':
                        sentence.append(word)
            procced_text.append(sentence)
    else:
        for word in words:
            word = re.sub("[^a-z0-9]", "", word)
            if word != '':
                if stem:
                    word = stemmer.stem(word)
                procced_text.append(word)

    if isinstance(text, list):
        for i in range(len(procced_text)):
            company_name = procced_text[i]
            if len(company_name) > 0 and company_name[0] != 'null':
                if False and company_name[-1] in ['srl', 'ltd', 'spa', 'ltda', 'sl', 'snc']:
                    contracted = ' '.join(company_name[:-1])
                    if not check_if_text_in_language(company_name[:-1]) and len(contracted) > 6 and not is_number(contracted) and contracted not in ['data', 'aprile','group',
                                                                                                'azienda', 'profilo', 'alumino',
                                                                                                'stato', 'roma', 'service',
                                                                                                'area', 'estate',
                                                                                                'date 4', 'work',
                                                                                                'altre', 'italia',
                                                                                                'stage', 'ottobre 2008',
                                                                                                'strada', '16 luglio',
                                                                                                'espresso', 'export',
                                                                                                'prime', 'sala', 'panelli']:

                        del company_name[-1]
                        if shorten:
                            while len(contracted) > 23:
                                if len(contracted) - len(company_name[0]) <15:
                                    break
                                del company_name[0]
                                contracted = ' '.join(procced_text)
                procced_text[i] = ' '.join(company_name)
    else:

        if len(procced_text) > 0 and procced_text[0] != 'null':
            if False and procced_text[-1] in ['srl', 'ltd', 'spa', 'ltda', 'sl', 'snc']:
                contracted = ' '.join(procced_text[:-1])
                if not check_if_text_in_language(procced_text[:-1]) and len(contracted) > 6 and not is_number(
                        contracted) and contracted not in ['data', 'aprile', 'group',
                                                           'azienda', 'profilo', 'allumino',
                                                           'stato', 'roma', 'service',
                                                           'area', 'estate', 'metalmeccanica'
                                                           'date 4', 'work', 'castel'
                                                           'altre', 'italia', 'controlo qualita',
                                                           'stage', 'ottobre 2008', 'atena',
                                                           'strada', '16 luglio', 'industriale',
                                                           'espresso', 'export',
                                                           'prime', 'sala', 'panelli']:
                    del procced_text[-1]
                    if shorten:
                        while len(contracted) > 23:
                            if len(contracted) - len(procced_text[0]) < 15:
                                break
                            del procced_text[0]
                            contracted = ' '.join(procced_text)

            procced_text = ' '.join(procced_text)

    return procced_text


def get_words(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=';', quotechar='"', skipinitialspace=True)
        texts = []
        labels1 = []
        labels2 = []
        for row in reader:
            texts.append(row[0])
            labels1.append(row[2])
            labels2.append(row[3])
    words = []
    for text in texts:
        words.append(nltk.word_tokenize(text))
    return words, labels1, labels2

def one_hot(nums, size=None):
    nums = np.array(nums, dtype=np.int)
    max_size = int(np.amax(nums)+1)
    if size is None:
        size = max_size
    result = []
    if size>=max_size:
        if len(np.shape(nums))==1:
            labels = np.zeros((len(nums), size), dtype=np.bool)
            labels[np.arange(len(nums)), nums] = 1
            return labels
        for sentence in nums:
            words = np.zeros((len(sentence), size), dtype=np.bool)
            words[np.arange(len(sentence)), sentence] = 1
            result.append(words)
        return np.array(result, dtype=np.bool)
    else:
        aux =np.concatenate((np.eye(size, dtype=np.bool), np.zeros((max_size-size, size), dtype=np.bool)), axis=0)
        if len(np.shape(nums)) == 1:
            return aux[nums]
        for sentence in nums:
            result.append(aux[sentence])
        return np.array(result, dtype=np.bool)

def one_hot_to_num(x):
    ax = len(np.shape(x))-1
    return np.argmax(x, axis=ax)

def get_centroid(text,word2vec):
    if len(text)<50:
        n = 0
        vec = np.zeros(300)
        for word in text:
            if word in word2vec:
                vec = np.add(vec,word2vec[word])
                n+=1

        return np.nan_to_num(vec/n)
    else:
        return np.nan_to_num( np.add(get_centroid(text[:len(text)//2],word2vec),get_centroid(text[len(text)//2:],word2vec))/2)

def label2cluster(texts, word2vec, labels, kmeans_file):
    with open(kmeans_file,'rb') as f:
        kmeans = pickle.load(f)

    for i in range(len(labels)):
        if labels[i] in kmeans:
            labels[i] = labels[i] + str( kmeans[labels[i]].predict([get_centroid(texts[i],word2vec)])[0] )

    return labels

def plot_confusion_matrix(cm, Score, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(Score)))
    plt.xticks(tick_marks, set(Score), rotation=45)
    plt.yticks(tick_marks, set(Score))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def append_data(file1,file2):
    rows1 = []
    ids1 = []
    encoding = 'utf-8'
    with codecs.open(file1, 'r', encoding=encoding) as f:
       reader = csv.reader(f, delimiter=',', quotechar='"', dialect='excel')
       for row in reader:
           rows1.append(row)
           try:
            ids1.append(row[26])
           except Exception:
            print(row)

    rows2 = []
    ids2 = []
    encoding = 'utf-16'
    with codecs.open(file2, 'r', encoding=encoding) as f:
       reader = csv.reader(f, delimiter=',', quotechar='"', dialect='excel')
       for row in reader:
           rows2.append(row)
           ids2.append(row[1])

    combined_rows = []
    for i2 in range(len(ids2)):
        id2 = ids2[i2]
        try:
            i = ids1.index(id2)
            row = rows1[i]+rows2[i2][-20:]
            combined_rows.append(row)
        except Exception:
            print(id2)
            continue

    filename = 'data/combined-fold'+file2[19]+'.csv'
    write_csv(filename,combined_rows)
    write_csv_to_xlsx(filename)

