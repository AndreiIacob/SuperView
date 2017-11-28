import data_helpers
import pickle
import collections
from nltk.corpus import wordnet as wn
from nltk import ne_chunk
import random
import nltk
import codecs
import csv
from nltk.chunk import conlltags2tree, tree2conlltags
import numpy as np
import tensorflow as tf
from nltk.util import ngrams, skipgrams

def get_word_synonyms(word, language):
    synsets = wn.synsets(word, lang=language)
    if len(synsets) == 0:
        return [word]
    synonyms = []
    for synset in synsets:
        synonyms += synset.lemma_names(language)
    return synonyms

def get_word_hypernyms(word, language):
    synsets = wn.synsets(word, lang=language)
    if len(synsets) == 0:
        return []
    hypernyms_synsets = []
    for synset in synsets:
        hypernyms_synsets += synset.hypernyms()
    if len(hypernyms_synsets) == 0:
        return []
    hypernyms = []
    for synset in hypernyms_synsets:
        hypernyms += synset.lemma_names(language)
    return hypernyms


def replace_with_synonyms(texts, language, max_count=4):
    if language == 'it':
        language = 'ita'

    words_flat = [word for sublist in texts for word in sublist]
    count = collections.Counter(words_flat).most_common()
    word2synonyms = dict()
    synonym2frequency = dict()
    for word, cnt in count:
        if cnt <= max_count:
            synonyms = get_word_synonyms(word, language)
            for synonym in synonyms:
                if synonym in synonym2frequency:
                    synonym2frequency[synonym] += 1
                else:
                    synonym2frequency[synonym] = 1
            word2synonyms[word] = synonyms

    word2best_synonym = dict()
    for word in word2synonyms:
        best = 0
        for synonym in word2synonyms[word]:
            if synonym2frequency[synonym] > best:
                best_synonym = synonym
                best = synonym2frequency[synonym]
        word2best_synonym[word] = best_synonym

    for text in texts:
        for i in range(len(text)):
            if text[i] in word2best_synonym:
                text[i] = word2best_synonym[text[i]]

    return texts


def replace_with_hypernyms(texts, language, max_count=4):
    if language == 'it':
        language = 'ita'

    words_flat = [word for sublist in texts for word in sublist]
    count = collections.Counter(words_flat).most_common()
    word2hypernyms = dict()
    hypernym2frequency = dict()
    for word, cnt in count:
        if cnt <= max_count:
            hypernyms = get_word_hypernyms(word, language)
            for hypernym in hypernyms:
                if hypernym in hypernym2frequency:
                    hypernym2frequency[hypernym] += 1
                else:
                    hypernym2frequency[hypernym] = 1
            word2hypernyms[word] = hypernyms

    word2best_hypernym = dict()
    for word in word2hypernyms:
        best = 0
        for hypernym in word2hypernyms[word]:
            if hypernym2frequency[hypernym] > best:
                best_hypernym = hypernym
                best = hypernym2frequency[hypernym]
        word2best_hypernym[word] = best_hypernym

    for text in texts:
        for i in range(len(text)):
            if text[i] in word2best_hypernym:
                text[i] = word2best_hypernym[text[i]]

    return texts

def process_text(texts):
    min_count = 5
    words_flat = [word for sublist in texts for word in sublist]
    count = collections.Counter(words_flat).most_common()
    low_count_words = 0
    for word, cnt in count:
        if cnt < min_count:
            low_count_words += 1

    print(low_count_words)

    for i in range(3):

        texts = replace_with_synonyms(texts, 'ita')
        with open('data/synonyms_used.p','wb+') as f:
            pickle.dump(texts,f)

        words_flat = [word for sublist in texts for word in sublist]
        count = collections.Counter(words_flat).most_common()
        low_count_words = 0
        for word, cnt in count:
            if cnt < min_count:
                low_count_words += 1

        print(low_count_words)

        texts = replace_with_synonyms(texts, 'ita')
        with open('data/hypernyms_used.p','wb+') as f:
            pickle.dump(texts,f)

        words_flat = [word for sublist in texts for word in sublist]
        count = collections.Counter(words_flat).most_common()
        low_count_words = 0
        for word, cnt in count:
            if cnt < min_count:
                low_count_words += 1

        print(low_count_words)
    return texts


def extend_short_texts_using_definition(texts, min_words = 5, language='ita'):
    if language == 'it':
        language = 'ita'

    for j in range(len(texts)):
        text = texts[j]
        if len(text)< min_words:
            new_text = []
            for i in range(len(text)):
                synsets = wn.synsets(text[i], lang=language)
                if len(synsets) > 0:
                    new_text += synsets[0].lemma_names(lang="ita")
                    text[i] = synsets[0].lemma_names(lang="ita")
                else:
                    new_text.append(text[i])
            texts[j] = new_text

    return texts

def generate_new_text(text, language):

    if language == 'it':
        language = 'ita'
    new_text = []

    for word in text:
        alternatives = get_word_synonyms(word,language) + get_word_hypernyms(word,language)
        if len(alternatives) <= 1:
                new_text.append(word)
                continue

        rand = random.randint(0,len(alternatives)-1)
        new_text.append(alternatives[rand])
    return new_text


def enrich_text(file='data/companies_enriced.p', language='it', min_count=1000, min_words = 5):
    f = open("data/folds_data.p", "rb")
    original_texts, fold_texts, fold_labels, fold_labels2, fold_ids, n_input, n_steps, n_classes, stop_points, word2number, number2word, label2number, number2label = pickle.load(
        f)
    f.close()

    fold_labels = []
    fold_labels2 = []
    fold_ids = []
    stop_points = []
    original_texts = []
    for fold in range(1,6):
        rows = []
        encoding = 'utf-16'
        with codecs.open('data/combined-fold'+str(fold)+'.txt', 'r', encoding=encoding) as f:
           reader = csv.reader(f, delimiter='\t', quotechar='"', dialect='excel')
           skip_initial = True
           i = 0
           for row in reader:
                if skip_initial:
                   skip_initial = False
                else:
                   rows.append(row)
                   fold_labels.append(row[1])
                   fold_labels2.append(row[2])
                   fold_ids.append(row[26])
                   original_texts.append(row[0])

           stop_points.append(len(rows))


    id_num = 0
    for id in fold_ids:
        num = int(id.split('_')[-1])
        if num>id_num:
            id_num = num

    id_num+=1

    with open(file, 'rb') as f:
        texts = pickle.load(f)

    print(len(texts),len(fold_labels))
    print(texts[0])
    texts = data_helpers.process_words(texts, language)
    texts = extend_short_texts_using_definition(texts, min_words)

    texts = process_text(texts)

    print(len(texts),len(fold_labels))
    print(texts[0])

    label2count = dict()
    count = collections.Counter(fold_labels).most_common()
    for label, cnt in count:
        label2count[label] = cnt

    make_more = True

    new_texts   = []
    new_labels  = []
    new_labels2 = []
    new_ids     = []
    while make_more:
        for i in range(len(fold_labels)):
            label = fold_labels[i]
            if label2count[label] < min_count:
                new_texts.append(generate_new_text(texts[i],language))
                new_labels.append(label)
                new_labels2.append(fold_labels2[i])
                new_ids.append('_'.join(fold_ids[i].split('-')[:-1] + [str(id_num)]))
                id_num += 1
                label2count[label]+=1
        make_more = False
        for label in label2count:
            if label2count[label] < min_count:
                make_more = True
                break

    text_ids = list(zip(new_texts,new_ids))

    statified_data = data_helpers.build_dict(text_ids,new_labels,new_labels2)
    for label in statified_data:
        for label2 in statified_data[label]:
            split_val = len(statified_data[label][label2])//5
            new_texts, new_ids = list(zip(*statified_data[label][label2]))
            new_labels = [label]*len(statified_data[label][label2])
            new_labels2 = [label2] * len(statified_data[label][label2])
            for i in range(5):
                texts = texts[:stop_points[i]] + list(new_texts[i*split_val:(i+1)*split_val]) + texts[stop_points[i]:]
                fold_labels = fold_labels[:stop_points[i]] + new_labels[i * split_val :(i + 1) * split_val] + fold_labels[
                                                                                                     stop_points[i]:]
                fold_labels2 = fold_labels2[:stop_points[i]] + new_labels2[i * split_val :(i + 1) * split_val] + fold_labels2[
                                                                                                     stop_points[i]:]

                fold_ids = fold_ids[:stop_points[i]] + list(new_ids[i * split_val :(i + 1) * split_val]) + fold_ids[
                                                                                                     stop_points[i]:]

                for j in range(4,i-1,-1):
                    stop_points[i]+= split_val

    fold_labels,label2number, number2label = data_helpers.convert_label_to_num(fold_labels)
    with open('data/enriched_dataset.p', 'wb+') as f:
        pickle.dump([original_texts, texts, fold_labels, fold_labels2, fold_ids, n_input, n_steps, n_classes, stop_points, word2number, number2word, label2number, number2label],f)
    return original_texts, texts, fold_labels, fold_labels2, fold_ids, n_input, n_steps, n_classes, stop_points, word2number, number2word, label2number, number2label

lemmatizer = nltk.stem.WordNetLemmatizer()

def get_closest_common_synset2(word1, word2, language = "eng"):
    distance = 0
    synsets1 = wn.synsets(lemmatizer.lemmatize(word1), lang=language)
    synsets2 = wn.synsets(lemmatizer.lemmatize(word2), lang=language)
    synset2distance = dict()
    synset2word     = dict()
    while len(synsets1) > 0 or len(synsets2) > 0:
        new_synsets1 = []
        for synset in synsets1:
            if synset in synset2distance:
                if synset2word == 2:
                    return (synset, distance + synset2distance[synset])
            else:
                synset2distance[synset] = distance
                synset2word[synset]     = 1
                new_synsets1 += synset.hypernyms()

        new_synsets2 = []
        for synset in synsets2:
            if synset in synset2distance:
                if synset2word[synset] == 1:
                    return (synset, distance + synset2distance[synset])
            else:
                synset2distance[synset] = distance
                synset2word[synset]     = 2
                new_synsets2 += synset.hypernyms()
        synsets1 = new_synsets1
        synsets2 = new_synsets2
        distance+=1
    return (None,-1)

def get_closest_common_synset(words, language = "eng", max_distance = 10000, full_coverage_result = True, max_responses = 5):
    distance = 0
    all_current_synsets = []
    for word in words:
        all_current_synsets.append(wn.synsets(lemmatizer.lemmatize(word), lang=language))

    result_synsets = []
    synset2distances= dict()
    synset2words    = dict()

    more_synsets_to_check = True
    while more_synsets_to_check:
        all_new_synsets = []
        i = -1
        for synsets in all_current_synsets:
            i += 1
            new_synsets = []
            for synset in synsets:
                if synset in synset2distances:
                    if i not in synset2words[synset]:
                        synset2words[synset].append(i)
                        synset2distances[synset][i] = distance
                        if len(synset2words[synset]) == len(words):
                            result_synsets.append((synset, synset2distances[synset]))
                else:
                    if synset not in synset2distances:
                        synset2distances[synset] = [-1] * len(words)
                    synset2distances[synset][i] = distance
                    synset2words[synset] = [i]
                    new_synsets += synset.hypernyms()
            all_new_synsets.append(new_synsets)

        if len(result_synsets) > 0:
            return result_synsets

        all_current_synsets = all_new_synsets
        distance+=1
        more_synsets_to_check = False

        if distance >= max_distance:
            break

        for synset in all_current_synsets:
            if len(synset) > 0:
                more_synsets_to_check = True
                break

    common_synsets = []
    if full_coverage_result:
        result_synsets = get_synsets_for_full_coverage(synset2words, number_of_words=len(words))
        for synset in result_synsets:
            common_synsets.append((synset, synset2distances[synset]))
    else:
        min_common_words = 1000
        for synset in synset2distances:
            if len(common_synsets) == max_responses:
                if len(synset2words[synset]) > min_common_words:
                    for i in range(len(common_synsets)):
                        selected_synset = common_synsets[i][0]
                        if len(synset2words[selected_synset]) == min_common_words:
                            common_synsets[i] = (synset, synset2distances[synset])
                            break
                    min_common_words = len(synset2words[synset])
                    for i in range(len(common_synsets)):
                        if min_common_words > len(synset2words[synset]):
                            min_common_words = len(synset2words[synset])
            else:
                common_synsets.append((synset, synset2distances[synset]))
                if len(synset2words[synset]) < min_common_words:
                    min_common_words = len(synset2words[synset])

    return common_synsets


def get_synsets_for_full_coverage(synset2words, number_of_words = None):
    if number_of_words is None:
        covered = []
        for synset in synset2words:
            for word in synset2words[synset]:
                if word not in covered:
                    covered.append(word)
        number_of_words = len(covered)

    result_synsets = []
    covered = set()
    while len(covered) < number_of_words:
        max_coverage = 0
        for synset in synset2words:
            uncovered = []
            for word in synset2words[synset]:
                if word not in covered:
                    uncovered.append(word)
            if len(uncovered) > max_coverage:
                max_coverage = len(synset2words[synset])
                max_synset   = synset
        covered |= set(synset2words[max_synset])
        result_synsets.append(max_synset)
    return result_synsets

def get_word_contexts(reviews, size = 2):
    word2context = dict()
    for review in reviews:
        i = -1
        while i < len(review["words"]):
            i+=1
            word = review["words"][i]
            if word not in word2context:
                word2context[word] = dict()
            for j in range(1, size + 1):
                if (i-j)>=0:
                    if review["words"][i-j] not in word2context[word]:
                        word2context[word][review["words"][i-j]] = 0
                    word2context[word][review["words"][i - j]] += 1
                if (i+j)<len(review["words"]):
                    if review["words"][i+j] not in word2context[word]:
                        word2context[word][review["words"][i+j]] = 0
                    word2context[word][review["words"][i + j]] += 1

def get_ngram_scores(reviews, n = 2, min_instances = None):
    if min_instances is None:
        min_instances = min([5, len(reviews) / 100])

    ngram2score = dict()
    for review in reviews:
        i = -1
        stars = review['stars']
        ngram = []
        while i < len(review["words"]):
            i+=1
            word = review["words"][i]
            ngram.append(word)
            if len(ngram) > n:
                ngram = ngram[1:]
            if len(ngram) == n:
                ngram_str = " ".join(ngram)
                if ngram_str not in ngram2score:
                    ngram2score[ngram_str] = [0.0, 0]
                ngram2score[ngram_str][0] += stars
                ngram2score[ngram_str][1] += 1

    for ngram in ngram2score:
        if ngram2score[ngram][1] < min_instances:
            del ngram2score[ngram]
        else:
            ngram2score[ngram][0] =  ngram2score[ngram][0]/ ngram2score[ngram][1]
    return ngram2score

def get_word_context_scores(reviews, n = 2, k=2, min_instances = None, min_percentage = 1.0):
    if min_instances is None:
        min_instances = min([5, int((len(reviews) / 100) * min_percentage)])

    word2context_score = dict()
    for review in reviews:
        i = -1
        stars = review['stars']
        skipgrams_list = skipgrams(review["words"],n,k)
        for skipgram in skipgrams_list:
            for i in range(len(skipgram)-1):
                word1 = skipgram[i]
                for j in range(i+1,len(skipgram)):
                    word2 = skipgram[j]
                    if word1 not in word2context_score:
                        word2context_score[word1] = dict()
                    if word2 not in word2context_score:
                        word2context_score[word2] = dict()
                    if word1 not in word2context_score[word2]:
                        word2context_score[word1][word2] = [0.0,0]
                    if word2 not in word2context_score[word1]:
                        word2context_score[word2][word1] = [0.0,0]
                    word2context_score[word1][word2][0] += stars
                    word2context_score[word1][word2][1] += 1
                    word2context_score[word2][word1][0] += stars
                    word2context_score[word2][word1][1] += 1

    for word1 in word2context_score:
        instances = 0
        for word2 in word2context_score[word1]:
            instances += word2context_score[word1][word2][1]
            if instances >= min_instances:
                break
        if instances<min_instances:
            del word2context_score[word1]
    return word2context_score

if __name__ == "__main__":
    synsets_distances = get_closest_common_synset(["room","kitchen", "service"])
    print(synsets_distances)
    for synset, distance in synsets_distances:
        print(synset.lemma_names())

    if False:
        enrich_text()
        with open('data/enriched_dataset.p', 'rb') as f:
            original_texts, texts, fold_labels, fold_labels2, fold_ids, n_input, n_steps, n_classes, stop_points, word2number, number2word, label2number, number2label = pickle.load(f)

        document_size = 100

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            document_size)

        texts = [' '.join(text) for text in texts]
        texts_transform_train = vocab_processor.fit_transform(texts)

        texts = np.array(list(texts_transform_train))

        number_of_words = len(vocab_processor.vocabulary_)
        print('Total words: %d' % number_of_words)

        word2number = vocab_processor.vocabulary_._mapping
        number2word = dict(zip(word2number.values(), word2number.keys()))

        n_input = number_of_words + 1
        n_classes = len(label2number)
        n_steps = document_size

        with open('data/enriched_dataset-processed.p', 'wb+') as f:
            pickle.dump([original_texts, texts, fold_labels, fold_labels2, fold_ids, n_input, n_steps, n_classes, stop_points,
                         word2number, number2word, label2number, number2label], f)


        f = open("data/enriched_dataset-processed.p", "rb")
        x,fold_texts,fold_labels,y2,fold_ids,n_input,n_steps,n_classes,stop_points,word2number,number2word,label2number,number2label = pickle.load( f)
        f.close()







