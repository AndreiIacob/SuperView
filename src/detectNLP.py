#from spacy.en import English
#parser = English()
from numpy import dot
from numpy.linalg import norm
import numpy as np
from helper_functions import *
import spacy
parser = spacy.load('en_core_web_md')

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# cosine similarity
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
lemmatizer = nltk.stem.WordNetLemmatizer()

def get_named_entities(sentence):
    parsedEx = parser(sentence)
    named_entities = []
    ents = list(parsedEx.ents)
    for ent in ents:
        named_entities.append((ent.label_, ent.text))
    return named_entities

def get_average_wordvec(sentence):
    parsedEx = parser(sentence)
    average_wordvec = np.zeros(300)
    total = 0
    for word in parsedEx:
        if word.has_vector:
            total += 1
            average_wordvec += word.vector
    average_wordvec /= total
    return average_wordvec

def most_similar_word(word):
   queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
   by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
   return by_similarity[:10]

def filter_words(word, filter_words_list):
    for filter_word in filter_words_list:
        if word.lower_ == filter_word:
            return False
    return True

def most_similar_vector(vector, filter_words_list = [], number_of_words = 10):
    allWords = list({w for w in parser.vocab if
                     w.has_vector and w.orth_.islower() and filter_words(w, filter_words_list)})
    allWords.sort(key=lambda w: cosine(w.vector, vector), reverse=True)
    return allWords[:number_of_words]

def get_closest_texts(text_to_compare, texts):
    important_text_vec = get_average_wordvec(text_to_compare)
    smallest_distance = -1

    for text in texts:
        text_vec = get_average_wordvec(text)
        distance = cosine(text_vec, important_text_vec)
        if distance > smallest_distance:
            smallest_distance = distance
            representative_text = text

    return representative_text

def get_number_of_words_in_list(text, word_list):
    words = process_words(text)
    score = 0
    for word in words:
        if word in word_list:
            score += 1
    return score

def get_contains_most_words(texts, word_list):
    best_score = -1
    for text in texts:
        score = get_number_of_words_in_list(text)
        if score > best_score:
            best_score = score
            best_text  = text
    return best_text

def process_words(sentence):
    tokenized_text = nltk.word_tokenize(sentence.lower())
    tags = tuple_to_list(nltk.pos_tag(tokenized_text))
    return tags

def tuple_to_list(t):
    l = []
    for i in t:
        l.append(list(i))
    return l

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

lemmatzr = WordNetLemmatizer()
def get_synsets(tags):
    synsets=[]
    synsets_found=0
    for token in tags:
        wn_tag = penn_to_wn(token[1])
        try:
            txt = token[0].lower()
            txt = txt.decode('utf-8').replace(u"’s", u"")
            if not wn_tag:
                lemma = lemmatzr.lemmatize(txt)
            else:
                lemma = lemmatzr.lemmatize(txt, pos=wn_tag)

            synsets.append(wn.synsets(lemma, pos=wn_tag))
            synsets_found += 1
        except Exception:
            pass
    return synsets,synsets_found


def replace_with_synonyms2(sentence):
    from PyDictionary import PyDictionary
    tokenized_text = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokenized_text)
    dictionary = PyDictionary()
    for i in range(len(tokenized_text)):
        word = tokenized_text[i]
        wn_tag = penn_to_wn(tags[i][1])
        if not wn_tag:
            continue
        if word.startswith("'"):
            continue
        if random.random() < 0.6:
            continue
        synonyms = dictionary.synonym(word)
        rand = random.randrange(0, len(synonyms))
        tokenized_text[i] = synonyms[rand]
    return nltk.untokenize(tokenized_text)


def get_hyponyms_synsets(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms_synsets(hyponym))
    return hyponyms | set(synset.hyponyms())


def get_hyponyms_names(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms_names(hyponym))
    lemmas = []
    for hyponym in synset.hyponyms():
        for word in hyponym.lemma_names():
            lemmas.append(word)
    return hyponyms | set(lemmas)


def get_hyponyms(word, tag=None):
    lemma = lemmatzr.lemmatize(word.lower(), pos=tag)
    hyponyms = set()
    for synset in wn.synsets(lemma, pos=tag):
        hyponyms |= get_hyponyms_names(synset)
    return hyponyms


def get_hypernyms_synsets(synset):
    hypernyms = set()
    for hyponym in synset.hypernyms():
        hypernyms |= set(get_hypernyms_synsets(hyponym))
    return hypernyms | set(synset.hypernyms())


def get_hypernyms_names(synset):
    hypernyms = set()
    for hyponym in synset.hypernyms():
        hypernyms |= set(get_hypernyms_names(hyponym))
    lemmas = []
    for hyponym in synset.hypernyms():
        for word in hyponym.lemma_names():
            lemmas.append(word)
    return hypernyms | set(lemmas)


def get_hypernyms(word, tag=None):
    if tag != None:
        lemma = lemmatzr.lemmatize(word.lower(), pos=tag)
    else:
        lemma = lemmatzr.lemmatize(word.lower())
    hypernyms = set()
    for synset in wn.synsets(lemma, pos=tag):
        hypernyms |= get_hypernyms_names(synset)
    return hypernyms


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

if __name__ == "__main__":

    words = ["place", "stay", "stayed", "strip", "view", "situated"]
    words = "room street smell night small alley staff traffic noise recommend stayed loud place stay hotel".split()
    for word in words:
        print(word)
        lemma = lemmatzr.lemmatize(word)
        print()
        for synset in wn.synsets(lemma):
            print(get_hypernyms_names(synset))
        print()


    sentence = "London is a big city in the United Kingdom."
    sentence = "London Jones is my best friend."
    sentence = "London Inc. is a terrible company."

    very_bad_food_1 = ["buffet bacchanal favorite best wynn better quality strip bellagio wicked spoon palace recommend expensive dinner experience went person going think","food quality selection variety amazing asian delicious chinese excellent fresh better mexican high price mediocre lot average different stations presentation","wait line time long hour hours minutes seated pay got 30 come waiting went people waited dinner came 15 lines","good pretty really service desserts overall sushi stuff thing prime rib experience tried looked taste actually quality variety lamb flavor","crab legs king oysters seafood shrimp hot cold snow fresh steamed claws rib prime stone mussels meat cocktail warm salty","worth price money definitely pricey wait think penny 50 expensive person totally dinner 60 eat experience overall high paid sure","rrb lrb dinner lunch person price rib prime sushi brunch chinese asian 99 dim sum spoon bacchanal wicked lamb chicken","vegas best buffets ve las definitely far time hands probably trip strip come favorite tried try bacchanal say better visit","great service selection experience variety overall friendly excellent customer staff attentive desserts server seafood quality drinks presentation nice time atmosphere","place just dessert like did try really seafood station section amazing eat desserts bar selection fresh meat love asian think"]

    print("Topics for Bacchanal Buffet")

    for topic in very_bad_food_1:
        words = topic.split()
        print(words)
        print(get_closest_common_synset(words))

    NMF_fn = ["room street smell night small alley staff traffic noise recommend stayed loud place stay hotel big helpful late comfortable location","hotel great pioneer breakfast square free don hotels downtown location homeless rooms rest com blocks district good got market bit","parking view city public did western best lot booked room space away square park needle deluxe blocks arrived pioneer stadium","just inside quiet station expensive near area clean night hotel room beds time friends deluxe extremely door given getting garage","told actually called needed said problem don business company trip like day room located felt garage friends friendly freeway free","door number walking distance complimentary located right friendly helpful just seattle staff breakfast hotel expensive garage friends district don freeway","pretty bed getting area stayed pillows customer bad parking large hotel service place best walk worst check rating ones rated","area near nearer places want closer market terrible far viaduct staying pike noisy hotel nearby better like people lot nice","space good place hotel open historic elevator bathroom building fine lobby walked station internet old price district small beds service","manager lobby door treated seattle think pull need outside counter days making time sure left times continental trying coming did"]
    NMF_kb = ["room staff hotel night place breakfast noise good stayed small stay nice location walk street clean seattle parking coming desk","pioneer square downtown breakfast nice better pike people worth hotel rooms market ve closer free overall elevator great homeless just","pioneer square parking away walk service stadium park city lot blocks view western best public booked given couple space did","area beds street hotel quiet expensive near night people inside just station extremely view nearer nearby lot noisy comfortable places","room like nights price don actually offered hotels stay small day location clean booked problem told company called needed rest","staff just seattle complimentary breakfast walking right helpful better friendly door did charged treated located key distance places 50 number","rate bad parking money feel worth extremely hotel area woman uncomfortable rated downtown expensive garage safe homeless blocks recommend bed","avoid best nice grungy minutes near check western area location worst went pretty walking bus nearer stop nicer rooms bars","place service clean western fine nights price stayed beds hotel historic lobby old just elevator best district good building neighborhood","continental money nice left like seattle time arrived older nearby pull stay slow treated ve building station nicely need large"]
    LDA_tf = ["parking hotel room stayed arrived area pretty seattle noise lot bad money blocks safe large staff away bed feel nearby","hotel pioneer breakfast square room rooms best western parking better downtown night just seattle stay great space noise did noisy","room area hotel staying street places nice smell alley like night better small near staff clean big hotels helpful away","overall hotel luggage night staying breakfast recommend 20 market rude weekend view people friendly closer itâ foot didn pioneer don","manager lobby door treated seattle days nice time sure breakfast think did coming counter stay couple continental like times trying","hotel good place space historic internet building price bathroom small nights station nice lobby clean staff walked room just elevator","room hotel view parking deluxe city public desk friends staff price garage stay booked beds inside small night space pioneer","hotel room expensive parking location got clean stay near neighborhood homeless night ve ones rating area seattle garage went itâ","hotel location good night stayed western best needle room nice place couple parking avoid service walk 50 area grungy nights","told door friendly seattle walking business breakfast actually room problem like staff right called number hotel day complimentary just located"]

    negative_topics = [NMF_fn, NMF_kb, LDA_tf]

    NMF_fn = ["hotel seattle room pioneer square great staff stay breakfast best rooms location walk good area free walking western downtown close","friendly helpful clean staff good character location fantastic desk nice free breakfast stay lots appointed complimentary nicely gem great definitely","blocks rail light station airport 50 amtrack took pike worked tac walk market sea walked cruise waterfront terminal person clean","terrific upgraded room view good quick exceptional suite arrived particularly street totally check single small fact extra facilities faced extremely","great size bed nights comfortable stayed room items spacious breakfasts days service instead expectations facilities far european fantastic famous family","site spot vintage appearance checking parking style use attractions blocks distance located waterfront close old walking hotel nice clean room","spring knew basket anniversary champagne activities closest week charming return waiting including fun year wonderful end district arrived feel distance","field safeco enjoy waterfront going game quest feel stocked special handy qwest ave walk years place blocks stayed times professional","station super ball st quiet king little underground great close amtrak game short store cool bus concierge 99 walk street","excellent 100 service right price night stay breakfast certainly help hotel home received way access having arrival travellers assistance finding"]
    NMF_kb = ["hotel seattle square staff pioneer breakfast room stayed great just western best waffles like time night stay location nights things","staff friendly breakfast location stay helpful clean walk stayed great nice good free rooms recommend desk easy waffles perfect water","pike walking market walk blocks station hotel place waterfront distance underground walked tour rail bus street restaurants took airport historic","room nice street use did good internet upgraded check high walking small ll area reviews quite rate night shops wi","seattle walking view great stayed room trip small street good historic nice rooms location walk water underground happy wonderful distance","stay hotel parking night staff time right located old friendly rooms service attractions lot don away large tour quite style","hotel seattle staff recommend walking room stay distance return wonderful including desk rooms transportation pike washington small extremely breakfast quiet","square room waterfront safeco pioneer staff western market staying stadiums place old pike years street professional stayed rooms helpful pikes","square pioneer restaurants quiet comfortable clean rooms helpful underground close bed western spacious great station pleasant little super located old","service stay way excellent night stayed thank towels breakfast home great day right getting outstanding foot traffic definitely housekeeping place"]
    LDA_tf = ["hotel great room seattle stay parking small waterfront free did square staff rooms tour desk best good walking block pioneer","hotel room location breakfast clean great staff friendly stay close seattle good place walking parking night nice stayed helpful free","hotel room seattle great staff pioneer square breakfast stay location best clean good friendly walk free helpful nice rooms walking","hotel stayed staff walk breakfast good parking camera pike shops market lot nice night seattle friendly away square downtown place","historic hotel staying helpful street stay allowed august exceptionally middle nice nights stars wonderful close south time lively couldn return","pioneer room square hotel parking area western perfectly located street charm worth staff downtown seattle cost zone close did television","hotel great market walk square walking underground pioneer bus tour property stay seattle area friendly old straight free close hard","staff hotel good great breakfast location helpful like friendly seattle close bus time free square stay rooms clean pioneer came","room free came quiet pioneer king large seattle modern place asked breakfast ferry staff friendly talked wifi period variety kept","hotel seattle square room pioneer breakfast blocks tour old staff free block pike western nights walked way excellent friendly great"]

    positive_topics = [NMF_fn, NMF_kb, LDA_tf]

    print("Negative")
    for method in negative_topics:
        for topic in method:
            avg_vec = get_average_wordvec(topic)
            words = most_similar_vector(avg_vec)
            most_similar_words = []
            for word in words:
                most_similar_words.append(word.lower_)
            print(topic)
            print(most_similar_words)
            print('\n')

    print("Positive")
    for method in positive_topics:
        for topic in method:
            avg_vec = get_average_wordvec(topic)
            words = most_similar_vector(avg_vec)
            most_similar_words = []
            for word in words:
                most_similar_words.append(word.lower_)
            print(topic)
            print(most_similar_words)
            print('\n')

    text = "good hotel"
    average_vec = get_average_wordvec(text)

    words = most_similar_vector(average_vec, filter_words_list = ["good", "hotel"],number_of_words=30)
    most_similar_words = []
    for word in words:
        most_similar_words.append(word.lower_)

    print(most_similar_words)

    text = "bad hotel"
    average_vec = get_average_wordvec(text)

    words = most_similar_vector(average_vec, filter_words_list = ["bad", "hotel"],number_of_words=30)
    most_similar_words = []
    for word in words:
        most_similar_words.append(word.lower_)

    print(most_similar_words)