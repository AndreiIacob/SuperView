from SentimentAnalysis import *
import detectNLP
import json

def get_radical_ngrams(reviews, min_instances = 5, max_number_of_results_to_print = 100):
    ngram2rating = dict()
    for review in reviews:
        star = float(review["stars"])
        ngram = []
        if "words" not in review:
            review["words"] = process_words(review["text"], "en")
        for word in review["words"]:
            ngram.append(word)
            if len(ngram) == 3:
                ngramStr = " ".join(ngram)
                if ngramStr not in ngram2rating:
                    ngram2rating[ngramStr] = [0.0, 0]
                ngram2rating[ngramStr][0] += star
                ngram2rating[ngramStr][1] += 1
                ngram = ngram[1:]
            if len(ngram) == 2:
                ngramStr = " ".join(ngram)
                if ngramStr not in ngram2rating:
                    ngram2rating[ngramStr] = [0.0, 0]
                ngram2rating[ngramStr][0] += star
                ngram2rating[ngramStr][1] += 1
    word_score_instances = []
    for word in ngram2rating:
        ngram2rating[word][0] = ngram2rating[word][0] / ngram2rating[word][1]
        if ngram2rating[word][1] > min_instances:
            word_score_instances.append([word] + ngram2rating[word])

    word_score_instances.sort(key=lambda x: abs(x[1] - 3), reverse=True)
    for entry in word_score_instances[:max_number_of_results_to_print]:
        print(entry[0] + " - rating:" + str(entry[1]) + "  instances:" + str(entry[2]))
    return word_score_instances


class TddFilterPhrases(unittest.TestCase):
    @timing
    def test_filter_phrases(self):
        with open("test_input/filter_phrases_input.json","r") as f:
            word_ratings = json.load(f)


        filter_for_completeness(word_ratings)
        filter_words_by_sentiment(word_ratings)
        wordlist = ["room","food","service","location","room","clean"]
        filter_words_by_word2vec_distance(word_ratings,wordlist)
        self.assertNotIn(("horrible customer ", 1.0),word_ratings)
        self.assertNotIn(("http wwwyelpcombizphotosresducsfiiihpdg selectywgfbzxkffoykutq ", 5.0), word_ratings)
        self.assertNotIn(("easily best ", 4.863636363636363), word_ratings)
        self.assertNotIn(("yum yum ", 4.857142857142857), word_ratings)
        self.assertIn(("horrible customer service ", 1.0), word_ratings)
        self.assertIn(("italian dessert ", 5.0), word_ratings)
        print(word_ratings)

def filter_words_by_sentiment(word_ratings, min_difference = 0.2):
    i = 0
    while i<len(word_ratings):
        word_rating = word_ratings[i]
        text = word_rating[0]
        rating = word_rating[1]
        delete_this = True
        for word in text.split():
            sentiment_value = get_senitment_textblob(word) * 4 + 1
            if abs(rating/sentiment_value-1.0) >= min_difference:
                delete_this = False
        if delete_this:
            del word_ratings[i]
        else:
            i += 1

@timing
def filter_words_by_word2vec_distance(word_ratings, word_list, min_distance_one = 0.5, min_distance_all = 0, max_distance = 1):
    cosine = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    i = 0
    while i<len(word_ratings):
        word_rating = word_ratings[i]
        text = word_rating[0]
        wordvec = detectNLP.get_average_wordvec(text)

        delete_this = False

        min_value = 1
        max_value = 0
        for word in word_list:
            filter_wordvec = detectNLP.get_average_wordvec(word)
            distance = cosine(wordvec,filter_wordvec)
            if distance > max_value:
                max_value = distance
                if max_distance < max_value:
                    delete_this = True
                    break

            if distance < min_value:
                min_value = distance
                if min_value < min_distance_all:
                    delete_this = True
                    break

        if max_value < min_distance_one:
            delete_this = True

        if delete_this:
            del word_ratings[i]
        else:
            i += 1

def filter_for_completeness(word_ratings):
    phrases = []
    for word_rating in word_ratings:
        phrases.append(word_rating[0])

    i = 0
    while i<len(word_ratings):
        word_rating = word_ratings[i]
        text = word_rating[0]
        delete_this = False
        for phrase in phrases:
            if text!=phrase:
                if text in phrase:
                    delete_this = True
                    break
        if delete_this:
            del word_ratings[i]
        else:
            i += 1

if __name__ == '__main__':
    unittest.main()