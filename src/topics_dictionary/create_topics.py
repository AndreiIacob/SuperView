from nltk.corpus import wordnet

topic_dict = dict()

def read_from_file():
    file = open('topics.txt', 'r')
    for line in file:
        line = line.rstrip('\n')
        (key, words) = line.split(': ')
        topic_dict[key] = words.split(', ')
    print(topic_dict)

def get_synonims(dictionary):
    file = open('topics_generated.txt', 'w')
    generated_topics = list()
    for key in dictionary.keys():
        file.write(key + ': ')
        for synset in wordnet.synsets(key):
            for lemma in synset.lemmas():
                if lemma.name() not in dictionary[key]:
                    dictionary[key].append(lemma.name())
                    file.write(lemma.name() + ', ')
        for word in dictionary[key]:
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    if lemma.name() not in generated_topics:
                        generated_topics.append(lemma.name())
                        file.write(lemma.name() + ', ')
        file.write('\n')
        
if __name__ == '__main__':
    read_from_file()
    get_synonims(topic_dict)
