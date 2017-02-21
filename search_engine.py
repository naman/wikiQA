from porter_stemmer import PorterStemmer
import os
import re
import sys
import math
import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

reload(sys)
sys.setdefaultencoding('utf-8')


def tab_split(string):
    return string.strip().split("\t")


def parse_data_files(path):
    for folder in os.listdir(path):
        for file_name in os.listdir(path + folder):
            if file_name.endswith(".txt.clean"):
                file_path = path + folder + "/" + file_name
                f = open(file_path)
                text = f.read()
                f.close()
                try:
                    sentences = nltk.sent_tokenize(text)
                    file_sentences[
                        "data" + file_path.split("data")[1].split(".")[0]] = sentences
                except Exception:
                    pass


def clean_sentences():
    for k in file_sentences:
        new_sentences = []
        for sentence in file_sentences[k]:
            cleaned = special_split(sentence)
            new_sentences.append(' '.join(cleaned))
        file_sentences[k] = new_sentences


def get_article_name(path, file_path):
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        title = words_in_line[0]
        doc = words_in_line[5]
        if doc == file_path:
            return title


def parse_set(set):
    parse_data_files(path_to_documents + set + "/data/")
    clean_sentences()

    file_sentences_temp = {}
    # replace file_path with article name
    for x in file_sentences:
        tmp = get_article_name(path_to_documents + set +
                               "/question_answer_pairs.txt", x)
        if isinstance(tmp, basestring):
            y = ' '.join(clean_words(special_split(tmp)))
            file_sentences_temp[y] = file_sentences[x]
            file_sentences[x] = []

    keys_to_remove = [key for key, value in file_sentences.iteritems()
                      if value == []]
    for key in keys_to_remove:
        del file_sentences[key]

    file_sentences.update(file_sentences_temp)


def build_sentence_index():
    parse_set("S08")
    parse_set("S09")
    parse_set("S10")


def parse_ground_truth_file(query, path):
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        q = words_in_line[1]
        a = words_in_line[2]

        if q == query:
            # exclude yes/no/null answers
            print a
    f.close()


def ground_truth(query):
    print "\nGround Truth"
    parse_ground_truth_file(query,
                            path_to_documents + "S08/question_answer_pairs.txt")
    parse_ground_truth_file(query,
                            path_to_documents + "S09/question_answer_pairs.txt")
    parse_ground_truth_file(query,
                            path_to_documents + "S10/question_answer_pairs.txt")


def load_stop_words():
    f = open(sys.argv[2])
    for word in f:
        stop_words.append(word.strip())


def clean_split(string):
    return re.split('|'.join(map(re.escape, delimiters)), string.lower().strip())


def special_split(string):
    x = clean_split(string)
    return filter(lambda a: a != "", x)


def write_to_file(text, path):
    with open(path, 'w') as outfile:
        json.dump(text, outfile, sort_keys=True, indent=4)


def intersection(lists):
    try:
        intersected = set(lists[0]).intersection(*lists)
    except ValueError:
        intersected = set()  # empty
    return list(intersected)


def cosine_sim(a, b):
    # ref:
    # https://stackoverflow.com/questions/23792781/tf-idf-feature-weights-using-sklearn-feature-extraction-text-tfidfvectorizer#23796566
    corpus = [a, b]
    vector = vectorizer.fit_transform(corpus)
    vector_transform = vector.T
    magnitude = (vector * vector_transform).A  # a * a_transform
    return magnitude[0, 1]


def load_index_in_memory(var):
    with open(sys.argv[1]) as data_file:
        var = dict(json.load(data_file))


def clean_words(array):
    cleaned_words = []
    for word in array:
        if (word is '') or (word in stop_words):
            continue
        else:
            word = porter.stem(word, 0, len(word) - 1)
            cleaned_words.append(word)
    return cleaned_words


def jaccard_similarity(key, focus_terms):
    word_set = set(focus_terms)
    scores = {}
    all_lists = [focus_terms]
    for x in file_sentences[key]:
        words_in_sentence = clean_words(special_split(x))
        word_set.update(words_in_sentence)
        all_lists.append(words_in_sentence)
        intersect = intersection(all_lists)
        scores[x] = len(intersect)
        all_lists.remove(words_in_sentence)

    length = float(len(word_set))

    # normalize
    for x in scores:
        scores[x] /= length

    print_scores(scores, "Jaccard")


def print_scores(scores, similarity):
    print "\n", similarity, "similarity"
    max_score = max(scores[x] for x in scores)
    for x in scores:
        if math.fabs(max_score - scores[x]) < 0.0001:
            print "[" + str(scores[x]) + "]", "\t", x


def cosine_similarity(key, query):
    scores = {}
    for x in file_sentences[key]:
        score = cosine_sim(query, x)
        scores[x] = score
    print_scores(scores, "Cosine")


def process_query(query):
    focus_terms = clean_words(special_split(query))
    key = ""
    for word in focus_terms:
        # search the article first
        for x in file_sentences:
            if word in x:
                key = x
                cosine_similarity(key, query)
                jaccard_similarity(key, focus_terms)
                ground_truth(query)


def run_query(query):
    # focus_terms = clean_words(words_in_query) dont do this, removes numbers
    process_query(query)


def take_commands():
    print "Please enter your query at the prompt!\n"
    while 1:
        sys.stdout.write("> ")
        query = raw_input().strip()
        run_query(query)


if len(sys.argv) < 4:
    print "USAGE: python search_engine.py <stop_words> <path_to_docs>\n"
    print "PLEASE USE STOP WORDS IF YOU ALREADY HAVE IT."
    print "PLEASE USE PATH TO DOCUMENTATION IF YOU ALREADY HAVE IT."

    exit(1)

path_to_documents = sys.argv[3]
file_sentences = {}


FOCUS_DISTANCE = 2
stop_words = []
delimiters = ['\n', ' ', ',', '.', '?', '!', ':', '#', '$', '[', ']',
              '(', ')', '-', '=', '@', '%', '&', '*', '_', '>', '<',
              '{', '}', '|', '/', '\\', '\'', '"', '\t', '+', '~', ':',
              '^', '\u']

special_delimiters = ['\n', ' ', '\t', '\u']
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [porter.stem(item, 0, len(item) - 1) for item in tokens]


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


porter = PorterStemmer()

os.system("clear")

print ".........................................................."
print "\t\tWelcome to WikiQA!"
print "..........................................................\n"

print "Loading Stop Words..."
load_stop_words()

print "Building inverted sentence index..."
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)
build_sentence_index()

print "Data munching complete! Use WikiQA now!"

print "Complete!\n"

take_commands()
