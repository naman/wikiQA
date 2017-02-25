from porter_stemmer import PorterStemmer
import os
import re
import sys
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words


def tab_split(string):
    return string.strip().split("\t")


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


def search_doc(set, article_name):
    path = path_to_documents + set + "/question_answer_pairs.txt"
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        title = ' '.join(clean_words(special_split(words_in_line[0])))
        doc = words_in_line[5]
        # s = cosine_sim(title, article_name)
        # print s
        # if s > 0.1:
        if title == article_name:
            return doc
    return -1


def get_doc_name(article_name):
    a = search_doc("S08", article_name)

    if isinstance(a, basestring):
        return a
    b = search_doc("S09", article_name)

    if isinstance(b, basestring):
        return b

    c = search_doc("S10", article_name)
    if isinstance(c, basestring):
        return c


def parse_ground_truth_file(query, path):
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        q = words_in_line[1]
        a = words_in_line[2]

        if q == query:
            # exclude yes/no/null answers
            print a
            relevant_ans.append(a)
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
    # x = stopwords.words("english")
    x = get_stop_words("en")
    return [s.encode('ascii') for s in x] + list(string.printable)


def clean_split(string):
    return re.split('|'.join(map(re.escape, delimiters)), string.lower().strip())


def special_split(string):
    x = clean_split(string)
    return filter(lambda a: a != "", x)


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


def clean_words(array):
    cleaned_words = []
    for word in array:
        if (word is '') or (word in stop_words):
            continue
        else:
            word = porter.stem(word, 0, len(word) - 1)
            cleaned_words.append(word)
    return cleaned_words


def jaccard_similarity(article_name, focus_terms):
    scores = {}
    for x in file_sentences[article_name]:
        all_lists = [focus_terms]
        word_set = set(focus_terms)
        words_in_sentence = clean_words(special_split(x))
        word_set.update(words_in_sentence)
        all_lists.append(words_in_sentence)
        intersect = intersection(all_lists)

        length = float(len(word_set))
        scores[x] = len(intersect) / length

    print_scores(article_name, scores, "Jaccard")


def jaccard_sim(t1, t2):
    a = clean_words(special_split(t1))
    b = clean_words(special_split(t2))
    word_set = set(a)
    word_set.update(b)

    intersect = intersection([a, b])
    length = float(len(word_set))
    score = len(intersect) / length
    return score


cosine_offset = 0.4
# jaccard_offset = 0.4


def parse_answers(query, path):
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        a = words_in_line[2]
        cos_score = cosine_sim(query, a)
        if cos_score > cosine_offset:
            print "Cosine Similarity"
            print "\t[" + str(cos_score) + "]", "\t", a
            retrieved_ans.append(a)
        # jac_score = jaccard_sim(query, a)
        # if jac_score > jaccard_offset:
            # print "Jaccard Similarity"
            # print "\t[" + str(jac_score) + "]", "\t", a
    f.close()


relevant_ans = []
retrieved_ans = []


def print_formula():
    a = relevant_ans
    b = retrieved_ans
    intersect = intersection([a, b])

    try:
        precision = len(intersect) / float(len(b))
        recall = len(intersect) / float(len(a))
    except Exception:
        precision = 0.5
        recall = 1.0

    print "Precision:", precision
    print "Recall:", recall


def parse_answers_corpus(query):
    parse_answers(query, path_to_documents + "S08/question_answer_pairs.txt")
    parse_answers(query, path_to_documents + "S09/question_answer_pairs.txt")
    parse_answers(query, path_to_documents + "S10/question_answer_pairs.txt")
    ground_truth(query)
    print_formula()
    relevant_ans = []
    retrieved_ans = []


def run_query(query):
    parse_answers_corpus(query)


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


def normalize(t):
    x = stem_tokens(nltk.word_tokenize(
        t.lower().translate(remove_punctuation_map)))
    return x


porter = PorterStemmer()

os.system("clear")

print ".........................................................."
print "\t\tWelcome to WikiQA part 1!"
print "..........................................................\n"

print "Loading Stop Words..."
stop_words = load_stop_words()
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)
take_commands()
