from porter_stemmer import PorterStemmer
import os
import re
import sys
import nltk
import string
import math
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words


def compute_avg(array, thing):
    avg_each = 0
    for x in array:
        avg_each += sum([y for y in x]) / float(len(x))
    mean_avg = avg_each / len(array)
    print "\tMean Average", thing, mean_avg


def dcg(array):
    if len(array) == 0:
        return []
    sum_dcg = 0
    dcg_array = []
    dcg_array.append(array[0])
    for i in xrange(1, len(array)):
        sum_dcg += array[i] / float(math.log(i + 1, 2))
        dcg_array.append(sum_dcg)
    return dcg_array


def ndcg(ranked, ideal_ranked):
    if len(ranked) == 0:
        return []
    if len(ideal_ranked) == 0:
        return []

    dcg_results = dcg(ranked)
    idcg_results = dcg(ideal_ranked)

    n = []
    for i in xrange(len(dcg_results)):
        if idcg_results[i] == 0:
            n[i].append(0)  # will never happen
        else:
            n.append(dcg_results[i] / float(idcg_results[i]))
    return n


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
    flag = True
    f = open(path)
    for line in f:
        words_in_line = tab_split(line)
        q = words_in_line[1]
        a = words_in_line[2]
        clean_ans = a.strip().lower().strip(".").strip("!")
        if q == query:
            # exclude yes/no/null answers
            print "\t", a
            relevant_ans.append(a)
            if clean_ans == "yes" or clean_ans == "no":
                # print clean_ans, a
                flag = False
                return flag
    f.close()
    return flag


def ground_truth(query):
    f1 = parse_ground_truth_file(query,
                                 path_to_documents + "S08/question_answer_pairs.txt")
    f2 = parse_ground_truth_file(query,
                                 path_to_documents + "S09/question_answer_pairs.txt")
    f3 = parse_ground_truth_file(query,
                                 path_to_documents + "S10/question_answer_pairs.txt")
    return all([f1, f2, f3])


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
            ranked.append(cos_score)
        # jac_score = jaccard_sim(query, a)
        # if jac_score > jaccard_offset:
            # print "Jaccard Similarity"
            # print "\t[" + str(jac_score) + "]", "\t", a
    f.close()


def write_to_file(text, path):
    with open(path, 'w') as outfile:
        json.dump(text, outfile, sort_keys=True, indent=4)


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

    r = sorted(ranked, reverse=True)
    ideal_ranked = r[:]
    random.shuffle(ideal_ranked)
    n = ndcg(r, ideal_ranked)

    print "\n"
    print "\tPrecision:", precision
    print "\tRecall:", recall
    print "\tNDCG:", n

    precisions.append([precision])
    recalls.append([recall])
    ndcgs.append(n)

    compute_avg(ndcgs, "NDCG")
    compute_avg(precisions, "Precision")
    compute_avg(recalls, "Recall")

    write_to_file(precisions, "precision_1.json")
    write_to_file(recalls, "recall_1.json")
    write_to_file(ndcgs, "ndcg_1.json")


def parse_answers_corpus(query):
    print "\n\tGround Truth"
    if ground_truth(query):
        parse_answers(query, path_to_documents +
                      "S08/question_answer_pairs.txt")
        parse_answers(query, path_to_documents +
                      "S09/question_answer_pairs.txt")
        parse_answers(query, path_to_documents +
                      "S10/question_answer_pairs.txt")
        print_formula()
        relevant_ans = []
        retrieved_ans = []


def run_query(query):
    parse_answers_corpus(query)


def load_index_in_memory(path):
    with open(path) as data_file:
        var = dict(json.load(data_file))
    return var


def take_commands():
    print "Please enter your query at the prompt!\n"
    while 1:
        sys.stdout.write("> ")
        query = raw_input().strip()
        run_query(query)


def stem_tokens(tokens):
    return [porter.stem(item, 0, len(item) - 1) for item in tokens]


def normalize(t):
    x = stem_tokens(nltk.word_tokenize(
        t.lower().translate(remove_punctuation_map)))
    return x


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
ndcgs = []
precisions = []
recalls = []
porter = PorterStemmer()
relevant_ans = []
retrieved_ans = []
ranked = []

os.system("clear")

print ".........................................................."
print "\t\tWelcome to WikiQA part 1!"
print "..........................................................\n"

print "Loading Stop Words..."
stop_words = load_stop_words()
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)
# ndcgs = list(load_index_in_memory("ndcg_1.json"))
# precisions = list(load_index_in_memory("precision_1.json"))
# recalls = list(load_index_in_memory("recall_1.json"))

take_commands()
