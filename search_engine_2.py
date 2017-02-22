from porter_stemmer import PorterStemmer
import os
import re
import sys
import math
import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer


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


def load_index_in_memory():
    with open(sys.argv[1]) as data_file:
        var = dict(json.load(data_file))
    return var


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


def print_scores(retrieved_docs, retrieved_docs_count, article_name, scores, similarity):
    if similarity == "Jaccard":
        cuttoff = 0.004
        threshold = 0.0001
    elif similarity == "Cosine":
        cuttoff = 0.1
        threshold = 0.0001

    max_score = max(scores[x] for x in scores)
    for x in scores:
        if scores[x] > cuttoff and math.fabs(max_score - scores[x]) < threshold:
            retrieved_docs_count += 1
            retrieved_docs[article_name] = True
            print "\nDocument:", article_name, get_doc_name(article_name)
            print "\t", similarity, "similarity"
            print "\t[" + str(scores[x]) + "]", "\t", x


def cosine_similarity(retrieved_docs, retrieved_docs_count, article_name, query):
    scores = {}
    for x in file_sentences[article_name]:
        score = cosine_sim(query, x)
        scores[x] = score
    print_scores(retrieved_docs, retrieved_docs_count,
                 article_name, scores, "Cosine")


def process_query(query):
    focus_terms = clean_words(special_split(query))
    ground_truth(query)

    relevant_docs = {doc: False for doc in file_sentences}
    retrieved_docs = {doc: False for doc in file_sentences}

    relevant_docs_count = 0
    retrieved_docs_count = 0
    for word in focus_terms:
        for doc in file_sentences:
            if word in doc:
                relevant_docs[doc] = True
                relevant_docs_count += 1
                break

    for doc in file_sentences:
        cosine_similarity(retrieved_docs, retrieved_docs_count, doc, query)
        # jaccard_similarity(doc, focus_terms)

    a = [doc for doc in relevant_docs if relevant_docs[doc] is True]
    b = [doc for doc in relevant_docs if retrieved_docs[doc] is True]
    intersect = intersection([a, b])
    precision = len(intersect) / float(len(b))
    recall = len(intersect) / float(len(a))

    print "Precision:", precision
    print "Recall:", recall


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


def normalize(t):
    x = stem_tokens(nltk.word_tokenize(
        t.lower().translate(remove_punctuation_map)))
    return x


porter = PorterStemmer()

os.system("clear")

print ".........................................................."
print "\t\tWelcome to WikiQA part 2!"
print "..........................................................\n"

print "Do you want to update/build inverted index?[y/n]"
if raw_input() == 'y':
    # part 2
    print "Loading Stop Words..."
    load_stop_words()
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)

    print "Building inverted sentence index..."
    build_sentence_index()

    print "Data munching complete! Use WikiQA now!"
    print "Writing the inverted index to", sys.argv[1]
    write_to_file(file_sentences, sys.argv[1])

    print "Complete!\n"
    # part 1

    take_commands()
else:
    print "Congrats! You just saved 10s in your life.\n"
    print "Loading Stop Words..."
    load_stop_words()
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)

    print "Loading inverted index in memory..."
    file_sentences = load_index_in_memory()
    if file_sentences == {}:
        print "error"
        exit(-1)
    print "Loaded inverted index in memory!"

take_commands()
