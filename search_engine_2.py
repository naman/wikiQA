import os
import re
import sys
import math
import json
import nltk
import string
import fnmatch
from porter_stemmer import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def add_to_dictionary(docname, term_frequency, position, word):
    if word not in dictionary.keys():  # or dictionary.keys()
        # print "Creating a new initial dictionary entry!"
        posting = {
            'name': docname,
            'frequency': 1,
            'tf_idf_weight': 0,
            'positions': [position]
        }

        dictionary[word] = [posting]  # creating postings list
    else:
        present = False
        for x in dictionary[word]:
            if x['name'] == docname:
                x['frequency'] += 1
                x['positions'].append(position)
                present = True
                break
            else:
                present = False

        if not present:  # doc not present in the term index
            # print "Creating a new dictionary entry!"
            posting = {
                'name': docname,
                'frequency': term_frequency,
                'tf_idf_weight': 0,
                'positions': [position]
            }

            dictionary[word].append(posting)


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def get_files(path, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def normalize_index():
    print "Calculating idf weights for each term(in all documents)!"
    idf_weights = {}

    files = get_files(path_to_documents, "*.txt.clean")
    N = len(files)

    for term in dictionary:
        df = len(dictionary[term])
        idf = math.log10(N / float(df))
        idf_weights[term] = idf

    print "Calculating term frequencies for a term in each doc."

    for file_path in files:
        doc_vector = []
        for term in dictionary:  # x is unknown term, we dont care
            for doc in dictionary[term]:
                if doc['name'] == file_path:
                    frequency = doc['frequency']
                    doc_vector.append(doc['frequency'])
                    break
        # magnitude of doc_vector
        D = math.sqrt(sum([x**2 for x in doc_vector]))
        # scoring of tf-idf for a term in each doc
        for term in dictionary:
            for doc in dictionary[term]:
                if doc['name'] == file_path:
                    frequency = doc['frequency']
                    tf = frequency / float(D)

                    # tf = frequency
                    # w_tf = 0
                    # if tf > 0:
                    #     w_tf = 1 + math.log10(tf)
                    #
                    # idf is constant for all the docs a term appears in
                    idf = idf_weights[term]
                    # score for each term in doc
                    doc['tf_idf_weight'] = tf * idf
                    break


def build_index():
    files = get_files(path_to_documents, "*.txt.clean")
    for file_path in files:
        f = open(file_path)
        term_frequency = 0
        position = 0
        for line in f:
            words_in_line = special_split(line)

            for word in words_in_line:
                if (word in stop_words) or (hasNumbers(word)):
                    continue
                else:
                    word = porter.stem(word, 0, len(word) - 1)
                    position += 1
                    term_frequency += 1
                    add_to_dictionary(
                        file_path, term_frequency, position, word)
        # break
        f.close()


def MultiWordQ(words_in_query):
    all_results = []

    for query in words_in_query:
        query = porter.stem(query, 0, len(query) - 1)
        if (query in stop_words) or (hasNumbers(query)) or (query not in dictionary):
            print query, "word is not in any document."
            continue
        else:
            for x in dictionary[query]:
                all_results.append(x['name'])

    final_results = list(set(all_results))

    if len(final_results) == 0:
        print "Sorry! No results found!"

    for x in xrange(len(final_results)):
        print "[" + str(x + 1) + "]", "in", final_results[x]


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
                    print file_path
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
    x = stopwords.words("english")
    return [s.encode('ascii') for s in x]


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
        cuttoff = 0.4
        threshold = 0.0001
    elif similarity == "Cosine":
        cuttoff = 0.4
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
    # process_query(query)
    words_in_query = special_split(query)
    MultiWordQ(words_in_query)


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
delimiters = ['\n', ' ', ',', '.', '?', '!', ':', ';', '#', '$', '[', ']',
              '(', ')', '-', '=', '@', '%', '&', '*', '_', '>', '<',
              '{', '}', '|', '/', '\\', '\'', '"', '\t', '+', '~',
              '^', '\u']
dictionary = {
    'code': [{
        'name': 'abc.txt',  # primary_key
        'frequency': 1,
        'tf_idf_weight': 0,
        'positions': [1]
    }]  # postings_list
}
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
    stop_words = load_stop_words()
    # vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)

    # print "Building inverted sentence index..."
    # build_sentence_index()
    print "Building inverted index..."
    build_index()

    print "normalizing!"
    normalize_index()

    print "Writing the inverted index to", sys.argv[1]
    write_to_file(dictionary, sys.argv[1])

    print "Data munching complete! Use WikiQA now!"
    # print "Writing the inverted index to", sys.argv[1]
    # write_to_file(file_sentences, sys.argv[1])

    print "Complete!\n"

    take_commands()
else:
    print "Congrats! You just saved 5 minutes in your life.\n"
    print "Loading Stop Words..."
    stop_words = load_stop_words()
    # vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)

    print "Loading inverted index in memory..."
    # file_sentences = load_index_in_memory()
    dictionary = load_index_in_memory()
    # if file_sentences == {}:
    if dictionary == {}:
        print "error"
        exit(-1)
    print "Loaded inverted index in memory!"
    take_commands()