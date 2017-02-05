from porter_stemmer import PorterStemmer
import os
import re
import sys
import math


def load_stop_words():
    f = open(sys.argv[2])
    for word in f:
        stop_words.append(word.strip())


def add_to_dictionary(docname, term_frequency, word):
    if word not in dictionary.keys():  # or dictionary.keys()
        # print "Creating a new dictionary entry!"
        posting = {
            'name': docname,
            'frequency': 1,
            'tf_idf_weight': 0
        }

        dictionary[word] = [posting]  # creating postings list
    else:
        present = False
        for x in dictionary[word]:
            if x['name'] == docname:
                x['frequency'] += 1
                present = True
                break
            else:
                present = False

        if not present:
            # print "Creating a new dictionary entry!"
            posting = {
                'name': docname,
                'frequency': term_frequency,
                'tf_idf_weight': 0
            }
            dictionary[word].append(posting)


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def normalize():
    print "Calculating idf weights for each term(in all documents)!"
    idf_weights = {}
    N = 101  # 101 documents
    for term in dictionary:
        df = len(dictionary[term])
        idf = math.log10(N / float(df))
        idf_weights[term] = idf

    print "Calculating term frequencies for a term in each doc."
    for file_name in os.listdir(path_to_documents):
        if file_name.endswith(".txt"):
            doc_vector = []
            for term in dictionary:  # x is unknown term, we dont care
                for doc in dictionary[term]:
                    if doc['name'] == file_name:
                        frequency = doc['frequency']
                        doc_vector.append(doc['frequency'])
                        break
            # magnitude of doc_vector
            D = math.sqrt(sum([x**2 for x in doc_vector]))
            # scoring of tf-idf for a term in each doc
            for term in dictionary:
                for doc in dictionary[term]:
                    if doc['name'] == file_name:
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
    for file_name in os.listdir(path_to_documents):
        if file_name.endswith(".txt"):
            f = open(path_to_documents + file_name)
            term_frequency = 0
            for line in f:
                words_in_line = clean_split(line)

                if len(words_in_line) > 1:
                    for word in words_in_line:
                        if (word not in stop_words) and (not hasNumbers(word)) and (word is not ''):
                            word = porter.stem(word, 0, len(word) - 1)
                            term_frequency += 1
                            add_to_dictionary(file_name, term_frequency, word)
            # break
            f.close()


def clean_split(string):
    return re.split('|'.join(map(re.escape, delimiters)), string.lower().strip())


def loadDocuments():
    os.system(
        "wget -nd -r -P ./Documents -A txt,doc http://www.textfiles.com/computers/DOCUMENTATION/")


def write_inverted_index_to_file():
    f = open(sys.argv[1], 'w')
    f.write(str(dictionary))
    f.close()


def intersection(lists):
    try:
        intersected = set(lists[0]).intersection(*lists)
    except ValueError:
        intersected = set()  # empty
    return list(intersected)


def MultiWordQ(words_in_query):
    all_results = []

    for query in words_in_query:
        query = porter.stem(query, 0, len(query) - 1)
        results = []
        if query not in stop_words and query not in dictionary:
            print query, "word is not in any document."
        else:
            for x in dictionary[query]:
                results.append(x['name'])
        all_results.append(results)

    print intersection(all_results)


def show_results():
    return 0


def OneWordQ(query):
    # or dictionary.keys()
    query = porter.stem(query, 0, len(query) - 1)
    if query not in stop_words and query not in dictionary:
        # short cicruiting at its best in python :D
        print "Sorry! No results found!"
    else:
        rank = []
        for x in dictionary[query]:
            rank.append(x['tf_idf_weight'])
        print "Found", len(rank), "results. Sorted with relevance!"

        rank = sorted(rank)
        for x in xrange(len(rank)):
            for result in dictionary[query]:
                if rank[x] == result['tf_idf_weight']:
                    print "[" + str(x + 1) + "]", "in", result['name'], result['frequency'], "times. Score:[", result['tf_idf_weight'], "]"
                    break


def PhraseQ(words_in_query):
    return 1


def load_index_in_memory():
    f = open(sys.argv[1])

    f.close()


def run_query(query):
    words_in_query = clean_split(query)
    if '"' in query:
        PhraseQ(words_in_query)
    elif len(words_in_query) == 1:
        OneWordQ(query)
    else:
        MultiWordQ(words_in_query)


def take_commands():
    print "Please enter your query!"
    while 1:
        print "Enter: "
        query = raw_input().strip()
        run_query(query)


if len(sys.argv) < 4:
    print "USAGE: python search_engine.py <inverted_index> <stop_words> <path_to_docs>\n"
    print "PLEASE USE INVERTED INDEX IF YOU ALREADY HAVE IT."
    print "PLEASE USE STOP WORDS IF YOU ALREADY HAVE IT."
    print "PLEASE USE PATH TO DOCUMENTATION IF YOU ALREADY HAVE IT."

    exit(1)

path_to_documents = sys.argv[3]

dictionary = {
    'code': [{
        'name': 'abc.txt',  # primary_key
        'frequency': 1,
        'tf_idf_weight': 0
    }]  # postings_list
}

stop_words = []
delimiters = ['\n', ' ', ',', '.', '?', '!', ':', '#', '$', '[', ']',
              '(', ')', '-', '=', '@', '%', '&', '*', '_', '>', '<',
              '{', '}', '|', '/', '\\', '\'', '"', '\\x']

porter = PorterStemmer()

os.system("clear")

print ".........................................................."
print "\t\tWelcome to Go!"
print "..........................................................\n"

print "Do you want to update/build inverted index?[y/n]"
if raw_input() == 'y':
    # process
    # loadDocuments()
    print "Loading Stop Words..."
    load_stop_words()
    print "Building inverted index..."
    build_index()
    print "normalizing!"
    normalize()
    print "Writing the inverted index to", sys.argv[1]
    write_inverted_index_to_file()
    print "Data munching complete! Use Go now!\n"

    print "Complete!"

    take_commands()
else:
    print "Congrats! You just saved 15s in your life.\n"
    print "Loading inverted index in memory..."
    dictionary = load_index_in_memory()
    print "Loaded inverted index in memory!"

    take_commands()
