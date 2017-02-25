# wikiQA: Search Engine
A basic search engine using jacard/cosine similarity and tf-idf score for Wikipedia Q/A. 
[Naman Gupta, 2013064]

# References

1. Porter stemmer from https://tartarus.org/~martin/PorterStemmer/
2. http://www.cs.cmu.edu/~ark/QA-data/

# Q1



# Q2

## Usage

`python search_engine_1.py <inverted_index> <stop_words> <path_to_docs>`
`python search_engine_2.py <inverted_index> <stop_words> <path_to_docs> <inverted_sentence_index>`

`python search_engine_1.py inverted_index.json stop_words.txt ./Documents/ file_sentences.json` part 1
`python search_engine_2.py inverted_index.json stop_words.txt ./Documents/ file_sentences.json` part 2

# Features Implemented

1. For part 1: Uses cosine and jacard similarity to fetch questions for the answers in answer corpus.

2. For part 2: It uses tf-idf to retrieve ranked results of possible documents which may contain the answer to the question. A faster method/alternative was also used which splits senteces (with the help of nltk) and matches the query with every sentence in every doc. The second approach is much more relevant to the the type of Search engine we are designing.
	EXAMPLE: We have a query, "Who did James Monroe marry?", Now, the tf-idf will output all the docs which contain these terms, the terms may be in a single sentence, or far apart. The only demerit of tf-idf scoring is that it doesn't consider how close the words are in the query. However, the sentence similarity scoring is much relevant in this case, because we get a finer list of results which contains these words closely.

3. For part 1 and 2: Precision/Recall is designed for binary relevance (relevant or not relevant). Therefore, NDCG is used to evaluate the search engine system where rank of the results is also considered. Since there is no way to determine the Ideal DCG (we don't know the ideal relevance of docs/answers), the ideal ranked answers/docs are assumed to be randomly shuffled and then ndcg is computed. Also, mean average precision, mean average recall and mean average ndcg is also calculated to evaluate the system (with 5 queries).

4. Cutoffs used for Jaccard and Cosine similarity is 0.4 +- 0.0001. 

5. Stores the inverted index (both inverted word index, and inverted sentence index) as a json allowing offline caching (saves precious time 7 minutes required to build inverted index, power and those CPU cycles on building the index again) (see `build_index()`, `load_index_in_memory()` and `write_to_file()`). The 
	The inverted index json stores as follows 
	
		term: {
			doc_name,
			frequency,
			score,
			positions[]
		}
	While, the sentence index json stores the following

		[
			doc_name: {
				sentences[]
			}
		]

6. For part 2: Logic for tf-idf ranked matching `MultiWordQ()` function: Search for the each query term in the inverted index. Take a union of results. Make a set. The output is the list of documents that contain any of the query terms. Sorted according to tf-idf scores/relevance. Prints the recall, precision and ndcg scores.

7. The query can be entered iteratively just like a normal shell. A prompt is visible where the query is entered and results are shown almost instantaneosly.

# Sample Queries for Q2, part 1
1. What are the similarities between beetles and grasshoppers?
2. What did Alessandro Volta invent in 1800?