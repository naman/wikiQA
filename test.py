import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')  # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


print cosine_sim('How old was Lincoln in 1816?', 'Lincoln was just seven years old when, in 1816, the family was forced to make a new start in Perry County (now in Spencer County), Indiana.')
print cosine_sim('How old was Lincoln in 1816?', 'How old was Lincoln in ?')

