import pprint
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim import models
from gensim import similarities

# Leer archivo 'txt'
file = open("data/manifiesto.txt", 'r').read()

# Tokenizar archivo con expresiones regulares para evitar puntuación
documents = sent_tokenize(file)
documents
# Convert a document into a list of lowercase tokens and remove accents marks
def preprocess_doc(corpus):
    for document in corpus:
        yield(gensim.utils.simple_preprocess(str(document), deacc=True))

text_corpus = list(preprocess_doc(documents))

# Definir stopwords en español
stop_words = stopwords.words('spanish')

# Filter out stopwords
texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in text_corpus]

# Count frequency of tokens
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

# Associate each word in the corpus with a unique integer ID, which means that each document will be represented by a n-dimensional vector
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
pprint.pprint(dictionary.token2id)

# Convert the entire original corpus to a list of vectors
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

# Train the model
tfidf = models.TfidfModel(bow_corpus)
# Transform the string and weight it
words = "pueblos crisis planeta futuro".lower().split()
print(tfidf[dictionary.doc2bow(words)])

# Transform the whole corpus via tfIdf and index it
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=48)

# Similarity of a query document against every document in the corpus
query_document = 'revertir crisis climática'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))

for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
