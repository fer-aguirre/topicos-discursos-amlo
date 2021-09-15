from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from pprint import pprint
#import pyLDAvis
#import pyLDAvis.gensim
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import pandas as pd
from matplotlib_venn_wordcloud import venn3_wordcloud

# Leer archivo 'txt'
file = open("data/manifiesto.txt", 'r').read()
file = file.strip('\n')
# Tokenizar archivo con expresiones regulares para evitar puntuación
sentences = sent_tokenize(file)
# Eliminar mayúsculas
sentences = [item.lower() for item in sentences]
# Definir stopwords en español
stop_words = stopwords.words('spanish')
# Agregar palabras a la lista de stopwords
#stop_words.extend(['word1', 'word2'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

# Tokeniza las oraciones
data_words = list(sent_to_words(sentences))
# Elimina stopwords
data_words = remove_stopwords(data_words)
# Crea diccionario
id2word = corpora.Dictionary(data_words)
# Crea corpus
texts = data_words
# Frecuencia de términos en el documento
corpus = [id2word.doc2bow(text) for text in texts]
# Preview
print(corpus)


# ENTRENAMIENTO MODELO LDA
# Número de tópicos
num_topics = 3
# Construye modelo LDA
lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
# Imprime las keywords en los 10 tópicos
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
# Preview
print(lda_model.print_topics(num_topics=3, num_words=5))


# Calcular perplejidad del modelo (entre más bajo el puntaje, mejor será el modelo)
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
# Puntaje de coherencia (la media de similitud de las palabras del tema)
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print("\nCoherence Score: ", coherence_lda)


# VISUALIZACIONES MODELO LDA
#pyLDAvis.enable_notebook()
#viz = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

# Wordcloud con los tópicos del modelo LDA
mask = np.array(Image.open('mask/mask.png'))
for t in range(lda_model.num_topics):
    plt.figure()
    wordcloud = WordCloud(background_color =  "Black", colormap = 'Oranges', min_font_size=20, width = 2000, height = 2000, mask = mask).fit_words(dict(lda_model.show_topic(t, 30)))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Topic #" + str(t))
    wordcloud.to_file(f'outputs/wordCloud_topic{t}.png')
    plt.show()

# Hacer una lista con 'n' palabras más repetidas por tópico
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t), topn = 5])

# Convertir en dataframe con 3 columnas: tópico, palabra y peso
df_topicos = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'Wight'])

# Guardar en un archivo 'csv'
#pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'Wight']).to_csv("data/topic_words.csv", index=False)
df_topicos = pd.read_csv('data/topic_words.csv')

# Función para obtener palabras por tópico
def getWords_perTopic(dataframe, topic):
    topic = dataframe['Word'].loc[dataframe['Topic'] == topic]
    topic = set(topic.to_list())
    return topic

topico1 = getWords_perTopic(df_topicos, 0)
topico2 = getWords_perTopic(df_topicos, 1)
topico3 = getWords_perTopic(df_topicos, 2)

# Color de texto
def color_func(*args, **kwargs):
    return "#000000"
# Diagrama de Venn con los tópicos
v = venn3_wordcloud(sets=(topico1, topico2, topico3), set_labels=("Tópico 1", "Tópico 2", "Tópico 3"), wordcloud_kwargs=dict(min_font_size=10, color_func=color_func), set_colors=['blue', 'red', '#8800CC'])
