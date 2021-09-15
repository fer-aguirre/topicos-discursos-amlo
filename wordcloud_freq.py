from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Leer archivo 'txt'
file = open("data/manifiesto.txt").read()

# Tokenizar archivo con expresiones regulares para evitar puntuación
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(file)

# Eliminar mayúsculas
tokens = [item.lower() for item in tokens]

# Definir stopwords en español
stop_words = set(stopwords.words('spanish'))

# Filtrar stopwords
tokensLimpios = [item for item in tokens if item not in stop_words]

# Contar la frecuencia de los tokens
freqDist = FreqDist(tokensLimpios)

# Buscar (n) tokens más frecuentes
masFrecuentes = (freqDist.most_common(30))

# Tranformar en un diccionario
masFrecuentes = dict(masFrecuentes)

# Máscara para la wordcloud
mask = np.array(Image.open('mask/mask.png'))

# Generar wordcloud
wordcloud = WordCloud(collocations=False, background_color =  "Black", colormap = 'Blues', stopwords = stopwords, min_font_size=20, width = 2000, height = 2000, mask = mask, max_words = 100)
wordcloud.generate_from_frequencies(masFrecuentes)
plt.imshow(wordcloud, interpolation='bilinear')

# Guardar wordcloud
wordcloud.to_file('outputs/wordCloud_frecuencia.png')
plt.axis('off')
plt.show()
