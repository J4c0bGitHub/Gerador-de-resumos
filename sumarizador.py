# -*- coding: utf-8 -*-

import nltk
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

#Código configurado para funcionar com a tag P no jornal Deustche Welle Brasil

link = Request('https://www.dw.com/pt-br/novo-bloqueio-atinge-r-348-milh%C3%B5es-da-educa%C3%A7%C3%A3o/a-49823988')

pagina = urlopen(link).read().decode('utf-8', 'ignore')

#para conseguirmos as informações é importante sabermos como elas estão dispostas no site
#no site Ultimo Segundo as noticias estão em uma div com id = 'noticia'

soup = BeautifulSoup(pagina,"lxml")
texto = soup.find_all('p')

#Vamos agora importar algumas funcionalidades da biblioteca NLTK
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

normalizado = ""
#Vamos dividir o texto em sentenças e palavras
for parte in texto:
    trecho = parte.text
    normalizado = normalizado + trecho

sentencas = sent_tokenize(str(normalizado))
palavras = word_tokenize(str(normalizado).lower())

#Precisamos agora remover stopwords e pontuações do texto

from nltk.corpus import stopwords
from string import punctuation

#usaremos set, pois neste caso, não se repetem
stopwords = set(stopwords.words('portuguese') + list(punctuation))

#Apesar de confuso, a lógica é simples,
#apenas colocarei na lista se uma palavra estiver em palavras e não for uma stopword ou pontuação
palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]

#Agora criaremos uma Distribuição de Frequencia para listarmos as palavras mais importantes

from nltk.probability import FreqDist

frequencia = FreqDist(palavras_sem_stopwords)

#Vamos agora tabelar scores nas sentenças para o numero de retições de palavras importantes
#Para isso usaremos um dicionário especial da lib collections

from collections import defaultdict
sentencas_importantes = defaultdict(int)

#Agora vamos popular nosso dicionário percorrendo as palavras nas sentenças

for i, sentenca in enumerate(sentencas):
    for palavra in word_tokenize(sentenca.lower()):
        if palavra in frequencia:
            sentencas_importantes[i] += frequencia[palavra]
            
#Note que o código acima popula o dicionário com o índice da sentença (key)
#e a soma da frequência de cada palavra presente na sentença (value).

#Agora podemos selecionar as N palavras mais importantes
#para resumir o que queremos, agora, usaremos a funcionaldiade
#nlargest da lib heapq

from heapq import nlargest

idx_sentencas_importantes = nlargest(4,sentencas_importantes,sentencas_importantes.get)

for i in sorted(idx_sentencas_importantes):
    print(sentencas[i])