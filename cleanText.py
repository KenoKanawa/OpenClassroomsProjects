from bs4 import BeautifulSoup
import re
import html as ihtml
import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
import contractions

# Mots du dictionnaire anglais
from nltk.corpus import words
dictionnaire = words.words()

# Ajout de quelques mots relatifs à l'informatique et à nots tags
filename = 'dicoProgramming.txt'
texte = []
with open(filename, 'r') as f:
    for line in f.readlines():
        texte.append(line.replace("\n", ""))
dictionnaireInfo = []
for i in range(len(texte)):
    dictionnaireInfo.append(texte[i].lower())

dictionnaireInfo2 = ['sqlite', 'sql', 'batch', 'shell', 'mysql',
    'scsi', 'xor', 'android', 'ios', 'ascii', 'asp', 'api', 'ipa', 'apk',
    'applet', 'assembler', 'bat', 'bcpl', 'cpl', 'css', 'csat', 'cvs',
    'datalog', 'html', 'php', 'dhtml', 'glitch', 'github', 'java',
    'javac', 'javascript', 'javafx', 'visualstudio', 'json', 'jupyter',
    'lua', 'matlab', 'microsoft', 'js', 'framework', 'scratch', 'snippet',
    'turing', 'xml', 'wml', 'xsl', 'tuple']
dictionnaire = dictionnaire + dictionnaireInfo
dictionnaire = list(set(dictionnaire))
dictionnaire.sort()

import gensim
from gensim.parsing.preprocessing import STOPWORDS as stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer

# Transformation des stopwords en liste
stopwords = list(stopwords)

# Regex qui ne conserve que les lettres
tokenizerLettres = RegexpTokenizer(r'[a-z]+')

# Alphabet sans la lettre c (langage de programmation)
alphabetSansC = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                 'v', 'w', 'x', 'y', 'z']

# Initialisation du lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialisation du stemmer
# stemmer = PorterStemmer()
stemmer = SnowballStemmer(language='english')

# Initialisation du correcteur
# spell = Speller(lang='en')


# Retrait des stopwords
def removeWords(text, words):
    
    # Tokenisation du texte
    text = nltk.word_tokenize(text)
    # Retrait des stopwords en passant par des sets
    text = list(set(text) - set(words))
    # Regroupement du textes
    text = " ".join(text)
    return text


# Conservation uniquement des mots du dictionnaire
def motsCorrects(text):
    
    # Tokenisation du texte
    text = nltk.word_tokenize(text)
    # On garde uniquement les mots du dictionnaire
    # text = list(set(text) & set(dictionnaire))
    text = [value for value in text if value in dictionnaire]
    # Regroupement du textes
    text = " ".join(text)
    return text


# Lemmatizaion des mots en fonction de leur type (nom, verbe, etc)
def lemmatize_text(text):
    
    # Découpage des mots en liste associé au tag du type de mot
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    
    # Vecteur du texte lemmatisé
    lemmatized_text = []
    
    # Boucle sur les mots et leurs tags
    for word, tag in tokens_tagged:
        # Adjectifs
        if tag.startswith('J'):
            # lemmatized_text.append(lemmatizer.lemmatize(word, 'a'))
            lemmatized_text.append('')
        # Adverbes
        elif tag.startswith('R'):
            # lemmatized_text.append(lemmatizer.lemmatize(word, 'r'))
            lemmatized_text.append('')
        # Verbes
        elif tag.startswith('V'):
            lemmatized_text.append(lemmatizer.lemmatize(word, 'v'))
        # Noms
        elif tag.startswith('N'):
            lemmatized_text.append(lemmatizer.lemmatize(word, 'n'))
        # Lematization simple pour le reste
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word))
    
    # Recollement du texte
    return " ".join(lemmatized_text)


# Stemming du texte
def stemming_text(text):

    text = nltk.word_tokenize(text)
    stem_text = []
    for word in text:
        stem_text.append(stemmer.stem(word))
    return " ".join(stem_text)


# Retrait des contractions et du slang
def suppressionContractions(text):

    # Tokenisation du texte
    text = contractions.fix(text)
    return text


# Fonction de nettoyage du texte qui regroupe plusieurs opérations
def clean_text(text):
    
    # Retrait du HTML par BeautifulSoup
    text = BeautifulSoup(ihtml.unescape(text)).text

    # Retrait des liens
    text = re.sub(r"http[s]?://\S+", "", text)

    # Mise en minuscule
    text = text.lower()

    # Retrait des contractions (doesn't) avant la ponctuation
    text = suppressionContractions(text)

    # Correction de l'orthographe (très couteux en ressources)
    # text = spell(text)
    
    # Retrait de la ponctuation
    text = tokenizerLettres.tokenize(text)
    text = " ".join(text)

    # Retrait des lettres seules sauf le C
    text = removeWords(text, alphabetSansC)

    # Retrait des stopwords
    text = removeWords(text, stopwords)

    # On ne garde que les mots de la langue anglaise
    text = motsCorrects(text)

    # Lemmatisation des mots
    text = lemmatize_text(text)

    # Stemming des mots
    # text = stemming_text(text)
    
    # Retrait des espaces en trop 
    text = re.sub(r"\s+", " ", text)
    
    # Retour
    return text