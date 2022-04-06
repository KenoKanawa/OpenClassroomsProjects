import streamlit as st
import pickle
from cleanText import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Titre de l'application
st.header('Prédiction de sujets Stackoverflow')

# Entrée titre
title = st.text_input('Titre de la question :')

# Entrée question
body = st.text_area('Corps de la question :')

# Appui sur le bouton de prédiction
if st.button("Prédire"):

	# Concaténation des deux champs
	titleBody = title + " " + body

	# Traitement du texte
	titleBody = clean_text(titleBody)
	st.subheader('Texte après traitement')
	st.write(titleBody)

	# Récupération des tags
	N_TAGS = 6
	dataTags = pd.read_csv('TF-IDFTags_CUT.csv')
	tags = dataTags.columns.tolist()
	tags = tags[-N_TAGS:]
	i = 0
	for x in tags:
		tags[i] = tags[i].replace(".1", "")
		i = i + 1
	st.subheader('Liste des tags')

	# Reconstruction du corpus de mots du modèle
	data = pd.read_csv('dataRaw_CUT.csv')
	corpus_LIGNES = []
	for i, x in data.iterrows():
	    corpus_LIGNES.append(x['TitleBody'])

	# Vectorisation du CORPUS
	SEUIL_MIN = 2
	SEUIL_MAX = 0.5
	MAX_FEATURES = 1500
	vectorizer = TfidfVectorizer(min_df=SEUIL_MIN, max_df=SEUIL_MAX,
		max_features=MAX_FEATURES)
	vectors = vectorizer.fit_transform(corpus_LIGNES)
	feature_names = vectorizer.get_feature_names_out()
	dense = vectors.todense()
	denselist = dense.tolist()
	CORPUS = pd.DataFrame(denselist, columns=feature_names)

	# Vectorisation de nos entrées sur ce modèle
	dataNEW = vectorizer.transform([titleBody])

	# Import des modèles
	rf = pickle.load(open('CV_RF_MULTI.pkl', 'rb'))
	gb = pickle.load(open('CV_GB_MULTI.pkl', 'rb'))

	# Prédiction
	predictRF = rf.predict(dataNEW)
	predictGB = gb.predict(dataNEW)

	# Résultats RF
	st.write('Tags random forest :')
	i = 0
	j = 0
	for x in tags:
		if predictRF[0, i] == 1:
			st.write(x)
			j = j + 1
		i = i + 1

	# Si aucun tag de trouvé, message
	if j == 0:
		st.write('Aucun tag de trouvé pour cette question.')

	# Résultats GB
	st.write('Tags gradient boosting :')
	i = 0
	j = 0
	for x in tags:
		if predictGB[0, i] == 1:
			st.write(x)
			j = j + 1
		i = i + 1

	# Si aucun tag de trouvé, message
	if j == 0:
		st.write('Aucun tag de trouvé pour cette question.')