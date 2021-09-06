#!/usr/bin/env python
# coding: utf-8

# In[4]:


import codecs
import nltk
import numpy as np
import os
import pandas as pd
import re
import spacy

from gensim.models.phrases import Phraser, Phrases
from gensim.models import FastText
from gensim_models_procrustes_align import smart_procrustes_align_gensim, intersection_align_gensim
from joblib import Parallel, delayed  
from nltk.corpus import stopwords
from prepare_functions import lemmatize_text_column, sentence_to_wordlist, prepare_text
from scipy import spatial

nlp = spacy.load('de_core_news_md')
stopwords = stopwords.words('german')
tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')


# In[5]:


def lemmatize_text_column(df, column):
    """
    transformes the Dataframe-column in a lemmatized string
    """
    text = ''
    for i in df[column]:
        doc = nlp(i)
        lemmas = ' '.join([x.lemma_ for x in doc])
        text = text + lemmas
    return text


def sentence_to_wordlist(raw:str):
    """
    cleans and tokenizes the sentences
    """
    text = re.sub('[^A-Za-z_äÄöÖüÜß]',' ', raw).split()
    filtered_text = [word for word in text if word not in stopwords]
    return filtered_text


def prepare_text(raw_text):
    """
    return a list of tokenized sentences
    """
    raw_sentences = tokenizer.tokenize(str(raw_text).lower())    
    tokenized_sentences = Parallel(n_jobs=-1)(delayed(sentence_to_wordlist)(raw_sentence) for raw_sentence in raw_sentences)
    phrases = Phrases(tokenized_sentences)
    bigram = Phraser(phrases)
    sentences = list(bigram[tokenized_sentences])
    return sentences

