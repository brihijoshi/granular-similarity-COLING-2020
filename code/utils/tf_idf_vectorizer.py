"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import hashlib
import spacy
import os
import re
import json
from collections import OrderedDict
from operator import itemgetter
from spacy.lang.en.stop_words import STOP_WORDS
import string
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from itertools import combinations
from spacy.lang.en import English # updated
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from utils.snap_preprocessed_df_handle import *



nlp = spacy.load('en_core_web_sm')
stop_list = list(STOP_WORDS).extend(['d','ll','m','re','s','ve'])


def preprocessor(text):
    regex = r'(?<!\d)[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~](?!\d)'
    return re.sub(regex, "", text, 0).lower()


def get_ordered_dict_from_df(df, display_name):
	df = dict(zip(df.id, df[display_name]))
	return OrderedDict(df)

def get_tf_idf(articles, display_name, preprocessor=preprocessor, stop_words=stop_list, ngram_range = (1,1)):
	ordered_dict_articles = get_ordered_dict_from_df(articles, display_name)
	od_vals = np.array(list(ordered_dict_articles.values()))
	od_keys = np.array(list(ordered_dict_articles.keys()))


	vectorizer = TfidfVectorizer(strip_accents='ascii',analyzer='word', stop_words=stop_words,\
                             lowercase=True, preprocessor=preprocessor, ngram_range=ngram_range,\
                             norm='l2',use_idf=True, smooth_idf=True)

	od_output = vectorizer.fit_transform(od_vals)

	return od_output, od_keys

if __name__ == "__main__":

	PATH = '../data/dataframes/df_unique_with_similarity.pkl'

	df_with_keywords = get_dataframe(PATH)

	articles = get_unique_combined_with_id(df_with_keywords, 'Input.article', 'article')

	od_output, od_keys = get_tf_idf(articles, 'article', preprocessor=preprocessor, stop_words=stop_list, ngram_range = (1,1))




