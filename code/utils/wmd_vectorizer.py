"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import hashlib
import spacy
import string
from sklearn.metrics.pairwise import cosine_similarity
import gensim.models.keyedvectors as word2vec

import argparse
import wmd
from utils.snap_preprocessed_df_handle import *


nlp = spacy.load('en_core_web_md')
nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)


def get_vectors(articles, name):
	print('Getting vectors')
	articles['vector'] = articles.apply(lambda row: nlp(row[name]),axis=1)
	print('Done getting vectors')
	
	return  articles.drop(['article'],axis=1).set_index('id').to_dict('index')

def get_wmd_gensim(df, name):
	model = word2vec.KeyedVectors.load_word2vec_format('/home/brihi16142/news_representation_learning/data/pretrained/GoogleNews-vectors-negative300.bin', binary=True)
	print('Applying WMD similarity')
	name1 = name+"1"
	name2 = name+"2"
	df['wmd'] = df.apply(lambda x: model.wmdistance(x[name1],x[name2]),axis=1)
	print('Done applying WMD similarity')
	df['wmd_similarity'] = 1/(1+df['wmd'])
	return df

def get_wmd(df, articles, name):
	print('Entered get_wmd function')
	dict_articles = get_vectors(articles, name)

	df['v1'] = df.apply(lambda row: dict_articles[row['id1']]['vector'], axis=1)
	df['v2'] = df.apply(lambda row: dict_articles[row['id2']]['vector'], axis=1)

	print('Applying WMD similarity')

	df['wmd'] = df.apply(lambda x: x['v1'].similarity(x['v2']),axis=1)

	print('Done applying WMD similarity')

	df['wmd_similarity'] = 1/(1+df['wmd'])

	return df

if __name__ == "__main__":

	# PATH = '../data/dataframes/df_unique_with_similarity.pkl'

	# df_with_keywords = get_dataframe(PATH)

	# articles = get_unique_combined_with_id(df_with_keywords, 'Input.article', 'article')

	# df_with_keywords = get_wmd(df_with_keywords, articles)

	subset = 'train'

	df_with_keywords = pd.read_pickle('../data/dataframes/df_newsgroup_'+subset+'.pkl')
	articles = get_unique_combined_with_id(df_with_keywords, 'article', 'article')
	df_with_keywords = get_wmd(df_with_keywords, articles)



