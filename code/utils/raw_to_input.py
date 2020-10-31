"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""

"""
This file takes in raw Mturk annotated data to extract relevant dataframes with fast similarity scores calculated
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import hashlib
import spacy
import os
from wordcloud import WordCloud
import json
from collections import OrderedDict
from operator import itemgetter
from spacy.lang.en.stop_words import STOP_WORDS
import string
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from itertools import combinations
from tqdm import tqdm_notebook
from bert_embedding import BertEmbedding
from summa import keywords


nlp = spacy.load('en_core_web_sm')

def get_annotated_df(l, path):
	df_list = []
	for file in l:
		if file!='test':
			df_list.append(pd.read_csv(path+file, keep_default_na=False))
	df = pd.concat(df_list, sort=False).reset_index(drop=True)
	return df[df['RequesterFeedback']==''].drop(\
													   list(df.columns[2:14])+\
													   list(df.columns[16:23])+\
													   list(df.columns[24:27])+\
													   list(df.columns[-2:]), axis=1).reset_index(drop=True)


def get_aggregated_df(df_approved):
	df_grouped_event = df_approved.groupby(by='HITId').sum()
	df_grouped_event.reset_index(level=0, inplace=True)
	df_grouped_event['majority_same_event'] = df_grouped_event['Answer.q31.yes'] >= df_grouped_event['Answer.q32.no']

	df_grouped_event['majority_topic_1'] = df_grouped_event[df_grouped_event.columns[3:10]].idxmax(axis=1).str.split(".").str.get(-1)

	print('Here')
	df_grouped_event['majority_topic_2'] = df_grouped_event[df_grouped_event.columns[10:17]].idxmax(axis=1).str.split(".").str.get(-1)


	"""
	Aggregating the majority consensus for topics and events
	"""
	return df_approved.merge(df_grouped_event[\
												  ['HITId', 'majority_topic_1', 'majority_topic_2', 'majority_same_event']],\
								 on='HITId', how='inner').reset_index(drop=True)

def get_vector_spacy(keywords):

	return nlp(keywords).vector	

def get_keywords(text):

	print('Entered')

	return keywords.keywords(text).split("\n")

def get_vector(keywords):

	return np.sum([get_vector_spacy(elem) for elem in keywords], axis=0)


def get_textrank_similarity(k1,k2):

	v1 = get_vector(k1)
	v2 = get_vector(k2)

	similarity = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

	print('Calculating Similarity')

	return similarity[0][0]


# For Building the test file

FILE_PATH = '../data/annotated/mturk_annotations/test/'
annotated = os.listdir(FILE_PATH)

print(annotated)

df_approved = get_annotated_df(annotated, FILE_PATH)

df_approved = get_aggregated_df(df_approved)

unique_inds = df_approved['HITId'].drop_duplicates().index

df_unique_rows = df_approved.loc[unique_inds].reset_index(drop=True)

# df_unique_rows = df_unique_rows.head(10)

df_unique_rows['k1'] = df_unique_rows['Input.article1'].apply(get_keywords)
print("Got First keywords")
df_unique_rows['k2'] = df_unique_rows['Input.article2'].apply(get_keywords)
print("Got Second keywords")


# df_unique_rows.read_pickle("../data/dataframes/df_unique_with_keywords.pkl")

df_unique_rows.to_pickle("../data/dataframes/df_test_unique_with_keywords.pkl")

print("Written Dataframe with keywords")

df_unique_rows['textrank_similarity'] = df_unique_rows[['k1','k2']].apply(lambda row : get_textrank_similarity(row['k1'],row['k2']), axis=1)


df_unique_rows.to_pickle("../data/dataframes/df_test_unique_with_similarity.pkl")

print("Written Dataframe with similarity")
