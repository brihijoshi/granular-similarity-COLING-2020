"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""


from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import spacy
import os
from collections import OrderedDict
from itertools import combinations
import itertools
from spacy.lang.en import English # updated

nlp = spacy.load('en_core_web_md')

def get_dataframe(subset, data_home):

	"""
	This method retrieves the dataframe and creates IDs for the dataframe
	"""

	data = fetch_20newsgroups(data_home=data_home, subset=subset, categories=None,\
						   shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'),\
						   download_if_missing=True, return_X_y=False)
	print('Data fetched for subset - ', subset)
	df = pd.DataFrame({'article': data.data, 'label': data.target})
	df['id'] = df.index

	return df

def get_unique_combined_with_id(df, field_name, display_name):

	"""
	This method gets combined info for both id1 and id2
	"""

	fn1 = field_name+"1"
	fn2 = field_name+"2"

 
	articles = pd.concat([df[['id1',fn1]].rename(columns={'id1':'id',fn1:display_name}), df[['id2',fn2]].rename(columns={'id2':'id',fn2:display_name})]).drop_duplicates().reset_index(drop=True)
	id_dup = articles['id'].drop_duplicates().index

	articles = articles.loc[id_dup].reset_index(drop=True)

	return articles


def get_dataframe_group_level(subset, data_home):

	"""
	This method retrieves the dataframe from a group level and creates IDs for the dataframe
	"""

	data = fetch_20newsgroups(data_home=data_home, subset=subset, categories=None,\
						   shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'),\
						   download_if_missing=True, return_X_y=False)
	map_labels_to_groups = {0:5,1:0,2:0,3:0,4:0,5:0,6:3,7:1,8:1,9:1,10:1,11:2,12:2,13:2,14:2,15:5,16:4,17:4,18:4,19:5}
	print('Data fetched for subset - ', subset)
	df = pd.DataFrame({'article': data.data, 'label_class': data.target})
	df['label'] = df['label_class'].apply(lambda x: map_labels_to_groups[x])
	df['id'] = df.index

	# print(df.head(10))

	return df

def format_dataframe(df_with_keywords):

	"""
	This method and formats it in the pair-wise format (by stratifying samples) for the similarity task
	"""

	print('Started formatting')
	a = np.array(list(combinations(df_with_keywords['id'], 2)))
	print('Built combinations')
	group_wise_df = {}
	gb = df_with_keywords.groupby('label')
	locs_to_ignore = []
	for name, group in gb:
		group_ids = group['id']
		id_combinations = np.array(list(combinations(group_ids, 2)))
		locs_to_ignore.extend(id_combinations)
		group_df1 = group.loc[id_combinations[:,0]].reset_index(drop=True)
		group_df2 = group.loc[id_combinations[:,1]].reset_index(drop=True)
		combined_group_df = pd.DataFrame({'id1': group_df1.id, 'id2': group_df2.id, \
										  'article1': group_df1.article, 'article2': group_df2.article,\
										  'label1': group_df1.label, 'label2': group_df2.label})
		combined_group_df = combined_group_df.reset_index(drop=True)
		print('Built group DF for group - ', name)
		done = []
		locs = []
		for index, row in combined_group_df.iterrows():
			if row['id1'] not in done and row['id2'] not in done:
				done.append(row['id1'])
				done.append(row['id2'])
				locs.append(index)
		combined_group_df = combined_group_df.loc[locs].reset_index(drop=True)
		group_wise_df[name] = combined_group_df
	locs_to_ignore = np.array(locs_to_ignore)
	combined_df = group_wise_df.values()
	combined_df = pd.concat(combined_df).reset_index(drop=True)
	print('Equal topic df built')
	unequal_add = combined_df.shape[0]
	dims = np.maximum(a.max(0),locs_to_ignore.max(0))+1
	out = a[~np.in1d(np.ravel_multi_index(a.T,dims),np.ravel_multi_index(locs_to_ignore.T,dims))]
	sampled_indices = np.random.choice(np.arange(out.shape[0]), size=unequal_add)
	sampled_indices = out[sampled_indices]
	group_df1 = df_with_keywords.loc[sampled_indices[:,0]].reset_index(drop=True)
	group_df2 = df_with_keywords.loc[sampled_indices[:,1]].reset_index(drop=True)
	combined_uncommon_df = pd.DataFrame({'id1': group_df1.id, 'id2': group_df2.id, \
                                  'article1': group_df1.article, 'article2': group_df2.article,\
                                  'label1': group_df1.label, 'label2': group_df2.label})
	print('Unequal topic df built')

	return pd.concat([combined_df, combined_uncommon_df]).reset_index(drop=True)


if __name__ == '__main__':
	df_with_keywords = get_dataframe_group_level('train', data_home='../../data/newsgroups/')
	train_df = format_dataframe(df_with_keywords)
	train_df.to_pickle('../../data/dataframes/df_newsgroup_group_train.pkl')












