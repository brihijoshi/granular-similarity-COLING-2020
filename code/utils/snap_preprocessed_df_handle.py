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
from summa import keywords
from spacy.lang.en import English # updated



nlp = spacy.load('en_core_web_sm')

def get_dataframe(path):

	"""
	This method retrieves the dataframe and creates IDs for the dataframe
	"""


	df = pd.read_pickle(path)
	df['id1'] = df['Input.link1'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
	df['id2'] = df['Input.link2'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

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




