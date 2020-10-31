"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
from joblib import dump, load

class EstimatorSelectionHelper:

	def __init__(self, models, params):
		if not set(models.keys()).issubset(set(params.keys())):
			missing_params = list(set(models.keys()) - set(params.keys()))
			raise ValueError("Some estimators are missing parameters: %s" % missing_params)
		self.models = models
		self.params = params
		self.keys = models.keys()
		self.grid_searches = {}

	def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=True):
		for key in self.keys:
			print("Running GridSearchCV for %s." % key)
			model = self.models[key]
			params = self.params[key]
			gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
							  verbose=verbose, scoring=scoring, refit=refit,
							  return_train_score=True)
			gs.fit(X,y)
			self.grid_searches[key] = gs    
		
	def save_models(self,path, name):
		for k in self.grid_searches:
			best_estimator = self.grid_searches[k].best_estimator_
			dump(best_estimator, path+k+'_'+name+'.joblib') 
			
	def summary(self, X_test, y_test):
		for k in self.grid_searches:
			best_estimator = self.grid_searches[k].best_estimator_
			print("--------------------"+k+"--------------------")
			score = best_estimator.score(X_test.values.reshape(-1, 1), y_test)
			y_pred = best_estimator.predict(X_test.values.reshape(-1, 1))
			print('Accuracy - ', score)
			print("Classfication Report - ")
			print(classification_report(y_test, y_pred))
			print('Confusion Matrix - ')
			disp = plot_confusion_matrix(best_estimator, X_test.values.reshape(-1, 1), y_test,
				cmap=plt.cm.Blues, normalize='true')
			print(disp.confusion_matrix)
			plt.show()
			print("----------------------------------------")