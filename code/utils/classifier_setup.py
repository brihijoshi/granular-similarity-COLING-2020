"""
Copyright (c) Snap Inc. 2020. This sample code is made available by Snap Inc. for informational purposes only. It is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, fitness for a particular purpose, or non-infringement. In no event will Snap Inc. be liable for any damages arising from the sample code or your use thereof.
"""


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# class_weights = compute_class_weight('balanced', [True, False], final['majority_same_event'])
models = {"Logistic Regression" : LogisticRegression(), 
           "Nearest Neighbors":KNeighborsClassifier() ,  
           "SVM": SVC(), 
           # "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0)),
           "Decision Tree" : DecisionTreeClassifier(), 
           "Random Forest": RandomForestClassifier(), 
           "Neural Net" : MLPClassifier(), 
           "AdaBoost" : AdaBoostClassifier(),
           "Naive Bayes" : GaussianNB() , 
           "QDA": QuadraticDiscriminantAnalysis()
}

params = {'Logistic Regression':{'penalty': ['l1', 'l2'], 'C': [1, 10], 'class_weight':[{True: w} for w in [1, 2, 5, 10, 50, 100]] },
           "Nearest Neighbors":{'n_neighbors':[3,5,8,10], 'algorithm':['ball_tree','kd_tree']},
           'SVM': [{'kernel': ['linear'], 'C': [1, 10], 'class_weight':[{True: w} for w in [1, 2, 5, 10, 50, 100]] },
                   {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001], 'class_weight':[{True: w} for w in [1, 2, 5, 10, 50, 100]]}],
           # 'Gaussian Process':{},
           "Decision Tree" : {"criterion": ["gini", "entropy"], 'class_weight':[{True: w} for w in [1, 2, 5, 10, 50, 100]] },
           'Random Forest': {'n_estimators': [16, 32], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                             'max_features': ['auto', 'sqrt'],
                             'min_samples_leaf': [1, 2, 4],
                             'min_samples_split': [2, 5, 10],
                            'class_weight':[{True: w} for w in [1, 2, 5, 10, 50, 100]] },
           "Neural Net" :{'activation': ['tanh', 'relu'],
                          'solver': ['sgd', 'adam'],
                          'alpha': [0.0001, 0.05],
                          'learning_rate': ['constant','adaptive']},
           'AdaBoost':  { 'n_estimators': [16, 32]},
           'Naive Bayes':{},
           'QDA':{}
}

# params = {'Logistic Regression':{'penalty': ['l1', 'l2'], 'C': [1, 10]},
#            "Nearest Neighbors":{'n_neighbors':[3,5,8,10], 'algorithm':['ball_tree','kd_tree']},
#            'SVM': [{'kernel': ['linear'], 'C': [1, 10]},
#                    {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}],
#            # 'Gaussian Process':{},
#            "Decision Tree" : {"criterion": ["gini", "entropy"]},
#            'Random Forest': {'n_estimators': [16, 32], 'max_depth': [10, 20, 50,80, 100, None],
#                              'max_features': ['auto', 'sqrt'],
#                              'min_samples_leaf': [1, 2, 4],
#                              'min_samples_split': [2, 5, 10]},
#            "Neural Net" :{'activation': ['tanh', 'relu'],
#                           'solver': ['sgd', 'adam'],
#                           'alpha': [0.0001, 0.05],
#                           'learning_rate': ['constant','adaptive']},
#            'AdaBoost':  { 'n_estimators': [16, 32]},
#            'Naive Bayes':{},
#            'QDA':{}
# }


def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=True)
