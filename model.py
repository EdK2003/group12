# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:20:08 2023

@author: 20183704
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('mol_desc_given.csv', index_col=0)

X = df[set(df.columns) - set(['ALDH1_inhibition'])]
y = df['ALDH1_inhibition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


pipeline = Pipeline(steps=[
    ('scaling', StandardScaler()),
    ('pca', PCA()),
    #('random_forest', RandomForestClassifier()),
    ('mlp', MLPClassifier())
])

parameters = {
    'pca_n_components': (5, 10, 20, 30, 50, 100), 
    #om_forest_n_estimators': (100, 500, 1000),
    'mlp_learning_rate_init': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    'mlp_max_iter': (100, 250, 500, 1000),
    'mlp_hidden_layer_sizes': ((100,), (1000,), (100,100), (1000,1000)),
    'mlp_alpha': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    'mlp_learning_rate': ('constant', 'invscaling', 'adaptive'),
    'mlp_momentum': (0.1, 0.5, 0.9),
    'mlp_early_stopping': (True, False),
    'mlp_beta_1': (1-1e-1, 1-1e-2, 1-1e-3, 1-1e-4, 1-1e-5),
    'mlp_beta_2': (1-1e-1, 1-1e-2, 1-1e-3, 1-1e-4, 1-1e-5),
    'mlp_validation_fraction': (0.1, 0.2, 0.5),
}
clf = GridSearchCV(pipeline, parameters, cv=5, scoring=balanced_accuracy_score)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

b_acc = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced accuracy: {b_acc}')

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')