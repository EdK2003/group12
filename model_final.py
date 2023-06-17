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
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import pprint


df = pd.read_csv('mol_desc_given.csv', index_col=0)

X = df[list(set(df.columns) - set(['ALDH1_inhibition']))]
y = df['ALDH1_inhibition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline(steps=[
    ('scaling', StandardScaler()),
    ('pca', PCA()),
    ('mlp', MLPClassifier())
])

parameters = {
    # For optimization
    # 'pca__n_components': (0.5, 0.9, None),
    # 'mlp__learning_rate_init': (0.1, 0.01, 0.001, 0.0001),
    # 'mlp__max_iter': (100, 200, 400),
    # 'mlp__hidden_layer_sizes': ((200,), (200, 200),),
    # 'mlp__alpha': (1e-5, 1e-6, 1e-7),
    # 'mlp__momentum': (0.1, 0.9, 0.99),
    # 'mlp__beta_1': (0.9, 0.99),
    # 'mlp__beta_2': (0.999, 0.9999),
    # 'mlp__early_stopping': (True, False),
    # 'mlp__tol': (1e-2, 1e-4, 1e-5),
    # 'mlp__n_iter_no_change': (5, 10, 20),
    ###
    # Best below
    'pca__n_components': (0.9,),
    'mlp__learning_rate_init': (0.001,),
    'mlp__max_iter': (200,),
    'mlp__hidden_layer_sizes': ((200, 200),),
    'mlp__alpha': (1e-6,),
    'mlp__momentum': (0.9,),
    'mlp__beta_1': (0.9,),
    'mlp__beta_2': (0.999,),
    'mlp__early_stopping': (True,),
    'mlp__tol': (1e-3,),
    'mlp__n_iter_no_change': (10,),
}

clf = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    # Optimize for harmonic mean between precision and recall
    scoring=make_scorer(balanced_accuracy_score),
    n_jobs=8,
)


# Determine best performance
clf.fit(X_train, y_train)

# Print best parameters for reporting
pprint.pprint(clf.best_params_)

# Determine expected performance on unseen data
y_pred = clf.predict(X_test)

b_acc = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced accuracy: {b_acc}')

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')

# Balanced accuracy: 0.7553492006250752
# Precision: 0.6581196581196581
# Recall: 0.652542372881356

# Take the best estimator according to the grid search
clf = clf.best_estimator_

# Retrain on all data
clf.fit(X, y)

# Read unseen data and make predictions
X_unseen = pd.read_csv('mol_desc_given_untested.csv', index_col=0)

# Second column of predictions is the positive class
X_unseen['proba'] = clf.predict_proba(X_unseen[X_train.columns])[:, 1]
X_unseen.sort_values('proba', ascending=False, inplace=True)
X_unseen.head(100).index.to_frame().to_csv('top100.csv', index=False)

# Also manually inspect the associated probabilities
X_unseen.head(100).to_csv('top100_details.csv', index=False)
