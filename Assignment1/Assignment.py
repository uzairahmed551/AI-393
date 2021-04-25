import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train.head()

test.head()

train.shape

test.shape

train.info()

test.info()

train.isna().sum()

for i in train.columns:
    if train[i].isna().sum() > 0:
        print(i)

test.isna().sum()

param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
              'fit_prior': [True,False]}

mnb_GS = GridSearchCV(mnb, param_grid, cv=5, verbose=2, n_jobs=1)

mnb_GS.fit(X_train, y_train)

mnb_GS_pred = mnb_GS.predict(X_test)

print(f"Accuracy: {round(metrics.accuracy_score(y_test, mnb_GS_pred)*100, 2)}%")

mnb_GS.best_params_

mnb_report = classification_report(y_true = y_test, y_pred = mnb_GS_pred)

print(mnb_report)

train['label'].value_counts(normalize=True)

plt.hist(train['label'], bins=20, alpha=0.7, color='#603c8e')
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(train[train.columns[1:785]], train[train.columns[0]]
                                                    , test_size=0.2)

# Predictor variables of training set
X_train.shape

# Dependent/outcome variable of training set
y_train.shape

# Predictor variables of validation set
X_test.shape

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

#Testing the model on validation set
mnb_pred = mnb.predict(X_test)

print(f"Accuracy: {round(metrics.accuracy_score(y_test, mnb_pred)*100, 2)}%")

target_names = list("0123456789")
mnb_report_2 = classification_report(y_true = y_test, y_pred = mnb_GS_pred
                                     , target_names=target_names, output_dict=True)
plt.figure(figsize=(10,10))
heat = sns.heatmap(pd.DataFrame(mnb_report_2).iloc[:-1, :].T, annot=True, xticklabels=True, yticklabels=True)
plt.title('Classification Report for NBC')
plt.show()


f1_score(y_test, mnb_pred, average='weighted')
