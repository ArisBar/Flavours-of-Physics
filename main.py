# Import lib and modules

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc;
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
import evaluation
import new_features

# Load training data
folder = 'data/'
train = pd.read_csv(folder + 'training.csv', index_col='id')
print('reading training data...')
train.head()

# Add new features
train = new_features(train)
#test = new_features(test)

# Here we place the features we will use for training
# (We remove SPDhits to pass the agreement test)
variables = list(train.columns) 
var_out = ['SPDhits', 'production', 'signal', 'mass', 'min_ANNmuon']
variables = [v for v in variables if v not in var_out]

# Baseline training...
baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
baseline.fit(train[variables], train['signal'])

# Check agreement test


# Check correlation test

# Compute weighted AUC on the training data with min_ANNmuon > 0.4¶
