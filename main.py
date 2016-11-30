# Import lib and modules

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc;
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import evaluation
from feature import new_feature



# Load training data
folder = 'data/'
train = pd.read_csv(folder + 'training.csv', index_col='id')
print('reading training data...')
train.head()


# Optional: Add new features
train = new_features(train)
#test = new_features(test)

# Here we place the features we will use for training
# (We remove SPDhits to pass the agreement test)
variables = list(train.columns) 
var_out = ['SPDhits', 'production', 'signal', 'mass', 'min_ANNmuon']
variables = [v for v in variables if v not in var_out]

# In case we want to eventually compare BostedTree and Neural network approaches 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train[variables], train['signal'], test_size=0.1, random_state=555)

 
kf = KFold(len(train), n_folds=5, random_state=555, shuffle=True)



# Baseline training...
baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
baseline.fit(X_train[variables], train['signal'])

# First attempts   

results=[]


from sklearn.ensemble import AdaBoostClassifier




for train_idx, test_idx in kf:
    X_train = train[variables].iloc[train_idx]
    y_train = train['signal'].iloc[train_idx]
    X_test = train[variables].iloc[test_idx]
    y_test = train['signal'].iloc[test_idx]
    from sklearn import svm
    #clf1 = DecisionTreeClassifier(random_state=0, max_depth=6)
    clf1 = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, max_depth=2), algorithm = "SAMME.R", 
                              n_estimators = 200)


    #clf2 = SVC(kernel='linear', C=1)


    #X_train, X_test, y_train, y_test = train_test_split(train[variables], train['signal'], test_size=0.4, random_state=0)

    clf1.fit(X_train, y_train)

    #X_test_eval = X_test[X_test['min_ANNmuon'] > 0.4]
    #y_test_eval = t_test[y_test['min_ANNmuon']>0.4]


    
    train_probs = clf1.predict_proba(X_test)[:, 1]
    AUC = evaluation.roc_auc_truncated(y_test, train_probs)
    print('AUC', AUC)
    results.append(AUC)


#print(train_probs)


#cross_val_score(clf1, train, train['signal'], cv=10)



# Check agreement test
evaluation.check_agreement(baseline)

# Check correlation test
evaluation.check_correlation(baseline)

# Compute weighted AUC on the training data with min_ANNmuon > 0.4¶
evaluation.compute_AUC(baseline, train)
