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
