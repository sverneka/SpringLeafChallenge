#forked from Gilberto Titericz Junior

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# load training and test datasets
train = pd.read_csv('trainNoNan.csv')
train.drop(train.columns[0],inplace=True,axis=1)
#test = pd.read_csv('newtest.csv')
labels = train.target
print("train columns")
print(train.columns)

train.drop('target', axis=1, inplace=True)
print("train columns")
print(train.columns)

# convert data to numpy array
train = np.array(train)
#test = np.array(test)


# object array to float
train = train.astype(float)
labelTrue = np.where(labels == 1)[0]
trainTrue = train[labelTrue,:]
train = np.concatenate((train,trainTrue),axis=0)
labels = np.concatenate((labels, labels[labelTrue]),axis=0)
#test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
#label_log = np.log1p(labels)

params = {"objective": "binary:logistic",
          "eta": 0.02,# used to be 0.2 or 0.1
          "max_depth": 9, # used to be 5 or 6
          "min_child_weight": 1,
		  "max_delta_step": 6,
          "silent": 1,
          "colsample_bytree": 0.7,
		  "subsample": 0.8,
		  "eval_metric" : "auc",
          "seed": 1}
plst = list(params.items())
#Using 5000 rows for early stopping. 
#offset = 4000
    
num_rounds = 3000
#xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train, label=labels, missing= -99999)
#xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
#dtrain = xgb.DMatrix(train, label=labels)
xgb.cv(params, xgtrain, num_rounds, nfold=4)
