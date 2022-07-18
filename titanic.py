import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic.csv', index_col='PassengerId')
df.drop('Ticket', inplace=True, axis=1)
for f in 'Sex', 'Cabin', 'Embarked':
    df[f] = df[f].astype('category')


train, test = train_test_split(df, shuffle=True)

train_rows = train
test_rows = test

test_label = test.Survived
train_label = train.Survived

for m in train_rows, test_rows:
    m.drop(['Survived', 'Name'], axis=1, inplace=True)

dtrain = xgb.DMatrix(train_rows, label=train_label, missing=np.NaN, enable_categorical=True)
dtest = xgb.DMatrix(test_rows, label=test_label, missing=np.NaN, enable_categorical=True)

clf = xgb.XGBClassifier(enable_categorical=True)

param = {
    'max_depth': 2,
    'eta': 1,
    'objective': 'binary:logistic',
    'nthread': 4,
    'eval_metric': 'auc'
}

evallist = [(dtest, 'eval'), (dtrain, 'train')]


num_round = 16

bst = xgb.train(param, dtrain, num_round, evals=evallist)

xgb.plot_importance(bst)

plt.show()
