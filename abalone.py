import numpy as np
import xgboost as xgb
import pandas as pd
from hyperopt import hp
from sklearn.model_selection import train_test_split

import hyper

df = pd.read_csv('abalone.csv')
df['Age'] = df.Rings + 1.5
labels = df['Age']
df.Sex = df.Sex.astype('category')
train_x, test_x, train_y, test_y = train_test_split(df, labels, shuffle=True)

for m in train_x, test_x:
    m.drop(['Age', 'Rings'], axis=1, inplace=True)

dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
dtest = xgb.DMatrix(test_x, label=test_y, enable_categorical=True)

optimizer = hyper.HyperOptTuner(
    dtrain, dtest, max_evals=200
)

trials = optimizer.optimize(
    {
        'num_boost_round': 2000,
        'eta': hp.quniform('eta', 0.025, 0.3, 0.025),
        'max_depth': hp.choice('max_depth', np.arange(1, 9, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, dtype=int)),
        'subsample': hp.quniform('subsample', 0.3, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.1, 20, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.25),
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror'
    }
)

print("Best Trial:", trials.best_trial)
