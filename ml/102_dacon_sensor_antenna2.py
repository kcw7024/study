import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import xgboost as xgb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)  # Seed 고정

path = './_data/dacon_sensor_antenna/'

train_df = pd.read_csv(path + 'train.csv')

train_x = train_df.filter(regex='X')
train_y = train_df.filter(regex='Y')

# plt.figure(figsize=(30, 5))
# plt.plot(train_x['X_01'], label='X_01')
# plt.legend()
# plt.show()

# plt.figure(figsize=(30, 5))
# plt.plot(train_x['X_02'], label='X_02')
# plt.legend()
# plt.show()

# plt.figure(figsize=(30, 5))
# plt.plot(train_x['X_03'], label='X_03')
# plt.legend()
# plt.show()

# plt.figure(figsize=(30,5))
# plt.plot(train_x['X_04'], label='X_04')
# plt.legend()
# plt.show()

# plt.figure(figsize=(30,5))
# plt.plot(train_x['X_05'], label='X_05')
# plt.legend()
# plt.show()

# XGB

# xgb = MultiOutputRegressor(xgb.XGBRegressor(
#     n_estimators=100, learning_rate=0.075,
#     gamma=0, subsample=0.75,
#     colsample_bytree=1, max_depth=7)
# ).fit(train_x, train_y)
# print('Done.')


# LinearRegressor
LR = MultiOutputRegressor(LinearRegression(

)
).fit(train_x, train_y)
print('Done.')


test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
preds = LR.predict(test_x)
print('Done.')

submit = pd.read_csv(path + 'sample_submission.csv')
for idx, col in enumerate(submit.columns):
    if col == 'ID':
        continue
    submit[col] = preds[:, idx-1]
print('Done.')

submit.to_csv(path + 'submit_lr.csv', index=False)
