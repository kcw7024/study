import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv')