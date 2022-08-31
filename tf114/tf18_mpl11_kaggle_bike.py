from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf


# 1. 데이터
path = 'C:/study/_data/bike_sharing/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011: 0, 2012: 1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011: 0, 2012: 1})

train_set.drop('datetime', axis=1, inplace=True)  # train_set에서 데이트타임 드랍
test_set.drop('datetime', axis=1, inplace=True)  # test_set에서 데이트타임 드랍
train_set.drop('casual', axis=1, inplace=True)  # casul 드랍 이유 모르겠음
train_set.drop('registered', axis=1, inplace=True)  # registered 드랍 이유 모르겠음

# print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape)  # (10886, 14) (10886,)

x = np.array(x)

y = np.array(y)
y = y.reshape(-1, 1)

print(x.shape, y.shape)  # (10886, 12) (10886, 1)

# y = y.reshape(-1, 1)
# y = pd.get_dummies(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([12, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul :: 행렬곱 함수
# hypothesis = :: y의 shape값과 같아야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 3-1. 컴파일

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)

# 4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})

r2 = r2_score(y_test, y_predict)
print("R2 :: ", r2)

sess.close()


# R2 ::  0.31689872461556534
