# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import tensorflow as tf

# 1. 데이터
path = './_data/ddarung/'
# index_col = n : n번째 칼럼을 인덱스로 인식
train_set = pd.read_csv(path+'train.csv', index_col=0)
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum())  # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0)  # 결측치 0으로 채움
print(train_set.isnull().sum())  # 없어졌는지 재확인

# axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape)  # (1328, 9) (1328,)

y = np.array(y)
y = y.reshape(-1, 1)

# y = pd.get_dummies(y)

print(x.shape, y.shape)  # (1459, 9) (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b  # matmul :: 행렬곱 함수
# hypothesis = :: y의 shape값과 같아야한다.

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
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
# mae = mean_absolute_error(y_test, y_predict)
# print("mae :: ", mae)

sess.close()

# R2 ::  -0.2639427377908956
