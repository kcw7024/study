import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.svm import LinearSVC, LinearSVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score


# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',  # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)  # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv',  # 예측에서 쓸거임
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"]
                                == 1].value_counts(normalize=True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"]
                                == 2].value_counts(normalize=True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"]
                                == 3].value_counts(normalize=True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] ==
                               'female'].value_counts(normalize=True)[1]*100
male = train_set["Survived"][train_set["Sex"] ==
                             'male'].value_counts(normalize=True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)


# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)


# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)

train_set = train_set.drop(['Ticket'], axis=1)
test_set = test_set.drop(['Ticket'], axis=1)

train_set = train_set.drop(['Cabin'], axis=1)
test_set = test_set.drop(['Cabin'], axis=1)

train_set = pd.get_dummies(train_set, drop_first=True)
test_set = pd.get_dummies(test_set, drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape)  # (891, 8)

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal(
    [8, 1]), name='weight', dtype=float)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal(
    [1, 1]), name='bias', dtype=float)

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# hypothesis = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# model.add(Dense(1,activation='sigmoid',input_dim=2))
# 3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1E-2)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 5000
start_time = time.time()
for epochs in range(epoch):
    cost_val, h_val, _ = sess.run([loss, hypothesis, train],
                                  feed_dict={x: x_train, y: y_train})
    if epochs % 10 == 0:
        print(epochs, '\t', "loss :", cost_val, '\n', h_val)
        # print(epochs,'\t',"loss :",cost_val)

y_predict = sess.run(tf.cast(h_val >= 0.5, dtype=tf.float32))
# h_val =abs(h_val)
# h_val = np.round(h_val,0)
print(y_predict)
acc = accuracy_score(y_train, y_predict)
end_time = time.time()-start_time
print('acc :', acc)
print('걸린 시간 :', end_time)
sess.close()
