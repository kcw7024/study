from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.svm import LinearSVC, LinearSVR



#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
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

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41

for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        # model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('Model Name : ', name)
        print('ACC : ', scores) 
        print('cross_val_score : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        #print(y_predict)
        
        #y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')
        
'''
Model Name :  AdaBoostClassifier
ACC :  [0.83333333 0.84722222 0.71830986 0.84507042 0.85915493 0.8028169
 0.88732394 0.76056338 0.76056338 0.76056338]
cross_val_score :  0.8075
Model Name :  BaggingClassifier
ACC :  [0.80555556 0.72222222 0.73239437 0.83098592 0.83098592 0.78873239
 0.81690141 0.8028169  0.73239437 0.77464789]
cross_val_score :  0.7838
Model Name :  BernoulliNB
ACC :  [0.77777778 0.80555556 0.73239437 0.84507042 0.84507042 0.76056338
 0.84507042 0.71830986 0.77464789 0.76056338]
cross_val_score :  0.7865
Model Name :  CalibratedClassifierCV
ACC :  [0.79166667 0.83333333 0.73239437 0.84507042 0.85915493 0.74647887
 0.85915493 0.76056338 0.76056338 0.73239437]
cross_val_score :  0.7921
Model Name :  CategoricalNB
ACC :  [       nan 0.83333333 0.76056338 0.84507042 0.84507042 0.76056338
 0.84507042        nan 0.78873239 0.76056338]
cross_val_score :  nan
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
Model Name :  ComplementNB
ACC :  [0.76388889 0.80555556 0.74647887 0.83098592 0.83098592 0.73239437
 0.81690141 0.74647887 0.77464789 0.76056338]
cross_val_score :  0.7809
Model Name :  DecisionTreeClassifier
ACC :  [0.76388889 0.68055556 0.73239437 0.76056338 0.84507042 0.77464789
 0.76056338 0.76056338 0.71830986 0.77464789]
cross_val_score :  0.7571
Model Name :  DummyClassifier
ACC :  [0.52777778 0.56944444 0.67605634 0.6056338  0.56338028 0.67605634
 0.67605634 0.67605634 0.63380282 0.47887324]
cross_val_score :  0.6083
Model Name :  ExtraTreeClassifier
ACC :  [0.77777778 0.77777778 0.77464789 0.78873239 0.77464789 0.81690141
 0.77464789 0.66197183 0.63380282 0.77464789]
cross_val_score :  0.7556
Model Name :  ExtraTreesClassifier
ACC :  [0.76388889 0.83333333 0.74647887 0.8028169  0.81690141 0.84507042
 0.8028169  0.77464789 0.70422535 0.78873239]
cross_val_score :  0.7879
Model Name :  GaussianNB
ACC :  [0.79166667 0.81944444 0.71830986 0.83098592 0.81690141 0.77464789
 0.83098592 0.74647887 0.77464789 0.77464789]
cross_val_score :  0.7879
Model Name :  GaussianProcessClassifier
ACC :  [0.79166667 0.81944444 0.76056338 0.84507042 0.85915493 0.76056338
 0.85915493 0.78873239 0.78873239 0.73239437]
cross_val_score :  0.8005
Model Name :  GradientBoostingClassifier
ACC :  [0.80555556 0.875      0.78873239 0.87323944 0.85915493 0.76056338
 0.83098592 0.83098592 0.8028169  0.74647887]
cross_val_score :  0.8174
Model Name :  HistGradientBoostingClassifier
ACC :  [0.80555556 0.77777778 0.78873239 0.87323944 0.84507042 0.77464789
 0.85915493 0.77464789 0.77464789 0.77464789]
cross_val_score :  0.8048
Model Name :  KNeighborsClassifier
ACC :  [0.77777778 0.83333333 0.74647887 0.83098592 0.84507042 0.78873239
 0.84507042 0.77464789 0.74647887 0.71830986]
cross_val_score :  0.7907
Model Name :  LabelPropagation
ACC :  [0.79166667 0.88888889 0.73239437 0.83098592 0.85915493 0.78873239
 0.87323944 0.8028169  0.78873239 0.73239437]
cross_val_score :  0.8089
Model Name :  LabelSpreading
ACC :  [0.79166667 0.84722222 0.73239437 0.83098592 0.85915493 0.81690141
 0.88732394 0.8028169  0.78873239 0.73239437]
cross_val_score :  0.809
Model Name :  LinearDiscriminantAnalysis
ACC :  [0.79166667 0.84722222 0.73239437 0.85915493 0.85915493 0.76056338
 0.85915493 0.77464789 0.77464789 0.74647887]
cross_val_score :  0.8005
Model Name :  LinearSVC
ACC :  [0.80555556 0.84722222 0.73239437 0.84507042 0.85915493 0.76056338
 0.85915493 0.76056338 0.77464789 0.73239437]
cross_val_score :  0.7977
Model Name :  LogisticRegression
ACC :  [0.79166667 0.84722222 0.73239437 0.84507042 0.85915493 0.77464789
 0.85915493 0.76056338 0.77464789 0.73239437]
cross_val_score :  0.7977
Model Name :  LogisticRegressionCV
ACC :  [0.81944444 0.80555556 0.73239437 0.83098592 0.85915493 0.76056338
 0.84507042 0.76056338 0.77464789 0.71830986]
cross_val_score :  0.7907
Model Name :  MLPClassifier
ACC :  [0.79166667 0.81944444 0.73239437 0.84507042 0.83098592 0.77464789
 0.85915493 0.77464789 0.78873239 0.76056338]
cross_val_score :  0.7977
MultiOutputClassifier 은 실행되지 않는다.
Model Name :  MultinomialNB
ACC :  [0.58333333 0.63888889 0.77464789 0.69014085 0.61971831 0.70422535
 0.71830986 0.66197183 0.61971831 0.53521127]
cross_val_score :  0.6546
Model Name :  NearestCentroid
ACC :  [0.77777778 0.81944444 0.74647887 0.84507042 0.84507042 0.76056338
 0.84507042 0.74647887 0.77464789 0.76056338]
cross_val_score :  0.7921
Model Name :  NuSVC
ACC :  [0.79166667 0.81944444 0.74647887 0.84507042 0.84507042 0.76056338
 0.84507042 0.76056338 0.78873239 0.76056338]
cross_val_score :  0.7963
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
Model Name :  PassiveAggressiveClassifier
ACC :  [0.63888889 0.84722222 0.76056338 0.81690141 0.77464789 0.76056338
 0.87323944 0.78873239 0.74647887 0.69014085]
cross_val_score :  0.7697
Model Name :  Perceptron
ACC :  [0.76388889 0.86111111 0.6056338  0.46478873 0.69014085 0.69014085
 0.50704225 0.78873239 0.76056338 0.73239437]
cross_val_score :  0.6864
Model Name :  QuadraticDiscriminantAnalysis
ACC :  [0.80555556 0.83333333 0.76056338 0.83098592 0.83098592 0.77464789
 0.84507042 0.73239437 0.77464789 0.76056338]
cross_val_score :  0.7949
Model Name :  RadiusNeighborsClassifier
ACC :  [0.79166667 0.81944444 0.74647887 0.84507042 0.84507042 0.76056338
 0.84507042 0.74647887 0.78873239 0.76056338]
cross_val_score :  0.7949
Model Name :  RandomForestClassifier
ACC :  [0.77777778 0.81944444 0.76056338 0.84507042 0.84507042 0.83098592
 0.83098592 0.78873239 0.76056338 0.77464789]
cross_val_score :  0.8034
Model Name :  RidgeClassifier
ACC :  [0.79166667 0.81944444 0.74647887 0.85915493 0.85915493 0.74647887
 0.85915493 0.76056338 0.77464789 0.73239437]
cross_val_score :  0.7949
Model Name :  RidgeClassifierCV
ACC :  [0.79166667 0.81944444 0.74647887 0.85915493 0.85915493 0.74647887
 0.85915493 0.76056338 0.77464789 0.73239437]
cross_val_score :  0.7949
Model Name :  SGDClassifier
ACC :  [0.72222222 0.79166667 0.74647887 0.83098592 0.83098592 0.84507042
 0.81690141 0.76056338 0.70422535 0.76056338]
cross_val_score :  0.781
Model Name :  SVC
ACC :  [0.79166667 0.83333333 0.74647887 0.8028169  0.81690141 0.76056338
 0.85915493 0.78873239 0.77464789 0.74647887]
cross_val_score :  0.7921
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.

'''