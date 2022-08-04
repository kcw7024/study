from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')
        
'''
AdaBoostClassifier 의 정답률 :  0.8044692737430168
BaggingClassifier 의 정답률 :  0.8659217877094972
BernoulliNB 의 정답률 :  0.7653631284916201
CalibratedClassifierCV 의 정답률 :  0.770949720670391
CategoricalNB 의 정답률 :  0.7653631284916201
ClassifierChain 은 실행되지 않는다.
ComplementNB 의 정답률 :  0.7653631284916201
DecisionTreeClassifier 의 정답률 :  0.8268156424581006
DummyClassifier 의 정답률 :  0.6480446927374302
ExtraTreeClassifier 의 정답률 :  0.8156424581005587
ExtraTreesClassifier 의 정답률 :  0.8212290502793296
GaussianNB 의 정답률 :  0.7821229050279329
GaussianProcessClassifier 의 정답률 :  0.8044692737430168
GradientBoostingClassifier 의 정답률 :  0.8491620111731844
HistGradientBoostingClassifier 의 정답률 :  0.8491620111731844
KNeighborsClassifier 의 정답률 :  0.8100558659217877
LabelPropagation 의 정답률 :  0.8044692737430168
LabelSpreading 의 정답률 :  0.8156424581005587
LinearDiscriminantAnalysis 의 정답률 :  0.776536312849162
LinearSVC 의 정답률 :  0.770949720670391
LogisticRegression 의 정답률 :  0.776536312849162
LogisticRegressionCV 의 정답률 :  0.770949720670391
MLPClassifier 의 정답률 :  0.8044692737430168
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 의 정답률 :  0.6312849162011173
NearestCentroid 의 정답률 :  0.7653631284916201
NuSVC 의 정답률 :  0.7653631284916201
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.7206703910614525
Perceptron 의 정답률 :  0.6089385474860335
QuadraticDiscriminantAnalysis 의 정답률 :  0.8268156424581006
RadiusNeighborsClassifier 의 정답률 :  0.7653631284916201
RandomForestClassifier 의 정답률 :  0.8491620111731844
RidgeClassifier 의 정답률 :  0.776536312849162
RidgeClassifierCV 의 정답률 :  0.776536312849162
SGDClassifier 의 정답률 :  0.8100558659217877
SVC 의 정답률 :  0.8044692737430168
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.


'''