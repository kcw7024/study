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


import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = x_train.data.shape[1] #features
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_train.feature_names)
    plt.xlabel('Feature Impotances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier

# model1 = DecisionTreeRegressor()
# model2 = RandomForestRegressor()
# model3 = GradientBoostingRegressor()
# model4 = XGBRegressor()

models = [DecisionTreeRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),XGBRegressor()]


for i in range(len(models)) :
    model = models[i]
    model.fit(x_train, y_train)
    plt.subplot(2, 2, i+1)
    plot_feature_importances(models[i])
    if models[i] == models[3] : 
        plt.title('XGBRegressor')
    else :
        plt.title(models[i])

# for model in use_models :
#     model = model()
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     print(name,'의 ACC : ', result)

#3. 훈련
# model1.fit(x_train, y_train)
# model2.fit(x_train, y_train)
# model3.fit(x_train, y_train)
# model4.fit(x_train, y_train)

# for model in models :
#     model = model()
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
    
#     result = model.score(x_test, y_test)
#     print(name,'의 ACC : ', result)

#4. 평가, 예측
# result = model.score(x_test, y_test)
# print("model.score : ", result)



from sklearn.metrics import accuracy_score, r2_score
#y_predict = model.predict(x_test)
#r2 = r2_score(y_test, y_predict)
#print('r2_score : ', r2)

#print("="*80)
#print(model1,':',model1.feature_importances_)

#plt.show()

# models = [model1,model2, model3, model4]

# for i in range(len(models)) :
#     plt.subplot(2, 2, i+1)
#     plot_feature_importances(models[i])
#     if models[i] == XGBRegressor : 
#         plt.title('XGBRegressor')
#     else :
#         plt.title(models[i])
plt.show()




'''

model.score :  0.046234925611890465
r2_score :  0.046234925611890465
================================================================================
DecisionTreeRegressor() : [0.07413783 0.01307606 0.34497272 0.08895758 0.02681448 0.10332579
 0.05579867 0.01331733 0.15841119 0.12118836]
 
model.score :  0.43909178187179143
r2_score :  0.43909178187179143
================================================================================
RandomForestRegressor() : [0.05664911 0.01281086 0.33767366 0.08644687 0.04574221 0.05556679
 0.06100216 0.03304795 0.22105261 0.09000776]

model.score :  0.4160185128926366
r2_score :  0.4160185128926366
================================================================================
GradientBoostingRegressor() : [0.04619565 0.01545641 0.33593563 0.09542872 0.03115416 0.06632719
 0.03859958 0.01412582 0.27768238 0.07909448]
 
model.score :  0.26078151031491137
r2_score :  0.26078151031491137
================================================================================
[0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191
 0.06551369 0.17944618 0.13779876 0.08540721]

 
'''


