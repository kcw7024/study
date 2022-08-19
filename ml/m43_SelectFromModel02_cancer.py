from cgitb import reset
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.feature_selection import SelectFromModel


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 

model = XGBClassifier(random_state=123, 
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                      gamma=1, )

#model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(
          x_train, y_train, early_stopping_rounds=200, 
          eval_set=[(x_train, y_train), (x_test, y_test)], # < [(훈련하고),(검증한다)]
          eval_metric = 'logloss',
          )

results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

print(model.feature_importances_)

# [0.03986917 0.04455113 0.25548902 0.07593288 0.04910125 0.04870857
#  0.06075545 0.05339111 0.30488744 0.06731401]

thresholds = model.feature_importances_
print("="*80)
for thresh in thresholds : 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # SelectFromModel :: 크거나 같은 컬럼을 다 뽑아준다
    select_x_train = selection.transform(x_train) 
    select_x_test = selection.transform(x_test) 
    selection_model = XGBClassifier(n_jobs=-1,
                                  random_state=123, 
                                  n_estimators=100,
                                  learning_rate=0.1,
                                  max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                                  gamma=1,  
                                  )
    
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    if (score*100) > 92.38 : #해당 점수 이상의 값만 출력하겠다.
        print(select_x_train.shape, select_x_test.shape)
        print("Thresh=%f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
    

   

    
    
'''
최종 스코어 :  0.9824561403508771
진짜 최종 test 점수 :  0.9824561403508771
[0.00847014 0.01691956 0.00783291 0.02105866 0.00828428 0.00585093
 /0.03084629/ 0.27008706 0.00795071 0.00598976 0.         0.
 0.00613586 0.01372907 0.         0.01098814 0.00882913 0.
 0.         0.         0.30543151 0.02641456 0.04537977 0.10479476
 0.01320044 0.0080536  0.02819064 0.04556232 0.         0.        ]        
================================================================================
(455, 15) (114, 15)
Thresh=0.008470, n=15, R2: 88.57%
(455, 10) (114, 10)
Thresh=0.016920, n=10, R2: 88.57%
(455, 19) (114, 19)
Thresh=0.007833, n=19, R2: 92.38%
(455, 9) (114, 9)
Thresh=0.021059, n=9, R2: 84.76%
(455, 16) (114, 16)
Thresh=0.008284, n=16, R2: 92.38%
(455, 22) (114, 22)
Thresh=0.005851, n=22, R2: 92.38%
(455, 6) (114, 6)
Thresh=0.030846, n=6, R2: 96.19%
(455, 2) (114, 2)
Thresh=0.270087, n=2, R2: 73.34%
(455, 18) (114, 18)
Thresh=0.007951, n=18, R2: 88.57%
(455, 21) (114, 21)
Thresh=0.005990, n=21, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 20) (114, 20)
Thresh=0.006136, n=20, R2: 92.38%
(455, 11) (114, 11)
Thresh=0.013729, n=11, R2: 84.76%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 13) (114, 13)
Thresh=0.010988, n=13, R2: 88.57%
(455, 14) (114, 14)
Thresh=0.008829, n=14, R2: 88.57%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 1) (114, 1)
Thresh=0.305432, n=1, R2: 69.53%
(455, 8) (114, 8)
Thresh=0.026415, n=8, R2: 84.76%
(455, 5) (114, 5)
Thresh=0.045380, n=5, R2: 92.38%
(455, 3) (114, 3)
Thresh=0.104795, n=3, R2: 80.96%
(455, 12) (114, 12)
Thresh=0.013200, n=12, R2: 92.38%
(455, 17) (114, 17)
Thresh=0.008054, n=17, R2: 92.38%
(455, 7) (114, 7)
Thresh=0.028191, n=7, R2: 88.57%
(455, 4) (114, 4)
Thresh=0.045562, n=4, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
(455, 30) (114, 30)
Thresh=0.000000, n=30, R2: 92.38%
'''    

# import xgboost as xg
# print(xg.__version__)    
    
    
    
    