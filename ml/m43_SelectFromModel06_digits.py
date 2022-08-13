from cgitb import reset
import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.feature_selection import SelectFromModel


#1. 데이터
datasets = load_digits()
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
          eval_metric = 'mlogloss',
          )

results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", r2)

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
    if (score*100) > 80.00 :
        print(select_x_train.shape, select_x_test.shape)
        print(select_x_test[0])
        print("Thresh=%f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
        
    
    
    
'''
최종 스코어 :  0.95
진짜 최종 test 점수 :  0.7990659568645108
[0.         0.00445988 0.01661914 0.006017   0.00568123 0.042983
 0.00713622 0.0085819  0.         0.01138893 0.02277877 0.00675452
 0.0112175  0.01069452 0.00442894 0.01301533 0.         0.00664086
 0.01238705 0.03576599 0.01574533 0.04260867 0.00399008 0.
 0.         0.01369596 0.02956568 0.01095368 0.03148233 0.0339727
 0.01699791 0.         0.         0.04642278 0.00324107 0.01229029
 0.06464192 0.01775245 0.02263567 0.         0.         0.00520551
 0.02987031 0.03734442 0.02138657 0.0134281  0.05332267 0.
 0.         0.00792074 0.00971168 0.0153794  0.00497166 0.03456596
 0.01718837 0.00850778 0.         0.00252259 0.02213941 0.0064396
 0.04181438 0.01644707 0.04247458 0.01681188]
================================================================================
(1437, 33) (360, 33)
Thresh=0.011218, n=33, R2: 80.90%
(1437, 29) (360, 29)
Thresh=0.013015, n=29, R2: 80.75%
(1437, 42) (360, 42)
Thresh=0.006641, n=42, R2: 80.12%
(1437, 30) (360, 30)
Thresh=0.012387, n=30, R2: 81.60%
(1437, 25) (360, 25)
Thresh=0.015745, n=25, R2: 80.86%
(1437, 14) (360, 14)
Thresh=0.029566, n=14, R2: 81.50%
(1437, 21) (360, 21)
Thresh=0.016998, n=21, R2: 80.30%
(1437, 31) (360, 31)
Thresh=0.012290, n=31, R2: 81.71%
(1437, 19) (360, 19)
Thresh=0.017752, n=19, R2: 84.00%
(1437, 18) (360, 18)
Thresh=0.021387, n=18, R2: 84.00%
(1437, 20) (360, 20)
Thresh=0.017188, n=20, R2: 80.75%
(1437, 17) (360, 17)
Thresh=0.022139, n=17, R2: 81.64%
(1437, 43) (360, 43)
Thresh=0.006440, n=43, R2: 80.12%
(1437, 22) (360, 22)
Thresh=0.016812, n=22, R2: 81.14%
'''    

# import xgboost as xg
# print(xg.__version__)    
    
    
    
    