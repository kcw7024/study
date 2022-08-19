from sklearn.datasets import load_boston, load_iris, load_breast_cancer, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 

from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_digits()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1234    
)

Kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델

model = make_pipeline(StandardScaler(), 
                      LogisticRegression()
                      )

model.fit(x_train, y_train)

print("기본 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=Kfold, scoring='r2')
print("기본 CV : ", scores)
print("기본 CV 나눈 값 : ", np.mean(scores))


########################################### PolynomialFeature 후

#pf = PolynomialFeatures(degree=2, include_bias=False)
pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp.shape) # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, train_size=0.8, random_state=1234    
)

#2. 모델

model = make_pipeline(StandardScaler(), 
                      LogisticRegression()
                      )

model.fit(x_train, y_train)

print("폴리 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=Kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))

'''
(1797, 64) (1797,)
기본 스코어 :  0.9638888888888889
기본 CV :  [0.89345156 0.8892609  0.86292828 0.88881853 0.93343331]
기본 CV 나눈 값 :  0.8935785150147669

(1797, 2145)
폴리 스코어 :  0.9861111111111112
폴리 CV :  [0.95352208 0.9963087  0.94705018 0.96594811 0.94666396]
폴리 CV 나눈 값 :  0.9618986039475207

'''