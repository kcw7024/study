from matplotlib.pyplot import sca
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

#model = SVC()
#model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 
#pipeline에서는 fit_transform이 적용된다.

model = Pipeline([
    ('minmax', MinMaxScaler()), 
    ('RF', RandomForestClassifier())
    ])


#3. 훈련

model.fit(x_train, y_train)
#pipeline fit이 포함되어있다.

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

# model.score :  1.0









