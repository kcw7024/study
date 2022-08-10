import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xg
print(xg.__version__) # 1.6.1

'''
01. iris
02. cancer
04. wine
05. fetch_covtype
06. digits
07. kaggle_titanic
'''



# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)
le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)
# x = pca.fit_transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
# stratify : 분류모델에서만 사용, 0 과 1을 확실하게 나눠주는 조건, < shuffle되어있기 때문에 
# stratify의 디폴트값은 None. < 회귀에서 쓸지 분류에서 쓸지 알수 없기때문에 알아서 적용해야함
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],
#       dtype=int64))
#lda = LinearDiscriminantAnalysis(n_components=6) # y라벨 개수보다 작아야만 한다
lda = LinearDiscriminantAnalysis() # y라벨 개수보다 작아야만 한다
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
#print(x_train.shape, y_train.shape)
#print(np.unique(y_train, return_counts=True))


# 2. 모델
from xgboost import XGBClassifier, XGBRegressor
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)


'''
PCA로 했을때 결과값 
XGBClassifier 의 결과값 :  1.0

LDA로 했을때 결과값
결과:  0.9
걸린 시간:  0.8326761722564697

'''