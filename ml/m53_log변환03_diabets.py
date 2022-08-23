from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer #scaling 
# :: QuantileTransformer, RobustScaler ->이상치에 자유로움
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123   
)

# scaler = StandardScaler()
# x_trian = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("기본 결과 : ", round(results,4))


################### 로그 변환!!! ######################
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
#print(df)

# df.plot.box()
# plt.title("boston")
# plt.xlabel('Feature')
# plt.ylabel('data values')
# plt.show()

#print(df['B'].head())
#print(df['B'].head())

# 로그변환
df['bmi'] = np.log1p(df['bmi'])
df['s1'] = np.log1p(df['s1']) # log1p :: +1 (log0일때의 에러때문에) 지수변환은 exp1p
#df['s2'] = np.log1p(df['s2'])
#df['s3'] = np.log1p(df['s3'])
#df['s4'] = np.log1p(df['s4'])
#df['s6'] = np.log1p(df['s6'])


x_train, x_test, y_train, y_test = train_test_split(
    df, y, train_size=0.8, random_state=123   
)

# scaler = StandardScaler()
# x_trian = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("LOG 변환 후 결과 : ", round(results,4))

'''
* 6개 컬럼 모두 변환 했을때
1.LogisticRegression 
기본 결과 :  0.5676
LOG 변환 후 결과 :  0.5686

2.RandomForestClassifier 
기본 결과 :  0.5376
LOG 변환 후 결과 :  0.5428

* 1개의 컬럼만 로그변환 한뒤에 실행했을경우와 
2개의 컬럼만 로그변환 한뒤에 실행했을경우에
스코어가 계속 떨어졌다.
 
'''