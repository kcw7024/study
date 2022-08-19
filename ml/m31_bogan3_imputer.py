import pandas as pd
import numpy as np

data = pd.DataFrame([
                     [2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan ],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

#print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#imputer = SimpleImputer() # 평균값으로 결측치 처리가 된다.
#imputer = SimpleImputer(strategy='mean') # 평균값으로 결측치 처리가 된다.
#imputer = SimpleImputer(strategy='median') # 중의값으로 결측치 처리가 된다.
#imputer = SimpleImputer(strategy='most_frequent') # 가장 사용비중이많은 값으로 결측치 처리가 된다.
#imputer = SimpleImputer(strategy='constant', fill_value=525) # 상수를 넣어준다. 기본값이 0 임 
#imputer = KNNImputer() #최근 접 이웃을 사용하여 결 측값을 완성하기위한 대치
imputer = IterativeImputer() #Round robin 방식을 반복하여 결측 값을 회귀하는 방식으로 결측치를 처리. 

imputer.fit(data)
data2 = imputer.transform(data)
print(data2)

