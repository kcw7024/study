import pandas as pd
import numpy as np


data = pd.DataFrame([
                     [2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan ],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   NaN  NaN
# 1   NaN  4.0   4.0  4.0
# 2   NaN  NaN   NaN  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
print(data.shape) #(5, 4)

#결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
print("1. 결측치 삭제", "="*40)
print(data.dropna())
print(data.dropna(axis=0)) 
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1))
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

#2-1. 특정값 :: 평균
print("2-1. 결측치 처리 mean()", "="*40)
means = data.mean()
print("평균 : ", means)
data2 = data.fillna(means)
print(data2)
# 데이터가 틀어질 위험이 크다

#2-2. 특정값 :: 중위값
print("2-2. 결측치 처리 median()", "="*40)
median = data.median()
print("평균 : ", median)
data3 = data.fillna(median)
print(data3)
# 데이터가 틀어질 위험이 크다

#2-3. 특정값 :: ffill, bfill
print("2-3. 결측치 처리 ffill, bfill", "="*40)
data4 =  data.fillna(method='ffill')
print(data4)
'''
0   2.0  2.0   2.0  NaN
1   2.0  4.0   4.0  4.0
2   6.0  4.0   6.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0
'''
data5 = data.fillna(method='bfill')
print(data5)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  4.0
1   6.0  4.0   4.0  4.0
2   6.0  8.0   6.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
#2-4. 특정값 :: 임의값으로 채우기
print("2-4. 결측치 처리 : 임의값으로 채우기", "="*40)
#data6 = data.fillna(77777)
data6 = data.fillna(value=77777)
print(data6)

# 특정컬럼만 채우기.

means = data['x1'].mean()
print(means) # 6.5
data['x1'] = data['x1'].fillna(means)
print(data)

meds = data['x2'].median()
print(meds) # 4.0
data['x2'] = data['x2'].fillna(meds)
print(data)

data['x4'] = data['x4'].fillna(77)
print(data)

'''
6.5
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   6.5  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
4.0
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   6.5  4.0   4.0  4.0
2   6.0  4.0   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  NaN
     x1   x2    x3    x4
0   2.0  2.0   2.0  77.0
1   6.5  4.0   4.0   4.0
2   6.0  4.0   6.0  77.0
3   8.0  8.0   8.0   8.0
4  10.0  4.0  10.0  77.0

'''