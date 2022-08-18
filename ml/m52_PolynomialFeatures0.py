import numpy as np 
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures



x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

print(x.shape) # (4, 2)

pf = PolynomialFeatures(degree=2)
# 증폭의 개념

x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# (4, 6)

'''
PolynomialFeatures
:: 단항을 다항으로 증폭시킨다. ::
메소드 degree 값에 따라 달라진다.
처음 1은 무조건 들어간다
두번째수는 자기자신의 수가 들어감
제곱식으로 증폭시킴
통상 degree = 2값정도를씀 최대 3
4이상은 잘 맞지 않는다.
'''



x = np.arange(12).reshape(4, 3)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

print(x.shape) # (4, 3)

pf = PolynomialFeatures(degree=2)
# 증폭의 개념

x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

############################################################

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

print(x.shape) # (4, 3)

pf = PolynomialFeatures(degree=3)
# 증폭의 개념

x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
