from statistics import quantiles
import numpy as np
from collections import Counter
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

# def outliers(data_out) : 
#     quantiles_1, q2, quantiles_3 = np.percentile(data_out, [25, 50, 75]) #백분위계산을해서 변수에 담아줌 :: 데이터의 25%, 50%(중위값), 75%
#     print("1사분위 : ", quantiles_1)
#     print("q2 : ", q2) #중위값확인
#     print("3사분위 : ", quantiles_3)
#     iqr = quantiles_3 - quantiles_1 
#     print("iqr : ", iqr)
#     lower_bound = quantiles_1 - (iqr * 1.5) # 해당 공식으로 최소범위를 정해줌.
#     print(lower_bound) # -5.0
#     upper_bound = quantiles_3 + (iqr * 1.5) # 공식으로 최대범위를 정해줌.
#     print(upper_bound) # 19.0
#     return np.where(
#                     (data_out>upper_bound) | #최대값(이 이상은 이상치로 치겠다.)
#                     (data_out<lower_bound)   #최소값(이 이하는 이상치로 치겠다.)
#                     )
    
# outliers_loc = outliers(aaa)
# print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

'''
1사분위 :  4.0
q2 :  7.0
3사분위 :  10.0
iqr :  6.0
-5.0
19.0
이상치의 위치 :  (array([ 0, 12], dtype=int64),)

1사분위의 수! : 데이터의 25%가 이 값보다 작거나 같음.
2사분위의 수! : 중위수 데이터의 50%가 이 값보다 작거나 같음.
3사분위의 수! : 데이터의 75%가 이 값보다 작거나 같음.
(iqr)사분위간 범위 -> 제1분위수와 제3분위수간의 거리가 3분위 - 1분위 이므로 데이터의 중간 50%에 대한 범위

1.측정값들을 최소부터 최대까지 순서대로 나열
2.중앙값을 구한다. (데이터가 홀수일때는 중앙에 위치한값. 짝수일때는 중앙에있는 두 자료의 평균값)
3.1사분위는 자료의 중앙값기준으로 왼쪽 값들의 중앙값을 의미
4.3사분위는 자료의 중앙값기준으로 오른쪽값들의 중앙값을 의미
5.3사분위(q3) - 1사분위(q2)를 사용하여 사분위간 범위를 구한다.

'''

def outliers(df, n, features) : 
    outlier_list = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) |
                              (df[col] > Q3 + outlier_step)].index
        outlier_list.extend(outlier_list_col)
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)
    
    return multiple_outliers 
            
#outliers_loc = outliers(aaa)
#print("이상치의 위치 : ", outliers_loc)



