from statistics import quantiles
import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa) # (13, 2)
print(aaa)
print(aaa[:,0]) #[-10   2   3   4   5   6   7   8   9  10  11  12  50]
print(aaa[:,1]) #[   100    200    -30    400    500    600 -70000    800    900   1000   210    420    350]


def outliers(data_out, i) : 
    quantiles_1, q2, quantiles_3 = np.percentile(data_out[:,i], [25, 50, 75]) 
    print("1사분위 : ", quantiles_1)
    print("q2 : ", q2) #중위값확인
    print("3사분위 : ", quantiles_3)
    iqr = quantiles_3 - quantiles_1 
    print("iqr : ", iqr)
    lower_bound = quantiles_1 - (iqr * 1.5) # 해당 공식으로 최소범위를 정해줌.
    print(lower_bound) # -5.0
    upper_bound = quantiles_3 + (iqr * 1.5) # 공식으로 최대범위를 정해줌.
    print(upper_bound) # 19.0
    return np.where(
                    (data_out[:,i]>upper_bound) | #최대값(이 이상은 이상치로 치겠다.)
                    (data_out[:,i]<lower_bound)   #최소값(이 이하는 이상치로 치겠다.)
                    )    
print("="*60)
outliers_loc1 = outliers(aaa, 0)
print("="*60)
outliers_loc2 = outliers(aaa, 1)
print("="*60)
print("이상치의 위치 : ", outliers_loc1)
print("이상치의 위치 : ", outliers_loc2)
print("="*60)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()





'''
============================================================
1사분위 :  4.0
q2 :  7.0
3사분위 :  10.0
iqr :  6.0
-5.0
19.0
============================================================
1사분위 :  200.0
q2 :  400.0
3사분위 :  600.0
iqr :  400.0
-400.0
1200.0
============================================================
이상치의 위치 :  (array([ 0, 12], dtype=int64),)
이상치의 위치 :  (array([6], dtype=int64),) 
============================================================

'''

