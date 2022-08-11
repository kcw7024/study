from statistics import quantiles
import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa)
#print(aaa[:,0])
#print(aaa[:,1])

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

# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)
# 이상치의 위치 :  (array([6], dtype=int64),)

outliers = EllipticEnvelope(contamination=.2) # .1 = 10%를 의미

outliers.fit(aaa, 0)
results1 = outliers.predict(aaa)

outliers.fit(aaa, 1)
results2 = outliers.predict(aaa)

print(results1, results2)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1] 
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]

