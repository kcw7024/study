from statistics import quantiles
from unittest import result
import numpy as np


aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.1) # .1 = 10%를 의미


outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)