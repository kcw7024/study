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
#print(xg.__version__) # 1.6.1


#1.데이터
# datasets = load_iris() #(150, 4) -> (150, 2)
# datasets = load_breast_cancer() #(569, 30) -> (569, 1)
# datasets = load_wine() # (178, 13) -> (178, 2)
# datasets = fetch_covtype() # (581012, 54) -> (581012, 6)
datasets = load_digits() # (1797, 64) -> (1797, 9)

x = datasets.data
y = datasets.target
print(x.shape)
#print(np.unique)
lda = LinearDiscriminantAnalysis()
#lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x, y)
#print(np.unique(y, return_counts=True))
x = lda.transform(x)
print(x.shape) 


lda_EVR = lda.explained_variance_ratio_

print(sum(lda_EVR))
cumsum = np.cumsum(lda_EVR)
print(cumsum)


'''

LDA 

01. iris
0.9999999999999999
[0.9912126 1.       ]

02. cancer
1.0
[1.]

03. wine
1.0
[0.68747889 1.        ]

04. fetch_covtype
0.9999999999999999
[0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]   

05. digits

1.0
[0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
 0.94984789 0.9791736  1.        ]

'''
