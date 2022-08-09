import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
#print(datasets.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets['target']

df = pd.DataFrame(x, columns=[['sepal length', 'sepal width', 'petal length', 'petal width']])
#print(df)
df['Target(Y)'] = y 
#print(df) #[150 rows x 5 columns]

print("="*30, "상관계수 히트 맵", "="*30)
print(df.corr()) 

#각 컬럼별 상관관계를 알수있음
# iris는 단순 linear 이기 때문에 정확하게 나옴

'''
============================== 상관계수 히트 맵 ==============================
             sepal length sepal width petal length petal width Target(Y)
sepal length     1.000000   -0.117570     0.871754    0.817941  0.782561
sepal width     -0.117570    1.000000    -0.428440   -0.366126 -0.426658
petal length     0.871754   -0.428440     1.000000    0.962865  0.949035
petal width      0.817941   -0.366126     0.962865    1.000000  0.956547
Target(Y)        0.782561   -0.426658     0.949035    0.956547  1.000000

'''
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()

'''
양의상관관계


'''