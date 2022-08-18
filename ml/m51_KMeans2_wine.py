import pandas as pd
from sklearn.datasets import load_iris, load_wine
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.cluster import KMeans

datasets = load_wine()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

print(df)

KMeans = KMeans(n_clusters=3, random_state=1234)

#PCA하고 클러스팅해주기도 함.
#n_clusters y라벨값기준


KMeans.fit(df)

print(KMeans.labels_)
# [0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 2 2 0 0 2 0 0 0 0 0 0 2 2
#  0 0 2 2 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 2 1 1 2 1 1 2 2 2 1 1 0
#  2 1 1 1 2 1 1 2 2 1 1 1 1 1 2 2 1 1 1 1 1 2 2 1 2 1 2 1 1 1 2 1 1 1 1 2 1
#  1 2 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 2 1 1 2 2 2 2 1 1 1 2 2 1 1 2 2 1 2
#  2 1 1 1 1 2 2 2 1 2 2 2 1 2 1 2 2 1 2 2 2 2 1 1 2 2 2 2 2 1]

print(datasets.target)

# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

df["cluster"] = KMeans.labels_

df["target"] = datasets.target

acc = accuracy_score(df["target"],df["cluster"])
print("acc :: ", acc)


# y 라벨을 생성해준다 :: KMeans

# acc ::  0.702247191011236

