# 다중공선성

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

datasets = fetch_california_housing()
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target
print(x.shape, y.shape)
# (20640, 8) (20640,)
# print(type(x)) # <class 'numpy.ndarray'>

x = pd.DataFrame(x, columns=datasets.feature_names)
print(x)


# 다중공선성
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(
    x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)

# drop_features = ['Longitude']
# drop_features = ['Longitude', 'AveRooms']
# drop_features = ['Longitude', 'AveRooms', 'Latitude']
drop_features = ['Longitude', 'Latitude']
x = x.drop(drop_features, axis=1)

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(
    x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
# 수치가 5이하가 좋다고함.(대신 컬럼이 너무 많을때는 10까지는 생각해볼수 있음)
print(vif)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

model = RandomForestRegressor(n_jobs=-1)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)

print('결과 : ', results)

'''

drop 안했을때
결과 :  0.8161750597940299

수치가 높은 컬럼부터 차례대로 삭제했을때

1개
결과 :  0.7338546508958719
2개
결과 :  0.730548128763727
3개 
결과 :  0.6589295204268708


drop_features = ['Longitude', 'Latitude']
결과 :  0.6848252393429657

'''

















