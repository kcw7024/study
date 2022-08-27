from sklearn.ensemble import ExtraTreesClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from torch import rand
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')




#### Fe Male 수정 
train_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')
test_set['Gender'] = test_set['Gender'].str.replace('Fe Male', 'Female')
# print(test_set.head(30))


# 1-1. 결측치 처리 (평균값으로 채워준다)
mean_cols = ['Age', 'NumberOfFollowups', 'PreferredPropertyStar',
             'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'DurationOfPitch']
for col in mean_cols:
    train_set[col] = train_set[col].fillna(train_set[col].mean())

mean_cols = ['Age', 'NumberOfFollowups', 'PreferredPropertyStar',
             'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'DurationOfPitch']
for col in mean_cols:
    test_set[col] = test_set[col].fillna(test_set[col].mean())

# 1-2. 문자형 결측치 처리 (Unknwon으로 채워준다)
train_set.TypeofContact = train_set.TypeofContact.fillna("Unknown")
print(train_set.isnull().sum())
# 확인 완료
# object type 컬럼 확인하기.
object_columns = train_set.columns[train_set.dtypes == 'object']
print('list : ', list(object_columns))
# list :  ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

test_set.TypeofContact = test_set.TypeofContact.fillna("Unknown")
print(test_set.isnull().sum())
# 확인 완료
# object type 컬럼 확인하기.
object_columns = test_set.columns[test_set.dtypes == 'object']
print('list : ', list(object_columns))
# list :  ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

# object 를 라벨인코더 처리 해줌
train_set = train_set.copy()
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_set[o_col])
    train_set[o_col] = encoder.transform(train_set[o_col])

print(train_set)

test_set = test_set.copy()

for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(test_set[o_col])
    test_set[o_col] = encoder.transform(test_set[o_col])

print(test_set)

# 원핫, feature important 확인,

scaler = MinMaxScaler()

train_set = train_set.copy()
test_set = test_set.copy()
scaler.fit(train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(
    train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
scaler.fit(test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(
    test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
print(train_set)
print(test_set)

# 불필요한 컬럼 날림

train_set = train_set.drop(columns=['NumberOfTrips', 'TypeofContact',
                           'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome'])
test_set = test_set.drop(columns=['NumberOfTrips', 'TypeofContact',
                         'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome'])

x = train_set.drop(['ProdTaken'], axis=1)
y = train_set['ProdTaken']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, stratify=y
)

bayesian_params = {
    'max_depth': (6, 16),
    # 'num_leaves' : (24, 64),
    # 'min_child_samples' : (10, 200),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'colsample_bylevel': (0.5, 1),
    'max_bin': (10, 500),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50)
}


def xgb_hamsu(max_depth, min_child_weight,
              subsample, colsample_bytree, colsample_bylevel, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 500, "learning_rate": 0.02,
        'max_depth': int(round(max_depth)),  # 무조건 정수형
        # 'num_leaves' : int(round(num_leaves)),
        # 'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample, 1), 0),  # 0~1사이의 값
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'colsample_bylevel': max(min(colsample_bylevel, 1), 0),
        'max_bin': max(int(round(max_bin)), 100),  # 무조건 10 이상
        'reg_lambda': max(reg_lambda, 0),  # 무조건 양수만
        'reg_alpha': max(reg_alpha, 0),
    }

    #  * :: 여려개의 인자를 받겠다
    # ** :: 키워드 받겠다(딕셔너리형태)

    model = XGBClassifier(**params)

    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              # eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)

    return results


# xgb_bo = BayesianOptimization(f=xgb_hamsu,
#                               pbounds=bayesian_params,
#                               random_state=123
#                               )

# xgb_bo.maximize(init_points=5, n_iter=50)

# print(xgb_bo.max)

# 2. 모델
'''
{
    'target': 0.8500851788756388,
    'params': {
        'colsample_bylevel': 1.0,
        'colsample_bytree': 1.0,
        'max_bin': 456.98384724799683,
        'max_depth': 16.0,
        'min_child_weight': 1.0,
        'reg_alpha': 0.01,
        'reg_lambda': 0.001,
        'subsample': 1.0
    }
}
'''

# model = XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.02,
#     colsample_bylevel=max(min(1.0, 1), 0),
#     colsample_bytree=max(min(1.0, 1), 0),
#     max_bin=max(int(round(456.98384724799683)), 100),
#     max_depth=int(round(16.0)),
#     min_child_weight=int(round(1.0)),
#     reg_alpha=max(0.01, 0),
#     reg_lambda=max(0.001, 0),
#     subsample=max(min(1.0, 1), 0)
# )



model = ExtraTreesClassifier(n_estimators=200, ccp_alpha=0.0, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, random_state=123,)

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측
score = accuracy_score(y_test, model.predict(x_test))
print("결과 : ", score)

# 5. 제출
y_submit = model.predict(test_set)
submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(path + 'submission_14.csv', index=True)

# 결과 :  0.8388746803069054
# 결과 :  0.858603066439523
# 결과 :  0.8960817717206133 <- ExtraTreesClassifier
# 결과 :  0.8909710391822828
# 결과 :  0.9114139693356048
# 결과 :  0.9165247018739353
# 결과 :  0.9309462915601023
