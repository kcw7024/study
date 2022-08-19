import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(x,y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# print(y_test)
# print(y_train)

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41

for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        # model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('Model Name : ', name)
        print('ACC : ', scores) 
        print('cross_val_score : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        #print(y_predict)
        
        #y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')
    
        

'''
Model Name :  AdaBoostClassifier
ACC :  [0.26984127 0.25396825 0.23809524 0.23015873 0.32539683 0.29365079
 0.22222222 0.312      0.232      0.232     ]
cross_val_score :  0.2609
Model Name :  BaggingClassifier
ACC :  [0.93650794 0.91269841 0.93650794 0.92857143 0.92063492 0.94444444
 0.91269841 0.912      0.912      0.936     ]
cross_val_score :  0.9252
Model Name :  BernoulliNB
ACC :  [0.87301587 0.81746032 0.88888889 0.82539683 0.88095238 0.88095238
 0.88095238 0.904      0.848      0.808     ]
cross_val_score :  0.8608
Model Name :  CalibratedClassifierCV
ACC :  [0.95238095 0.92857143 0.96825397 0.96031746 0.95238095 0.96031746
 0.96825397 0.976      0.96       0.936     ]
cross_val_score :  0.9562
Model Name :  CategoricalNB
ACC :  [       nan 0.84920635        nan 0.96031746        nan        nan
 0.88888889        nan        nan        nan]
cross_val_score :  nan
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
Model Name :  ComplementNB
ACC :  [0.83333333 0.73015873 0.85714286 0.84920635 0.76190476 0.84920635
 0.80952381 0.832      0.76       0.792     ]
cross_val_score :  0.8074
Model Name :  DecisionTreeClassifier
ACC :  [0.87301587 0.82539683 0.85714286 0.87301587 0.77777778 0.86507937
 0.84126984 0.832      0.856      0.824     ]
cross_val_score :  0.8425
Model Name :  DummyClassifier
ACC :  [0.0952381  0.07142857 0.08730159 0.07142857 0.12698413 0.05555556
 0.11904762 0.08       0.096      0.088     ]
cross_val_score :  0.0891
Model Name :  ExtraTreeClassifier
ACC :  [0.78571429 0.69047619 0.84126984 0.8015873  0.73015873 0.78571429
 0.75396825 0.816      0.776      0.664     ]
cross_val_score :  0.7645
Model Name :  ExtraTreesClassifier
ACC :  [0.98412698 0.96031746 0.99206349 0.98412698 0.98412698 1.
 0.99206349 0.984      0.96       0.96      ]
cross_val_score :  0.9801
Model Name :  GaussianNB
ACC :  [0.87301587 0.78571429 0.81746032 0.9047619  0.85714286 0.92063492
 0.83333333 0.904      0.784      0.752     ]
cross_val_score :  0.8432
Model Name :  GaussianProcessClassifier
ACC :  [0.1031746  0.07936508 0.12698413 0.04761905 0.1031746  0.07142857
 0.11904762 0.096      0.104      0.072     ]
cross_val_score :  0.0923
Model Name :  GradientBoostingClassifier
ACC :  [0.98412698 0.93650794 0.97619048 0.95238095 0.95238095 0.96825397
 0.95238095 0.976      0.928      0.976     ]
cross_val_score :  0.9602
Model Name :  HistGradientBoostingClassifier
ACC :  [1.         0.95238095 0.98412698 0.97619048 0.95238095 0.96031746
 0.95238095 0.976      0.944      0.96      ]
cross_val_score :  0.9658
Model Name :  KNeighborsClassifier
ACC :  [0.98412698 0.97619048 1.         0.96031746 0.97619048 1.
 0.98412698 0.968      0.96       0.976     ]
cross_val_score :  0.9785
Model Name :  LabelPropagation
ACC :  [0.13492063 0.1031746  0.07142857 0.08730159 0.05555556 0.11111111
 0.07142857 0.112      0.096      0.12      ]
cross_val_score :  0.0963
Model Name :  LabelSpreading
ACC :  [0.13492063 0.1031746  0.07142857 0.08730159 0.05555556 0.11111111
 0.07142857 0.112      0.096      0.12      ]
cross_val_score :  0.0963
Model Name :  LinearDiscriminantAnalysis
ACC :  [0.95238095 0.92063492 0.96031746 0.95238095 0.95238095 0.96825397
 0.95238095 0.968      0.904      0.944     ]
cross_val_score :  0.9475
Model Name :  LinearSVC
ACC :  [0.93650794 0.9047619  0.96825397 0.92857143 0.94444444 0.92857143
 0.95238095 0.944      0.936      0.912     ]
cross_val_score :  0.9355
Model Name :  LogisticRegression
ACC :  [0.96825397 0.95238095 0.96825397 0.97619048 0.96825397 0.97619048
 0.96031746 0.976      0.96       0.968     ]
cross_val_score :  0.9674
Model Name :  LogisticRegressionCV
ACC :  [0.96825397 0.96031746 0.96825397 0.96825397 0.96825397 0.97619048
 0.96031746 0.984      0.944      0.96      ]
cross_val_score :  0.9658
Model Name :  MLPClassifier
ACC :  [0.98412698 0.94444444 0.96825397 0.96825397 0.96031746 0.99206349
 0.96825397 0.96       0.936      0.968     ]
cross_val_score :  0.965
MultiOutputClassifier 은 실행되지 않는다.
Model Name :  MultinomialNB
ACC :  [0.8968254  0.84920635 0.95238095 0.92857143 0.87301587 0.92063492
 0.92063492 0.928      0.88       0.856     ]
cross_val_score :  0.9005
Model Name :  NearestCentroid
ACC :  [0.88095238 0.84126984 0.9047619  0.93650794 0.88095238 0.94444444
 0.91269841 0.928      0.872      0.888     ]
cross_val_score :  0.899
Model Name :  NuSVC
ACC :  [0.96825397 0.93650794 0.97619048 0.96031746 0.97619048 0.99206349
 0.96031746 0.96       0.936      0.936     ]
cross_val_score :  0.9602
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
Model Name :  PassiveAggressiveClassifier
ACC :  [0.96031746 0.92063492 0.97619048 0.96825397 0.96825397 0.97619048
 0.95238095 0.952      0.936      0.952     ]
cross_val_score :  0.9562
Model Name :  Perceptron
ACC :  [0.95238095 0.92063492 0.93650794 0.94444444 0.92857143 0.96031746
 0.93650794 0.952      0.912      0.96      ]
cross_val_score :  0.9403
Model Name :  QuadraticDiscriminantAnalysis
ACC :  [0.82539683 0.78571429 0.84126984 0.8015873  0.84920635 0.81746032
 0.79365079 0.904      0.752      0.712     ]
cross_val_score :  0.8082
Model Name :  RadiusNeighborsClassifier
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
RadiusNeighborsClassifier 은 실행되지 않는다.
Model Name :  RandomForestClassifier
ACC :  [0.97619048 0.92857143 0.98412698 0.97619048 0.96825397 0.98412698
 0.97619048 0.976      0.952      0.96      ]
cross_val_score :  0.9682
Model Name :  RidgeClassifier
ACC :  [0.92063492 0.92063492 0.93650794 0.92063492 0.95238095 0.96031746
 0.93650794 0.928      0.912      0.944     ]
cross_val_score :  0.9332
Model Name :  RidgeClassifierCV
ACC :  [0.92063492 0.92063492 0.93650794 0.92857143 0.95238095 0.96031746
 0.93650794 0.928      0.912      0.944     ]
cross_val_score :  0.934
Model Name :  SGDClassifier
ACC :  [0.93650794 0.8968254  0.96031746 0.93650794 0.95238095 0.97619048
 0.96031746 0.992      0.944      0.936     ]
cross_val_score :  0.9491
Model Name :  SVC
ACC :  [0.99206349 0.97619048 1.         0.98412698 0.99206349 1.
 0.97619048 0.984      0.96       0.984     ]
cross_val_score :  0.9849
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.


'''