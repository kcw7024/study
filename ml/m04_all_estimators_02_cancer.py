from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import time

#1. 데이터
datasets = load_breast_cancer()
# print(datasets) (569,30)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data   #['data']
y = datasets.target #['target']
print(x.shape, y.shape) # (569,30), (569,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


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
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')


'''
AdaBoostClassifier 의 정답률 :  0.9532163742690059
BaggingClassifier 의 정답률 :  0.9415204678362573
BernoulliNB 의 정답률 :  0.9298245614035088
CalibratedClassifierCV 의 정답률 :  0.9590643274853801
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
ComplementNB 은 실행되지 않는다.
DecisionTreeClassifier 의 정답률 :  0.9473684210526315
DummyClassifier 의 정답률 :  0.6432748538011696
ExtraTreeClassifier 의 정답률 :  0.9239766081871345
ExtraTreesClassifier 의 정답률 :  0.9707602339181286
GaussianNB 의 정답률 :  0.9473684210526315
GaussianProcessClassifier 의 정답률 :  0.9707602339181286
GradientBoostingClassifier 의 정답률 :  0.9532163742690059
HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
KNeighborsClassifier 의 정답률 :  0.9590643274853801
LabelPropagation 의 정답률 :  0.9415204678362573
LabelSpreading 의 정답률 :  0.9415204678362573
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.9766081871345029
LogisticRegression 의 정답률 :  0.9883040935672515
LogisticRegressionCV 의 정답률 :  0.9883040935672515
MLPClassifier 의 정답률 :  0.9707602339181286
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 은 실행되지 않는다.
NearestCentroid 의 정답률 :  0.9415204678362573
NuSVC 의 정답률 :  0.9473684210526315
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.9707602339181286
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
RadiusNeighborsClassifier 은 실행되지 않는다.
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.9590643274853801
RidgeClassifierCV 의 정답률 :  0.9590643274853801
SGDClassifier 의 정답률 :  0.9532163742690059
SVC 의 정답률 :  0.9707602339181286
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.

'''