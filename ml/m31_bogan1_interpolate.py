# 결측치 처리
# 1. 행 또는 열삭제
# 2. 임의의 값
#  = 평균(mean), 중위값(median), 0(fillna), 앞(ffill) 또는 뒤(bfill)의 값, 특정값(지정한 값을때려넣는다~) 외 등등
# 3. 보간(interpolate) : 선형회귀방식과 같다, Nan값을 linear방식으로 찾음
# 4. 모델 - predict * 예시로들어주신것) model.predict([3]) 모델, 훈련 후 predict에 결측치를 넣어버림
# 5. 부스팅계열 : 통상 결측치, 이상치에 대해 자유롭다. 믿거나 말거나 ㅎㅎ
# : 결측치나 이상치를 포함하여 작업함 (트리계열) 신뢰도가 애매하다.

import pandas as pd
import numpy as np
from datetime import date, datetime


dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022']

dates = pd.to_datetime(dates)
print(dates)
print("="*50)
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)
print("="*50)
ts = ts.interpolate()
print(ts)


'''

==================================================
2022-08-10     2.0
2022-08-11     NaN
2022-08-12     NaN
2022-08-13     8.0
2022-08-14    10.0
dtype: float64
==================================================
2022-08-10     2.0
2022-08-11     4.0
2022-08-12     6.0
2022-08-13     8.0
2022-08-14    10.0
dtype: float64


'''
