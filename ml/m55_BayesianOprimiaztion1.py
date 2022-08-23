param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4),
                }

def y_function(x1, x2) : 
    return -x1 ** 2 - (x2 -2) **2 + 10



from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
                                f=y_function, # 파라미터를 찾고자 하는 함수
                                pbounds=param_bounds, # 해당 함수의 파라미터를 딕셔너리 형태로 넣는다
                                random_state=1234
                                )


optimizer.maximize(
    init_points=2, #초기값
    n_iter=20, #횟수                   
)

print(optimizer.max)
# {'target': 9.999835918969607, 
#  'params': {'x1': 0.00783279093916099, 
#             'x2': 1.9898644972252864}}

