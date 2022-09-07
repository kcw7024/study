from operator import le
import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

# def f(x):
#   temp = x**2 - 4*x + 6
#   return temp

gradient = lambda x : 2*x - 4 

x = 8.0 # 초기값
epochs = 20
learning_rete = 0.25

print("step\t x\t f(x)")

for i in range(epochs):
  x = x - learning_rete * gradient(x) 
  print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))

  
  



