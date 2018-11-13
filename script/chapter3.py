#3.2.2 ステップ関数の実装
def step_function_proto(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function(x):
    y = x > 0
    return y.astype(np.int)

import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x
y = x > 0
y
y = y.astype(np.int)
y

#3.2.3 ステップ関数のグラフ
import matplotlib.pylab as plt

def step_function(x):
   return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)

#3.2.4 シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)

t = np.array([1.0, 2.0, 3.0])
1.0 + t
1.0 / t

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)

#3.2.7 ReLU関数
def relu(x):
    return np.maximum(0, x)

#3.3.1 多次元配列
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A)
A.shape
A.shape[0]

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
np.ndim(B)
B.shape

#行列の積
A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape
np.dot(A,B)
A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape
B = np.array([[1, 2], [3, 4], [5, 6]])
B.shape
np.dot(A, B)

C = np.array([[1, 2], [3, 4]])
C.shape
A.shape
np.dot(A, C)
A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape
B = np.array([7, 8])
B.shape
np.dot(A, B)

#3.3.3 ニューラルネットワークの行列の積
X = np.array([1, 2])
X.shape
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
W.shape
Y = np.dot(X, W)
print(Y)
