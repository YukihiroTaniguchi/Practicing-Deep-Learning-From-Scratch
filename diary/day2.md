### 181113(Tue)  
p.21 - p.58  
chapter2.py : L1 - L67  
chapter3.py : L1 - L92
##### *Remember me*  
numpy.sum(x) #各要素の総和  

y = x > 0 #array([False, True, True], dtype=bool)  
numpy.array(x > 0, dtype = np.int)  
-> array(0, 1, 1)  

シグモイド関数  
sigmoid(x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return 1 / (1 + numpy.exp(-x))  

ReLU関数  
def relu(x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return np.maximum(0, x) #入力値の大きい方を出力  
->入力が0を超えていればそのまま、0以下ならば0を出力

x = numpy.array([1, 2, 3, 4])
numpy.ndim(x) #次元数  
-> 1

行列の積  
np.dot(A, B)  
-> Aの列数とBの列数を一致させる
