### 181113(Tue)  
p.21 - p.82  
chapter2.py : L1 - L67  
chapter3.py : L1 - L303
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

出力層の活性化関数  
回帰問題 : 恒等関数  
分類問題 : ソフトマックス関数

回帰問題  
例) 人の写った画像からその人の体重を予測する


ソフトマックス関数  
def softmax(a):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
exp_a = numpy.exp(a)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
sum_exp_a = numpy.sum(exp_a)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
y = exp_a / sum_exp_a  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return y

オーバーフロー  
-> 数には有効桁数があり、大きすぎる値は表現できない


ソフトマックス関数(ソフトマックス関数対策)  
def softmax(a):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
c = np.max(a)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
exp_a = numpy.exp(a - c)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
sum_exp_a = numpy.sum(exp_a)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
y = exp_a / sum_exp_a  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return y

ソフトマックス関数の特徴  
1. 出力が0から1.0の実数になる
2. 出力の総和が1になる  
->出力を確立として解釈できる  
->分類問題に適する

numpy.argmax(y)  
->最も大きな要素のインデックスを取得

Normalaize : 正規化  
-> データをある決められた範囲に変換する処理

前処理  
-> 入力データに対して決められた変換を行うこと

range(start, end, step)  
list( range(0, 10, 3) )  
-> [0, 3, 6, 9]

y = numpy.array([1, 2, 1, 0])  
t = numpy.array([1, 2, 0, 0])  
print(y == t) #[True True False True]  
numpy.sum(y == t)  
-> 3
