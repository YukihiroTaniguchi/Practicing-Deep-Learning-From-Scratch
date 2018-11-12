
#1.3.1 算術計算
1 - 2
4 * 5
7 / 5
3 ** 2

#1.3.2 データ型
type(10)
type(2.718)
type("hello")

#1.3.3 変数
x = 10
print(x)
x = 100
print(x)
y = 3.14
x * y
type(x * y)

#1.3.4 リスト
a = [1, 2, 3, 4, 5]
print(a)
len(a)
a[0]
a[4]
a[4] = 99
print(a)

a[0:2]
a[1:]
a[:3]
a[:-1]
a[:-2]

#1.3.5 ディクショナリ
me = {'height' : 180}
me['height']
me['weight'] = 70
print(me)

#1.3.6 ブーリアン
hungry = True
sleepy = False
type(hungry)
not hungry
hungry and sleepy
hungry or sleepy

#1.3.7 if文
hungry = True
if hungry :
    print("I'm hungry")

hungry = False
if hungry :
    print("I'm hungry")
else :
    print("I'm not hungry")
    print("I'm sleepy")

#1.3.8 for文
for i in [1, 2, 3] :
    print(i)

#1.3.9 関数
def hello() :
    print("Hello World!")

hello()

def hello(object) :
    print(f"Hello {object}!")

hello("cat")

#1.4.2 クラス
class Man :
    def __init__(self, name) :
        self.name = name
        print("Initialize!")

    def hello(self) :
        print(f"Hello {self.name}!")

    def goodbye(self) :
        print(f"Good-bye {self.name}!")

m = Man("David")
m.hello()
m.goodbye()

#1.5.1 NumPyのインポート
import numpy as np

#1.5.2 NumPy配列の生成
x = np.array([1.0, 2.0, 3.0])
print(x)
type(x)

#1.5.3 NumPyの算術計算
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
x + y
x - y
x * y
x / y
x = np.array([1.0, 2.0, 3.0])
x / 2.0

#1.5.4 NumPyのN次元配列
A = np.array([[1, 2], [3, 4]])
print(A)
A.shape
A.dtype
B = np.array([[3, 0], [0, 6]])
A + B
A * B

A * 10

#1.5.5 ブロードキャスト
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A * B

#1.5.6 要素へのアクセス
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
X[0]
X[0][1]

for row in X:
    print(row)

X = X.flatten()
X[np.array([0, 2, 4])]

X > 15
X[X > 15]

#1.6.1 単純なグラフの描画
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)
plt.plot(x, y)

#1.6.2 pyplotの機能
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label = "sin")
plt.plot(x, y2, linestyle = "--", label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()

#1.6.3 画像の表示
from matplotlib.image import imread
img = imread('./others/macaroon.jpg')
plt.imshow(img)
