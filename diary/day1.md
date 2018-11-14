### 181112(Mon)  
p.1 - p.20  
chapter1.py : L1 - L163
##### *Remember me*  
```
list = [1, 2, 3, 4, 5]  
list = [:-1] #最初から最後の要素の一つ前まで
```
```
dict = {'one' : 1}  
dict['two'] = 2 #新しい要素を追加
```
```
x = numpy.array([1.0. 2.0, 3.0])  
type(x) #class 'numpy.ndarray'  
-> N-dimensional array  
-> N次元配列
```

ブロードキャスト  
スカラ値

```
x = numpy.array([18, 25, 13])  
x[np.array([0, 2])] #インデックス0, 2番目を取得  
x > 15  
-> array([True, True, False])  
x[x > 15]  
-> array([18, 25])
```
