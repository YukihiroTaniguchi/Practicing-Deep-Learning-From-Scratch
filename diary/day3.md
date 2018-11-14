### 181114(Wed)  
p.83 - p.96
chapter4.py : L1 - L63
##### *Remember me*  

損失関数 #値が小さいほうが良い  
2乗和誤差  
交差エントロピー誤差   


交差エントロピー誤差  
cross_entropy_error(x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
delta = 1e-7  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return -numpy.sum(t * numpy.log(y + delta))

ミニバッチ学習  
-> データの中から一部を選び出し、  
その一部のデータを全体の「近似」として利用する

0から60000未満の数字の中からランダムに10子の数字を選び出す  
numpy.random.choice(60000, 10)  
-> array([38226, 20416,  9692, 16798, 40272, 29526,  3784, 12109, 52873,
       45077])

配列を行列にする(行数 : 1)  
x = x.reshape(1, x.size)

交差エントロピー誤差(ミニバッチ対応 : one-hot)  
cross_entropy_error(x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
if y.ndim == 1 :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
t = t.reshape(1, t.size)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
y = y.reshape(1, y.size)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
delta = 1e-7  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
batch_size = y.shape[0]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return -numpy.sum(t * numpy.log(y + delta)) / batch_size

arangeについて  
batch_size = 5  
numpy.arange(batch_size) #[0, 1, 2, 3 4]  
t -> [2, 7, 0, 9, 4]  
y[numpy.arange(batch_size), t]  
-> [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]  
-> 各データの正解ラベルに対応するニューラルネットワークの出力のみを抽出



交差エントロピー誤差(ミニバッチ対応 : !one-hot)  
cross_entropy_error(x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
if y.ndim == 1 :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
t = t.reshape(1, t.size)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
y = y.reshape(1, y.size)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
delta = 1e-7  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
batch_size = y.shape[0]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
return -numpy.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
