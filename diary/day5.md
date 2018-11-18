### 181117(Sat)  
p.165 - p.190  
chapter6.py : L1 - L149
##### *Remember me*  
確率的勾配降下法(SGD)  
-> 欠点 : 関数の形状が等方的でないと非効率な経路で探索することになる

モーメンタム(Momentum)  
ボールが地面を転がるように動く  
物体が勾配方向に力を受け、その力によって物体の速度が加算されるという物理法則
```
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = Momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

AdaGrad  
学習係数の減衰(lerning rate decay)を行う  
->パラメータの要素の中でよく動いた(大きく更新された)要素は、学習係数が小さくなる
```
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

Adam  
Momentum と AdaGrad のハイブリッド

隠れ層のアクティベーション  
-> 活性化関数のあとの出力データ

0と1に偏ったデータ分布では逆伝播での勾配の値がどんどん小さくなって消えてしまう  
->勾配消失

各層のアクティベーションの分布は、適度な広がりを持つことが求められる  
-> 適度に多様性のあるデータが各層を流れることで、ニューラルネットワークの学習が効率的に行える  
-> 偏ったデータが流れると勾配消失や「表現力の制限」が問題になる

Xavier の初期値  
```
node_num = 100 #前層のノード数
1/np.sqrt(node_num) #初期化
```

tanh関数  
原点(0, 0)で対称なS字カーブ


He の初期値  
```
node_num = 100 #前層のノード数
1/np.sqrt(2 /node_num) #初期化
```

ReLU関数  
->Xavierの初期値よりHeの初期値の方が良い

活性化関数  
ReLU関数 : Heの初期値  
Sigmoid関数 : Xavierの初期値  
Tanh関数 : Tanh関数  

Batch Normalization   
1. 学習を速く進行させる  
2. 初期値にそれほど依存しない  
3. 過学習を抑制する  
***
Hello Im Kazuki Ueno.  
Its important to study reguraly!  
Stick to it!  
