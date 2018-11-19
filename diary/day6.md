### 181118(Sun)  
p.165 - p.222  
chapter6.py : L150 - L185  
chapter7.py : L1 - L5
##### *Remember me*

過学習が起きる要因  
1. パラメータを大量に持ち、表現力の高いモデルであること
2. 訓練データが少ないこと

荷重減衰(Weight decay)  
大きな重みを持つことに対してペナルティを課すことで、過学習を抑制

Dropout  
ニューロンをランダムに消去しながら学習することで、過学習を抑制

検証データ(Validation data)  
テストデータを使ってハイパーパラメータを調整するとテストデータに対して過学習を起こすことになる  
訓練データから20%程度を検証データとして先に分離する

ハイパーパラメータの最適化
->「良い値」が存在する範囲を徐々に絞り込んでいく  
-> ざっくり指定する
-> 10のべき乗のスケールで範囲を指定する
```
例
weigt_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
```
うまくいきそうなハイパーパラメータの範囲を観察し、値の範囲を小さくしていく。


畳み込みニューラルネットワーク(convolutional neural network)  
->CNN

全結合層(Affineレイヤ)  
データの形状が無視されてしまう

畳み込み層(Convolution レイヤ)  
形状を維持する

畳み込み演算  
->フィルター演算  
フィルター-> カーネル という

パディング : データの周囲に固定のデータ(0など)を埋める  
ストライド : フィルターを適用する窓の間隔を調節する


プーリング層(Pooling レイヤ)  
縦横の方向の空間を小さくする  
領域を一つの要素に集約する
1. 学習するパラメータがない
2. チャンネル数は変化しない
3. 微小な位置変化に対してロバスト


im2col関数  
4次元データを2次元に変換する
```
im2col(input_data, filter_h, filter_w, stride=1, pad=0)
input_data : (データ数、チャンネル、高さ、横幅)の4次元配列からなる入力データ
filter_h : フィルターの高さ
filter_w : フィルターの横幅
stride : ストライド
pad : パディング
```

畳み込み層(Convolution レイヤ)  
```
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T #フィルターの展開
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        #transpose は、多次元配列の軸の順番を入れ替える関数
        #形をもとに戻す

        return out

```
プーリング層(Pooling レイヤ)  
1. 入力データを展開する
2. 行ごとに最大値を求める
3. 適切な出力サイズに整形する  

```
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W =x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        #形をもとに戻す

        return out
```
