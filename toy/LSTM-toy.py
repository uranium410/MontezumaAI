#以下からお借りしました
#https://qiita.com/elm200/items/efd03a26ee9cb56a920a

import torch
import numpy as np
from itertools import chain

dict_size = 10
depth = 3
hidden_size = 6

# モデル定義
#embeddingは単語などの離散的なデータをベクトルで表現するためのもの(？)
#詳しくはここにhttp://kento1109.hatenablog.com/entry/2018/03/02/162807
embedding = torch.nn.Embedding(dict_size, depth)
#hidden_sizeのサイズ長のデータが出力される？
#cellstateも同じ長さ
lstm = torch.nn.LSTM(input_size=depth,
                            hidden_size=hidden_size,
                            batch_first=True)
linear = torch.nn.Linear(hidden_size, dict_size)
#交差エントロピー誤差
criterion = torch.nn.CrossEntropyLoss()
params = chain.from_iterable([
    embedding.parameters(),
    lstm.parameters(),
    linear.parameters(),
    criterion.parameters()
])
optimizer = torch.optim.SGD(params, lr=0.01)

# 訓練用データ
x = [[1,2, 3, 4]]
y = [5]

# 学習
for i in range(100):
    tensor_y = torch.tensor(y)
    input_ = torch.tensor(x)
    tensor = embedding(input_)
    #LSTMの出力は系列*隠れ層で出力される。
    #これを系列*出力ラベル数に変換しないといけない。

    #outputは各ゲート、メモリセルの状態の出力。実際の出力はtensorの部分(多分)
    output, (tensor, c_n) = lstm(tensor)
    #次元を減らしてるんかな？
    tensor = tensor[0]
    tensor = linear(tensor)
    loss = criterion(tensor, tensor_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print("{}: {}".format(i + 1, loss.data.item()))