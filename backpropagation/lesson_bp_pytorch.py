import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###################################################
# 合成関数の微分 (y -> z -> x)
x = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
z = 2 * x  # 中間変数
z.retain_grad()  # 中間変数の微分値を保存するための設定（デフォでは保存されない）
y = z ** 2
y.backward()
dy_dz = z.grad
dy_dx = x.grad
###################################################

###################################################
# NN(2層)のBP
x = torch.tensor(np.array([0.05, 0.10], dtype=np.float32))  # 入力
# torch.manual_seed(0)

### 第1層 ###
L1 = nn.Linear(2, 2)  # 2入力, 2出力
W1 = torch.tensor([[0.15, 0.20],
                   [0.25, 0.30]])  # 重み
b1 = torch.tensor([0.35, 0.35])  # バイアス

L1.weight.data = W1
L1.bias.data = b1

h0 = L1(x)  # 線形重み
h1 = torch.sigmoid(h0)  # 活性化関数を通す

### 第2層 ###
L2 = nn.Linear(2, 2)  # 2入力, 2出力
W2 = torch.tensor([[0.40, 0.45],
                   [0.50, 0.55]])  # 重み
b2 = torch.tensor([0.6, 0.6])  # バイアス

L2.weight.data = W2
L2.bias.data = b2

h2 = L2(h1)  # 線形重み
y = torch.sigmoid(h2)  # 活性化関数を通す

### 損失計算 ###
target_y = torch.tensor(np.array([0.15, 0.99]))
error_vec = (y - target_y) ** 2
loss = 1/2 * torch.sum(error_vec)
loss.backward()  # BP実行

print("\n d-loss_d-W1:", L1.weight.grad,
      "\n d-loss_d-b1:", L1.bias.grad,
      "\n d-loss_d-W2:", L2.weight.grad,
      "\n d-loss_d-b2:", L2.bias.grad)
###################################################

