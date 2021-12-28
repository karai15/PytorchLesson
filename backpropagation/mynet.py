import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Weight param
W1 = np.array([[0.15, 0.20],
               [0.25, 0.30]], dtype=np.float32)
b1 = np.array([0.35, 0.35], dtype=np.float32)
W2 = np.array([[0.4, 0.45],
               [0.50, 0.55]], dtype=np.float32)
b2 = np.array([0.6, 0.6], dtype=np.float32)

# Network
class MyNet(nn.Module):  # nn.Moduleを継承
    def __init__(self):
        super(MyNet, self).__init__()  # 継承クラスのコンストラクタを実行
        self.l1 = nn.Linear(2, 2)  # 第一層 2 -> 2
        self.l2 = nn.Linear(2, 2)  # 第二層 2 -> 2
        with torch.no_grad():  # この中で定義した変数は requires_grad = False となる
            self.l1.weight = nn.Parameter(torch.tensor(W1))
            self.l1.bias = nn.Parameter(torch.tensor(b1))
            self.l2.weight = nn.Parameter(torch.tensor(W2))
            self.l2.bias = nn.Parameter(torch.tensor(b2))

    def __call__(self, x):  # Foward
        h = torch.sigmoid(self.l1(x))  # (torch.tanh, torch.relu)でも可
        h = torch.sigmoid(self.l2(h))
        return h
