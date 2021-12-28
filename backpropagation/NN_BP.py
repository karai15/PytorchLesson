import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from mynet import MyNet

# param
learning_rate = 1e-1  # 学習率
iteration = 100

# NN
net = MyNet()  # Network
x = torch.tensor(np.array([0.05, 0.10], dtype=np.float32))  # input
t = torch.tensor(np.array([0.05, 0.99], dtype=np.float32))  # target
optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # SGDにパラメータセット

# 初期状態
y0 = net(x)  # foward
loss0 = F.mse_loss(y0, t)

# 学習
time_start = time.time()
for i in range(iteration):
    y = net(x)  # forward NN
    loss = F.mse_loss(y, t)
    net.zero_grad()  # init gradient (これをしないと勾配が加算されていく)
    loss.backward()
    optimizer.step()  # weight 更新
time_elasped = time.time() - time_start
print("time_elasped:", time_elasped)

# print
y = net(x)
loss = F.mse_loss(y, t)
W1_update = net.l1.weight
b1_update = net.l1.bias
W2_update = net.l2.weight
b2_update = net.l2.bias
print("\n # Loss before optimization:", loss0,
      "\n # Loss after optimization:", loss,
      "\n # W1_update:", W1_update,
      "\n # b1_update:", b1_update,
      "\n # W2_update:", W2_update,
      "\n # b2_update:", b2_update,)
