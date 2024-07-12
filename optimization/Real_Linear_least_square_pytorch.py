import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    線形モデルの最小２乗問題をPytorchで解く
    min_{x} || A @ x - b ||^2
    """

    # 乱数シードの設定
    seed = 1
    torch.manual_seed(seed)

    # データ生成
    M, N = 2, 2
    A = torch.randn((M, N))
    b = torch.randn(N)
    x = torch.randn(N, dtype=torch.float32, requires_grad=True)  # 未知パラメータ

    # 最適化手法の選択
    learning_rate = 1e-1
    # optimizer = torch.optim.Adam([x], lr=learning_rate)
    optimizer = torch.optim.SGD([x], lr=learning_rate)  # 未知パラメータxを登録

    # パラメータの更新
    N_iter = 100
    Loss_data = np.zeros(N_iter, dtype=np.float32)
    for n_iter in range(N_iter):
        loss = torch.sum(torch.abs(A @ x - b) ** 2)
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        loss.backward()
        optimizer.step()
        Loss_data[n_iter] = loss
        print(f"loss = {loss}")

    # 評価
    x_hat_GD = x
    x_hat_LS = torch.linalg.pinv(A) @ b
    error = torch.abs(x_hat_GD - x_hat_LS)
    print(f"error = {error}")

    # plot
    plt.plot(Loss_data, "x-")
    plt.show()

main()