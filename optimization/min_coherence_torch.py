import torch
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    """
    相互コヒーレンスの最小化
    minimize \mu(X)
    """

    # 乱数シードの設定
    seed = 0
    torch.manual_seed(seed)

    # データ生成
    M, N = 25, 50
    X = torch.randn((M, N), dtype=torch.complex64, requires_grad=True)

    # 最適化手法の選択
    learning_rate = 1e-1
    # optimizer = torch.optim.Adam([X], lr=learning_rate)
    optimizer = torch.optim.SGD([X], lr=learning_rate)  # 未知パラメータxを登録

    # パラメータの更新
    N_iter = 30000
    Loss_data = np.zeros(N_iter, dtype=np.float32)
    for n_iter in range(N_iter):

        # コヒーレンスの計算
        mutual_coherence = calc_coherence(X)

        # パラメータの更新
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        mutual_coherence.backward()
        optimizer.step()
        Loss_data[n_iter] = mutual_coherence
        print(f"({n_iter}) mutual_coherence = {mutual_coherence}")
        # print(f"x = {x}")

    # Welch bound
    Welch_bound = np.sqrt((N - M) / (M * (N - 1))) * np.ones(N_iter)

    # plot
    plt.plot(Loss_data, "x-", label="SGD")
    plt.plot(Welch_bound, "--", label="Welch bound")
    plt.legend()
    plt.show()


def calc_coherence(X):
    M,N = X.shape
    norm_col = torch.sqrt(torch.sum(torch.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = torch.conj(X_normlized).T @ X_normlized
    mutual_coherence = torch.max(torch.abs(Gram - torch.eye(N)))
    return mutual_coherence

t_1 = time.time()
main()
t_2 = time.time()
print(f"time = {t_2 - t_1}")