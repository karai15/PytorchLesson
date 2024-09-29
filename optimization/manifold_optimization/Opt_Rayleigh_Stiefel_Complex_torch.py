import numpy as np
import torch
import matplotlib.pyplot as plt

"""
max_X f(X) = tr(X^H A X)
s.t. X^H X = I 
"""

def main():
    torch.manual_seed(0)
    N_iter = 300
    N, M = 10, 4

    # 初期値
    A_tmp = torch.randn(N, N, dtype=torch.complex64)
    A = A_tmp @ torch.conj(A_tmp).T
    X_tmp = torch.randn(N, M, dtype=torch.complex64)
    X = torch.linalg.qr(X_tmp)[0]
    X.requires_grad = True

    # Adamのパラメータ https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    beta_mt = 0.9
    beta_vt = 0.999
    eps = 1e-8
    lr = 1e-2
    mt = torch.zeros(N,M, dtype=torch.complex64)  # 1次モーメントの初期値
    vt = torch.zeros(N,M, dtype=torch.complex64)  # 2次モーメントの初期値

    # Loop
    data_loss = np.zeros(N_iter + 1)
    data_loss[0] = torch.real(torch.trace(torch.conj(X.T) @ A @ X)).to('cpu').detach().numpy().copy()

    for n_iter in range(N_iter):

        loss = torch.real(torch.trace(torch.conj(X).T @ A @ X))
        print(f"(n_iter) loss = {loss}")

        #################
        # # パターン１ 手動で微分計算する場合
        # df_dX = A @ X * 2
        # grad_f = df_dX - X @ sym(torch.conj(X).T @ df_dX)
        # X = torch.linalg.qr(X + lr * grad_f)[0]
        #################

        # #################
        # # パターン2 自動微分 (GD)
        # loss.backward()
        # with torch.no_grad():
        #     df_dX = X.grad
        #     grad_f = df_dX - X @ sym(torch.conj(X).T @ df_dX)
        #     X.data = torch.linalg.qr(X.data + lr * grad_f)[0]
        #     X.grad.zero_()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        # #################

        #################
        # パターン3 自動微分 (ADAM)
        loss.backward()
        with torch.no_grad():
            df_dX = X.grad  # ユークリッド勾配
            grad_f = df_dX - X @ sym(torch.conj(X).T @ df_dX)  # リーマン勾配
            mt = beta_mt * mt + (1 - beta_mt) * grad_f
            vt = beta_vt * vt + (1 - beta_vt) * torch.abs(grad_f) ** 2
            mt_hat = mt / (1 - beta_mt ** (n_iter + 1))
            vt_hat = vt / (1 - beta_vt ** (n_iter + 1))
            X.data = torch.linalg.qr(X.data + lr * mt_hat / (vt_hat ** (1 / 2) + eps))[0]
            X.grad.zero_()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        #################

        data_loss[n_iter + 1] = loss.to('cpu').detach().numpy().copy()

    # 最適値
    U, s_value, V_T = np.linalg.svd(A.to('cpu').detach().numpy().copy())
    loss_opt = np.sum(s_value[0:M])

    print(f"loss_opt = {loss_opt}")
    print(f"loss_solv = {data_loss[-1]}")
    plt.plot(data_loss, "x")
    plt.show()


def sym(A):
    return (A + torch.conj(A).T) / 2


main()
