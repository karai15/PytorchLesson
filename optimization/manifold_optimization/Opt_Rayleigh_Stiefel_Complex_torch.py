import numpy as np
import torch
import matplotlib.pyplot as plt

"""
max_X f(X) = tr(X^H A X)
s.t. X^H X = I 
"""

def main():
    torch.manual_seed(0)
    lr = 1e-2
    N_iter = 100
    N, M = 10, 4

    # 初期値
    A_tmp = torch.randn(N, N, dtype=torch.complex64)
    A = A_tmp @ torch.conj(A_tmp).T
    X_tmp = torch.randn(N, M, dtype=torch.complex64)
    X = torch.linalg.qr(X_tmp)[0]
    X.requires_grad = True

    # Loop
    data_loss = np.zeros(N_iter + 1)
    data_loss[0] = torch.real(torch.trace(torch.conj(X.T) @ A @ X)).to('cpu').detach().numpy().copy()

    for n_iter in range(N_iter):

        loss = torch.real(torch.trace(torch.conj(X).T @ A @ X))
        print(f"(n_iter) loss = {loss}")

        #################
        # # 手動で微分計算する場合
        # df_dX = A @ X * 2
        # grad_f = df_dX - X @ sym(torch.conj(X).T @ df_dX)
        # X = torch.linalg.qr(X + lr * grad_f)[0]
        #################

        #################
        # 自動微分で計算する場合
        loss.backward()
        with torch.no_grad():
            df_dX = X.grad
            grad_f = df_dX - X @ sym(torch.conj(X).T @ df_dX)
            X = torch.linalg.qr(X + lr * grad_f)[0]
        X.requires_grad = True
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
