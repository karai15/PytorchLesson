import numpy as np
import matplotlib.pyplot as plt

"""
max_X f(X) = tr(X^T A X)
s.t. X^T X = I 
"""

def main():
    np.random.seed(4)
    lr = 1e-2
    N_iter = 100
    N, M = 10, 2

    # 初期値
    A_tmp = np.random.randn(N, N)
    A = A_tmp @ A_tmp.T
    X_tmp = np.random.randn(N, M)
    X = np.linalg.qr(X_tmp)[0]

    # Loop
    data_loss = np.zeros(N_iter + 1)
    data_loss[0] = np.trace(X.T @ A @ X)
    for n_iter in range(N_iter):
        df_dX = (A + A.T) @ X
        grad_f = df_dX - X @ sym(X.T @ df_dX)
        X = np.linalg.qr(X + lr * grad_f)[0]
        data_loss[n_iter + 1] = np.trace(X.T @ A @ X)

    # 最適値
    U, s_value, V_T = np.linalg.svd(A)
    loss_opt = np.sum(s_value[0:M])

    print(f"loss_opt = {loss_opt}")
    print(f"loss_solv = {data_loss[-1]}")
    plt.plot(data_loss, "x")
    plt.show()


def sym(A):
    return (A + A.T) / 2


main()
