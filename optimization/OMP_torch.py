import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    # 乱数シードの設定
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    device = "cpu"

    ####################
    # データ生成
    N, M, L = 10, 30, 3
    #
    # フレームの作成
    A_true = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
    A_true = A_true / np.sqrt(np.sum(np.abs(A_true) ** 2, axis=0))[None, :]
    SNR_A_dB = 20
    Noise_A = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2 * 10 ** (SNR_A_dB / 10))
    A = A_true + Noise_A  # 観測が作成された時と，推定する時でDictionaryが変化
    #
    # 観測の作成
    id_nonzero = np.random.choice(range(M), L, replace=False)
    x = np.zeros(M, dtype=np.complex128)
    x[id_nonzero] = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)
    y = A_true @ x
    ####################

    ####################
    # Numpy -> Torch
    A = torch.from_numpy(A.astype(np.complex64)).to(device)
    A.requires_grad = True
    x = torch.from_numpy(x.astype(np.complex64)).to(device)
    y = torch.from_numpy(y.astype(np.complex64)).to(device)
    ####################

    N_iter = 10
    for n_iter in range(N_iter):

        """
        memo
            ・Lossは単調に下がっているけど，Aの自由度が大きすぎて観測にフィットしているだけで，実際のｘを再現するように動いているわけではない．
            ・現時点では，Aのパワー制約が入っていない．
            （Aのパワーの列ノルムとｘの大きさはどちらを大きくしても再構成結果は同じになるので，一意に絞れない）
        """

        ####################
        # DOMP
        beta = 100

        ###
        norm_A = torch.sqrt(torch.sum(torch.abs(A) ** 2, axis=0))
        A_nrm = A / norm_A[None, :]
        x_hat, A_hat = differentiable_OMP(y, A_nrm, L, beta)
        ###

        # x_hat, A_hat = differentiable_OMP(y, A, L, beta)
        # 評価
        NMSE = torch.sum(torch.abs(y - A_hat @ x_hat) ** 2) / torch.sum(torch.abs(y) ** 2)
        print(f"({n_iter}) NMSE = {10 * torch.log10(NMSE)} [dB]")
        ####################

        ####################
        # # 確認用
        # OMP
        y_np = y.to('cpu').detach().numpy().copy()
        # A_np = A.to('cpu').detach().numpy().copy()
        A_np = A_nrm.to('cpu').detach().numpy().copy()

        x_np = x.to('cpu').detach().numpy().copy()
        x_hat_np, S = OMP(y_np, A_np, L)
        # 評価
        NMSE = np.sum(np.abs(y_np - A_np[:, S == 1] @ x_hat_np[S == 1]) ** 2) / np.sum(np.abs(y_np) ** 2)
        print(f"({n_iter}) NMSE_np = {10 * np.log10(NMSE)} [dB]")
        # plot
        plt.plot(np.abs(x_hat_np), "o", label="Est")
        plt.plot(np.abs(x_np), "x", label="True")
        plt.legend()
        plt.show()
        ####################

        ###########
        # 自動微分
        loss = torch.sum(torch.abs(y - A_hat @ x_hat) ** 2)
        loss.backward(retain_graph=False)
        with torch.no_grad():  # このブロック内部の記述は，計算グラフは構築されない
            learning_rate = 1e0
            df_dA_auto = A.grad
            A.data = A.data - learning_rate * df_dA_auto
            A.grad.zero_()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        ###########

        test = 1

    #################
    # Tensor -> Numpy
    # sigma = sigma.to('cpu').detach().numpy().copy()
    # plt.plot(sigma, "x")
    # plt.show()
    #################



def differentiable_OMP(y, A, L, beta):
    N, M = A.shape

    # フレームの正規化
    norm_A = torch.sqrt(torch.sum(torch.abs(A) ** 2, axis=0))
    A_nrm = A / norm_A[None, :]

    r = y
    for l in range(L):
        corr = torch.abs(torch.conj(A_nrm).T @ r)
        sigma = torch.nn.functional.softmax(beta * corr, dim=-1)
        sigma = sigma.to(torch.complex64)
        if l == 0:
            A_hat_l = A @ sigma
            x_hat_l = torch.linalg.pinv(A_hat_l[:, None]) @ y
            r = y - A_hat_l * x_hat_l
        else:
            a_hat_l = A @ sigma
            if l == 1:
                A_hat_l = torch.stack((A_hat_l, a_hat_l), dim=1)
            else:
                A_hat_l = torch.cat((A_hat_l, a_hat_l[:, None]), dim=1)
            x_hat_l = torch.linalg.pinv(A_hat_l) @ y
            r = y - A_hat_l @ x_hat_l

    return x_hat_l, A_hat_l


def OMP(y, A, L):
    """
    y = Ax
    :param y: measurement (N, 1)
    :param A: measurement matrix (N, M)
    :param e_th: iteration break condition
    :param L: num of nonzero coefficient
    :return:
        S: support (M, 1)
        x: solution (M, 1)
    """

    # param
    N, M = A.shape
    x = np.zeros(M, dtype=complex)
    S = np.zeros(M, dtype=np.uint8)
    r = y.copy()  # residual error
    rr = np.dot(r, np.conj(r))

    # Normalize
    A_res = np.copy(A)
    norm_A_res = np.sqrt(np.sum(np.abs(A_res) ** 2, axis=0))  # column norm  (M)
    A_res = A_res / norm_A_res[None, :]  # normalize

    for l in range(L):
        # print("OMP : iter = ", m)
        # calc metric
        err = rr - np.abs(np.conj(A_res[:, S == 0]).T @ r) ** 2  # (M, 1)

        # update support
        ndx = np.where(S == 0)[0]
        S[ndx[err.argmin()]] = 1

        # update x
        As = A[:, S == 1]
        As_pinv = np.linalg.pinv(As)
        xs = As_pinv @ y

        # update residual error
        r = y - As @ xs
        rr = np.dot(r, np.conj(r))

    x[S == 1] = xs

    return x, S


if __name__ == '__main__':
    main()
