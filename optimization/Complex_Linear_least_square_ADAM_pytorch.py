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
    A = torch.randn((M, N)) + 1j * torch.randn((M, N))
    b = torch.randn(N) + 1j * torch.randn(N)
    x = torch.randn(N, dtype=torch.complex64, requires_grad=True)  # 未知パラメータ

    # ADAMのパラメータ
    mt = 0  # 1次のモーメント
    vt = 0  # 2次のモーメント
    beta_mt = 0.9
    beta_vt = 0.999
    eps = 1e-7
    learning_rate = 1e-1

    # 最適化手法の選択
    optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(beta_mt, beta_vt), eps=eps,
                                 weight_decay=0, amsgrad=False)

    # パラメータの更新
    N_iter = 5
    Loss_data = np.zeros(N_iter, dtype=np.float32)
    for n_iter in range(N_iter):
        loss = torch.sum(torch.abs(A @ x - b) ** 2)

        # # ###########
        # # パターン１ (Optimizerを使う方法)
        # optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        # loss.backward(retain_graph=False)  # retain_graph=False: backward後に計算グラフを開放（デフォルトでFalse）
        # optimizer.step()
        # # ###########

        ###########
        # パターン２（自動微分） https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        loss.backward(retain_graph=False)
        with torch.no_grad():  # このブロック内部の記述は，計算グラフは構築されない
            df_dx_auto = x.grad
            mt = beta_mt * mt + (1 - beta_mt) * df_dx_auto
            vt = beta_vt * vt + (1 - beta_vt) * torch.abs(df_dx_auto) ** 2
            # vt = beta_vt * vt + (1 - beta_vt) * torch.conj(df_dx_auto) @ df_dx_auto
            # vt = beta_vt * vt + (1 - beta_vt) * torch.sum(torch.abs(df_dx_auto) ** 2)
            mt_hat = mt / (1 - beta_mt ** (n_iter + 1))
            vt_hat = vt / (1 - beta_vt ** (n_iter + 1))
            x.data = x.data - learning_rate * mt_hat / (vt_hat ** (1 / 2) + eps)
            x.grad.zero_()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        ###########

        Loss_data[n_iter] = loss
        print(f"({n_iter}) loss = {loss}")
        # print(f"x = {x}")

    # 評価
    x_hat_GD = x
    x_hat_LS = torch.linalg.pinv(A) @ b
    NMSE = 10 * torch.log10(torch.sum(torch.abs(x_hat_GD - x_hat_LS) ** 2))
    print(f"NMSE = {NMSE} [dB]")

    # plot
    plt.plot(Loss_data, "x-")
    plt.show()


main()
