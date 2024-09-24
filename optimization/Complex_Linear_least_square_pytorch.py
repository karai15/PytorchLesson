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

    # 最適化手法の選択
    learning_rate = 1e-1
    # optimizer = torch.optim.Adam([x], lr=learning_rate)
    optimizer = torch.optim.SGD([x], lr=learning_rate)  # 未知パラメータxを登録

    # パラメータの更新
    N_iter = 100
    Loss_data = np.zeros(N_iter, dtype=np.float32)
    for n_iter in range(N_iter):
        loss = torch.sum(torch.abs(A @ x - b) ** 2)

        # # ###########
        # # パターン１ (Optimizerを使う方法)
        # optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        # loss.backward()
        # optimizer.step()
        # # ###########

        ###########
        # パターン２（自動微分）
        loss.backward(retain_graph=False) # retain_graph=False: backward後に計算グラフを開放（デフォルトでFalse）
        with torch.no_grad():  # このブロック内部では計算グラフは構築されない
            df_dx_auto = x.grad
            x.data = x.data - learning_rate * df_dx_auto
            x.grad.zero_()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        ###########

        # ###########
        # # パターン３ (自動微分：ステップ後のｘを新しい中間変数として保存)
        # loss.backward(retain_graph=False)
        # with torch.no_grad():
        #     df_dx_auto = x.grad
        # x = x - learning_rate * df_dx_auto
        # x.retain_grad()  # 中間変数の微分値を保存するための設定
        # ###########

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