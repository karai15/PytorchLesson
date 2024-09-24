import numpy as np
import torch


def main():
    # ##################
    # # 逆行列を含む関数の自動微分
    # torch.manual_seed(0)
    # N = 4
    # X = torch.randn(N, N, dtype=torch.complex64)
    # X.requires_grad = True
    # A = torch.randn((N, N), dtype=torch.complex64)
    # f = torch.norm(torch.linalg.inv(A @ X), "fro")
    # f.backward()
    # df_dx_auto = X.grad
    # test = 1
    # # ##################

    # ##################
    # # 逆行列＋行列の格納　を含む場合の自動微分
    # torch.manual_seed(0)
    # N = 4
    # X = torch.randn(N, N, dtype=torch.complex64)
    # X.requires_grad = True
    # A = torch.randn((N, N), dtype=torch.complex64)
    # B = torch.randn((N, N), dtype=torch.complex64)
    # C = torch.randn((N, N), dtype=torch.complex64)
    # D = torch.randn((N, N), dtype=torch.complex64)
    # #
    # # lossの計算
    # E = torch.zeros((2 * N, 2 * N), dtype=torch.complex64)
    # E[0:N, 0:N] = A @ X
    # E[0:N, N:] = B @ X
    # E[N:, 0:N] = C @ X
    # E[N:, N:] = D @ X
    # f = torch.norm(torch.linalg.inv(E), "fro")
    # f.backward()
    # df_dx_auto = X.grad
    # test = 1
    # # ##################

    ##################
    # 逆行列＋行列の格納　を含む場合の自動微分

    """
    次回：
    Eが単純なNxNの時にエラーが出ないことを確認（格納によって微分できていないことを確認しとくべき）
    """

    torch.manual_seed(0)
    N = 4
    X = torch.randn(N, N, dtype=torch.complex64)
    X.requires_grad = True
    A = torch.randn((N, N), dtype=torch.complex64)
    B = torch.randn((N, N), dtype=torch.complex64)
    C = torch.randn((N, N), dtype=torch.complex64)
    D = torch.randn((N, N), dtype=torch.complex64)

    # learning_rate = 1e-1
    learning_rate = 1e-3
    optimizer = torch.optim.SGD([X], lr=learning_rate)  # 未知パラメータxを登録

    # E = torch.zeros((2 * N, 2 * N), dtype=torch.complex64)
    # E.requires_grad = False
    N_iter = 10
    for n_iter in range(N_iter):

        # lossの計算
        # E = torch.zeros((2 * N, 2 * N), dtype=torch.complex64)
        # E[0:N, 0:N] = A @ X
        # E[0:N, N:] = B @ X
        # E[N:, 0:N] = C @ X
        # E[N:, N:] = D @ X
        E = A @ X

        f = torch.norm(torch.linalg.inv(E), "fro")
        # f = torch.norm(E, "fro")
        # f.backward()

        ###########
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        f.backward(retain_graph=True)
        # f.backward(retain_graph=False)
        # f.backward()
        optimizer.step()
        ###########

        # ###########
        # f.backward(retain_graph=False)
        # with torch.no_grad():
        #     df_dx_auto = X.grad
        #     X -= learning_rate * df_dx_auto
        # ###########

        # ###########
        # f.backward(retain_graph=False)
        # with torch.no_grad():
        #     df_dx_auto = X.grad
        # X = X - learning_rate * df_dx_auto
        # X.retain_grad()  # 中間変数の微分値を保存するための設定（デフォでは保存されない）
        # ###########

        # ###########
        # f.backward(retain_graph=False)
        # with torch.no_grad():
        #     df_dx_auto = X.grad
        #     X.data = X.data - learning_rate * df_dx_auto
        # ###########

        # ############
        # # df_dx_auto = torch.autograd.grad(f, X, create_graph=True)[0]
        # # df_dx_auto = torch.autograd.grad(f, X, create_graph=False)[0]
        # df_dx_auto = torch.autograd.grad(f, X, retain_graph=True)[0]
        # # df_dx_auto = torch.autograd.grad(f, X, retain_graph=False)[0]
        # X = X - learning_rate * df_dx_auto
        # ###########

        print(f"({n_iter}) Loss = {f}")

    # ##################

###############################
# メモ（一般的なこと）
###############################
# y.retain_grad()
#　中間変数の勾配を保存したい時に利用 (y.retain_grad がTrueになる)
# backward() した後に中間変数の微分が消えることを防ぐ
"""
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2  # 中間変数
y.retain_grad()  # 中間テンソルの勾配を保持 
z = y.sum()  # # 出力
print(f"y.grad: {y.grad}")
"""

# with torch.no_grad()　について
#　with torch.no_grad()ブロックを使うと、指定した範囲内で計算グラフの生成を無効にし、不要な計算グラフの構築を防ぐことができます。例えば、モデルの評価や勾配が不要な計算を行う場合に効果的です。

# retain_graph=True　について
# retain_graph=True を設定すると、計算グラフを保持したままにします。通常、backward() や torch.autograd.grad() を呼び出すと、計算グラフはその後解放されます。計算グラフを保持したい場合に retain_graph=True を指定します。
# 同じ微分計算を２回微分したい時とかに使う（計算グラフの開放を防ぐ）

# 20240922 疑問
# 計算グラフは，バックワードを計算すると，自動で解放されるにも関わらず，Forループでうまく動かないのはなぜ？　
# ブロック行列にしない場合は，うまく動くので何が違うのかがわからん．
# ただし，f.backward(retain_graph=True) にするとうまく動いてくれる．
# （このやり方だと，ループごとにグラフが保存されてるけど問題ないの？　Xが更新されるたびに新しい中間変数として扱われている？）
###############################

###############################
# メモ（今回のこと）
###############################
# 行列格納することで，X.gradが計算できなくなる
# 今の書き方で，X.retain_grad()　が必要な理由は，ステップ更新した時に，Xが新しい中間変数Xに生まれ変わって，その中間変数に対して，勾配保存を指定することで，次のステップで自動微分が実行できるようになってるのでは？
###############################

main()
