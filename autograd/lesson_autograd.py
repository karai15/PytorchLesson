import torch


def main():
    # #####################
    # N = 5
    # A = torch.randn((N, N), dtype=torch.complex64)
    # X = torch.randn((N, N), dtype=torch.complex64)
    # X.requires_grad = True
    #
    # p = 3
    # c = torch.trace(torch.conj(A).T @ X)
    # f = torch.abs(c) ** p
    #
    # f.backward()
    # df_dX_auto = X.grad / 2
    # df_dX = p/2 * torch.abs(c) ** (p - 2) * c * A
    #
    # eee = torch.abs(df_dX_auto - df_dX)
    # eee_max = torch.max(eee)
    # test = 1
    # #####################

    # #####################
    # x = torch.randn(1, dtype=torch.complex64)
    # x.requires_grad = True
    # # f = torch.real(x)
    # # f = 3 * torch.real(x)
    # f = 3 * x
    # f = 3 * torch.conj(x)
    # # f = 3 * (torch.conj(x) + x)
    # f.backward()
    # df_dx_auto = x.grad
    # df_dx = 1 / 2
    # eee = torch.abs(df_dx_auto - df_dx)
    # test = 1
    # #####################

    # ##################
    # torch.manual_seed(0)
    # N = 2
    # x = torch.randn(N, dtype=torch.complex64)
    # x.requires_grad = True
    # A = torch.randn((N, N), dtype=torch.complex64)
    # f = torch.conj(x).T @ A @ x
    # # df_dx_auto_2 = torch.autograd.grad(f, x, create_graph=True)
    # f.backward()
    # df_dx_auto = x.grad
    #
    # # df_dx_auto = f.grad
    # # df_dx = torch.conj(A).T @ x
    # df_dx = (A + torch.conj(A).T) @ x
    # # df_dx = torch.conj(A).T @ x + torch.conj(torch.conj(A).T @ x)
    # # df_dx = A.T @ torch.conj(x)
    # # df_dx = A @ x
    # eee = torch.abs(df_dx_auto - df_dx)
    # test = 1
    # # ##################

    ##################
    torch.manual_seed(0)
    N = 2
    x = torch.randn(N, dtype=torch.complex64)
    x.requires_grad = True
    b = torch.randn(N, dtype=torch.complex64)
    # f = torch.conj(b).T @ x
    f = torch.conj(x).T @ b

    f.backward()
    f.retain_grad()  # 中間変数の微分値を保存するための設定（デフォでは保存されない）
    # df_dx_auto = torch.autograd.grad(f, x, create_graph=True)

    aaa = f.grad

    test = 1
    # ##################


main()
