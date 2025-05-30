import numpy as np
import cvxpy as cvx


def main():

    # https://www.cvxpy.org/functions/index.html


    np.random.seed(1)
    # 次元数
    m = 10
    n = 4
    # 定数・変数
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x = cvx.Variable(n, complex=True)
    # 問題定義
    obj = cvx.Minimize(cvx.sum_squares(A @ x - b))
    prob = cvx.Problem(obj)
    prob.solve(verbose=True)
    # 結果表示
    print("obj: ", prob.value)
    print("x: ", x.value)

main()


def SDP_test():

    import cvxpy as cp
    import numpy as np

    # Hermitian 変数（2x2 複素エルミート）
    X = cp.Variable((2, 2), hermitian=True)

    # Hermitian 行列 C
    C = np.array([[1, 1j],
                  [-1j, 1]])

    # 制約行列 A
    A = np.array([[1, 0],
                  [0, 0]])

    # 最適化問題の定義
    objective = cp.Minimize(cp.real(cp.trace(C @ X)))
    constraints = [
        cp.trace(A @ X) == 1,
        X >> 0  # X is Hermitian positive semidefinite
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # 結果表示
    print("Optimal X:")
    print(X.value)
