import matplotlib.pyplot as plt
import numpy as np
import scipy
import time


def main():
    ########################
    # Model : y = Ax + Noise
    # y(N), A(N,M), x(M) (SMV)
    # L : num of nonzero elements
    ########################]

    """
    次回：ループ作って，データ改修
    """

    ########################
    # Generate Model
    ########################
    np.random.seed(seed=0)
    N, M, L = 250, 500, 25
    snr = 20


    pn = 10 ** (-snr / 10)
    A_frame = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    id_nonzero = np.random.randint(0, M - 1, L)
    x = np.zeros(M, dtype=np.complex128)
    x[id_nonzero] = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)
    Noise = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2) * pn
    y = A_frame @ x + Noise

    ########################
    # Estimate X
    ########################
    # SOMP
    t1 = time.time()
    x_omp, S = OMP(y, A_frame, L=L)
    t2 = time.time()

    ########################
    # Evaluate NMSE
    ########################
    NMSE_y_omp = 10 * np.log10(NMSE(y, A_frame @ x_omp))
    NMSE_x_omp = 10 * np.log10(NMSE(x, x_omp))
    print(f"(OMP)     NMSE(x) = {NMSE_x_omp} [dB], NMSE(y) = {NMSE_y_omp} [dB], Time = {t2-t1} [s]")

    ########################
    # PLOT (|X| vs index)
    ########################
    plt.plot(np.abs(x), "o", label="True")
    plt.plot(np.abs(x_omp), "^", label="OMP")
    plt.xlabel("index of x")
    plt.ylabel("|x|^2")
    plt.legend()
    plt.show()


def NMSE(X, X_hat):
    return np.sum(np.abs(X - X_hat) ** 2) / np.sum(np.abs(X) ** 2)


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
        err = rr - np.abs(np.conj(A_res[:, S == 0]).T @ r) ** 2 # (M, 1)

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



main()
