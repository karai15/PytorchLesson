import matplotlib.pyplot as plt
import numpy as np
import time
import os
from joblib import Parallel, delayed
import pickle
import scipy.io


def main():
    ########################
    # Model : y = Ax + Noise
    # y(N), A(N,M), x(M) (SMV)
    # L : num of nonzero elements
    ########################

    """
    ・SBL
    ・フレーム変えた場合
    ・
    """

    # Param
    opt_multi_process = 0
    param = {
        "save_path": "../data/CS/N64M256L25_Oracle_Minimized/",  # N64M256L25_Oracle_Gauss
        "load_path_frame": "../data/Frame/N64_M256_Niter1000/",
        "filename_frame": "frame.pickle",
        "opt_frame": "SIDCO",  # "Gauss", "Minimized", "SIDCO"
        "sidco_size": "64x256",
        "method": "Oracle",  # "OMP", "Oracle", "SBL"
        "N": 64,
        "M": 256,
        "L": 25,
    }
    snr_list = np.linspace(0, 40, 4)  # (start, stop, num)
    # snr_list = np.array([40])  # (start, stop, num)
    rep_list = np.arange(100)
    my_mkdir(param["save_path"])

    # Run
    if opt_multi_process == 1:  # multi_process
        print("run sim in multi_process")
        data = Parallel(n_jobs=-1)(
            [delayed(run_CS)(param, snr, n_rep) for n_rep in rep_list for snr in snr_list]
        )
    else:  # single_process
        data = []
        for n_rep in rep_list:
            for snr in snr_list:
                _data = run_CS(param, snr, n_rep)
                data.append(_data)

    ################
    # Convert to Numpy format
    N_rep = len(rep_list)
    N_snr = len(snr_list)
    NMSE_x = np.zeros((N_rep, N_snr), dtype=np.float64)
    cnt = 0
    for n_rep, _ in enumerate(rep_list):
        for n_snr, snr in enumerate(snr_list):
            NMSE_x[n_rep, n_snr] = data[cnt]
            cnt = cnt + 1
    mean_NMSE = np.mean(NMSE_x, axis=0)
    save_path = param["save_path"]
    ################

    # ################
    # NMSE_vs_SNR
    x = snr_list
    y = 10 * np.log10(mean_NMSE)
    plot_csv(x, y, save_path, filename="NMSE_vs_SNR")
    # CDF
    n_snr = -1
    plot_csv_cdf(10 * np.log10(NMSE_x[:, n_snr]), save_path, filename="CDF_NMSE")
    # ################


def plot_csv(x, y, save_path, filename):
    # plt
    plt.figure()
    plt.plot(x, y, "x-")
    plt.savefig(os.path.join(save_path, filename + ".jpeg"))
    # csv
    data_csv = np.stack([x, y], 1)
    np.savetxt(os.path.join(save_path, filename + ".csv"), data_csv, delimiter=',')  # (x, y)


def plot_csv_cdf(data, save_path, filename):
    N_bins = 100
    h_weight = np.ones(len(data)) / len(data)
    cdf = plt.hist(data, bins=N_bins, weights=h_weight, cumulative=True, histtype='step')  # (y, x)
    y = cdf[0][0:N_bins]
    x = (cdf[1][0:N_bins])
    plot_csv(x, y, save_path, filename)


def run_CS(param, snr, n_rep):
    np.random.seed(seed=n_rep)

    # param
    method = param["method"]
    N = param["N"]
    M = param["M"]
    L = param["L"]
    opt_frame = param["opt_frame"]

    ########################
    # Generate Model
    ########################
    pn = 10 ** (-snr / 10)

    # Frame
    if opt_frame == "Gauss":
        A_frame = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
    elif opt_frame == "Minimized":
        fp = os.path.join(param["load_path_frame"], param["filename_frame"])
        with open(fp, "rb") as f:
            A_frame = pickle.load(f)
            print("load data: " + fp)
    elif opt_frame == "SIDCO":
        fp = f'./SIDCO/{param["sidco_size"]}.mat'
        Dictionary_mat = scipy.io.loadmat(fp)
        A_frame = Dictionary_mat["A_QCS"]


    # mutual_coherence, coherence_set = calc_coherence(A_frame)
    A_frame = (A_frame / np.sqrt(np.sum(np.abs(A_frame) ** 2))) * np.sqrt(N * M)  # 平均電力を1に制約
    id_nonzero = np.random.choice(range(M), L, replace=False)

    x = np.zeros(M, dtype=np.complex128)
    x[id_nonzero] = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)
    Noise = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2) * np.sqrt(pn)
    y = A_frame @ x + Noise

    ########################
    # Estimate X
    ########################
    if method == "OMP":
        x_est, S = OMP(y, A_frame, L=L)

    elif method == "Oracle":
        A_true = A_frame[:, id_nonzero]
        # _x_est = np.linalg.inv(np.conj(A_true).T @ A_true + pn * np.eye(L)) @ np.conj(A_true).T @ y
        _x_est = np.linalg.pinv(A_true) @ y
        x_est = np.zeros(M, dtype=np.complex128)
        x_est[id_nonzero] = _x_est

    elif method == "SBL":
        N_iter_sbl = 150
        alpha_0 = 1e-2 * np.ones(M, dtype=np.float64)
        X_sbl, alpha = SBL(y[:, None], A_frame, x[:, None], 1/pn, alpha_0, N_iter_sbl)
        x_est = X_sbl[:, 0]


    ########################
    # Evaluate NMSE
    ########################
    NMSE_x = NMSE(x, x_est)
    print(f"NMSE(x) = {10 * np.log10(NMSE_x)} [dB]")

    # # ########################
    # # # PLOT (|X| vs index)
    # # ########################
    # plt.plot(np.abs(x), "o", label="True")
    # plt.plot(np.abs(x_est), "x", label=method)
    # plt.xlabel("index of x")
    # plt.ylabel("|x|^2")
    # plt.legend()
    # plt.show()

    return NMSE_x


def NMSE(X, X_hat):
    return np.sum(np.abs(X - X_hat) ** 2) / np.sum(np.abs(X) ** 2)

def calc_coherence(X):
    M, N = X.shape
    norm_col = np.sqrt(np.sum(np.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = np.conj(X_normlized).T @ X_normlized
    in_coherence = np.abs(Gram - np.eye(N))
    mutual_coherence = np.max(in_coherence)
    return mutual_coherence, in_coherence

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


def SBL(Y, H, X, beta, alpha, N_iter):
    # alpha : 事前精度（事前分散の逆数）
    # beta : 雑音精度（雑音分散の逆数）
    N, K = Y.shape
    M = H.shape[1]

    opt_debug = 0
    if opt_debug == 1:
        Obj = np.zeros(N_iter, dtype=np.float64)
    if N >= M:
        bHH = beta * np.conj(H).T @ H  # (M,M)
    bHY = beta * np.conj(H).T @ Y
    alpha_inv = 1 / alpha

    for n_iter in range(N_iter):
        print(f"n_iter = {n_iter}")

        if N >= M:
            Phi = np.diag(alpha)
            Gamma = Phi + bHH  # (M,M)
            Gamma_inv = np.linalg.pinv(Gamma)  # (M,M)
        else:  # 逆行列補題利用した場合の計算 (M > N のとき有効)
            C_inv = np.einsum('nm,m,mq->nq', H, alpha_inv, np.conj(H).T, optimize='optimal')  # (N, N)
            C_inv[range(N), range(N)] = C_inv[range(N), range(N)] + 1 / beta
            C_inv = np.linalg.inv(C_inv)
            Gamma_inv = - np.einsum('m,mn,nq,qp,p->mp', alpha_inv, np.conj(H).T, C_inv, H, alpha_inv,
                                    optimize='optimal')  # # (M, M)
            Gamma_inv[range(M), range(M)] = Gamma_inv[range(M), range(M)] + alpha_inv

        X_bar = Gamma_inv @ bHY  # (M,K)
        alpha_inv = np.real(Gamma_inv[range(M), range(M)]) + np.sum(np.abs(X_bar) ** 2, axis=1)
        alpha = 1 / alpha_inv

        # calc objective function
        if opt_debug == 1:
            Sigma_y = H @ np.diag(alpha_inv) @ np.conj(H).T + 1 / beta * np.eye(N)
            det_Sigma_y = np.linalg.det(Sigma_y)
            Sigma_y_inv = np.linalg.pinv(Sigma_y)
            Obj[n_iter] = np.real(np.log(det_Sigma_y) + np.trace(Sigma_y_inv @ Y @ np.conj(Y).T))
            NMSE_sbl = 10 * np.log10(np.sum(np.abs(Y - H @ X_bar) ** 2) / np.sum(np.abs(Y) ** 2))
            print(f"({n_iter}) Obj_func = {Obj[n_iter]},  NMSE(Y-HX) = {NMSE_sbl} [dB]")

        # #########################
        # TEST PLOT (|X|)
        # plt.plot(np.mean(np.abs(X) ** 2, axis=1), "o", label="True(|X|^2)")
        # plt.plot(alpha_inv, "^", label="SBL(1/alpha)")
        # plt.plot(Gamma_inv[range(M), range(M)], "v", label="SBL(Gamma_inv)")
        # plt.plot(np.mean(np.abs(X_bar) ** 2, axis=1), "x", label="SBL(|X|^2)")
        # plt.legend()
        # plt.show()
        # #########################

    # #########################
    # TEST PLOT (Minimized objective function) (収束の確認)
    plt.plot(Obj)
    plt.xlabel("n_iter")
    plt.ylabel("Minimized objective function")
    plt.show()
    # #########################

    return X_bar, alpha


def my_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


main()
