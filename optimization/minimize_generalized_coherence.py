import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
import pickle


def main():
    # 乱数シードの設定
    seed = 0
    torch.manual_seed(seed)

    # param
    N_iter = 10000
    N, M = 50, 200
    p = 2
    learning_rate = 1e-4
    device = "cpu"

    # 初期値
    X0 = torch.randn((N, M), dtype=torch.complex64, device=device)
    X_pow = (torch.sum(torch.abs(X0) ** 2)) ** (1 / 2)
    X = (X0 / X_pow)
    X.requires_grad = True
    X_ini = X.to('cpu').detach().numpy().copy()  # 初期フレームの保存

    # 最適化手法の選択
    optimizer = torch.optim.SGD([X], lr=learning_rate)  # 未知パラメータxを登録
    # optimizer = torch.optim.Adam([X], lr=learning_rate)

    # 保存データ
    data_Loss = np.zeros(N_iter, dtype=np.float32)
    I_M = torch.eye(M, dtype=torch.complex64, device=device)

    # 最適化ループ
    for n_iter in range(N_iter):

        X_pow = (torch.sum(torch.abs(X) ** 2)) ** (1 / 2)
        X_norm = (X / X_pow)
        mutual_coherence, coherence_set = calc_coherence(X_norm, I_M)
        loss = torch.norm(coherence_set, p=p)
        # loss = mutual_coherence

        print(f"({n_iter}) loss = {loss:.3f}")

        # パラメータの更新
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        loss.backward()
        optimizer.step()

        # データ回収
        data_Loss[n_iter] = loss.to('cpu').detach().numpy().copy()

    #################
    # Tensor -> Numpy
    X = X.to('cpu').detach().numpy().copy()
    #################

    welch_bound = np.sqrt((M - N) / (N * (M - 1)))
    print(f"welch_bound = {welch_bound}")

    plt.figure
    plt.plot(data_Loss, "-x")
    # plt.show()

    mutual_coherence, coherence_set = calc_coherence_np(X)
    save_path = "./csv_data"
    plot_csv_cdf(coherence_set.reshape(-1), save_path, filename="CDF_coherence")
    plt.show()


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


def plot_cmap(data, save_path, filename, vmin, vmax):
    plt.figure()
    plt.imshow(data, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(os.path.join(save_path, filename + ".jpeg"))


def calc_coherence(X, I_N):
    norm_col = torch.sqrt(torch.sum(torch.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = torch.conj(X_normlized).T @ X_normlized
    coherence_set = torch.abs(Gram - I_N)
    mutual_coherence = torch.max(coherence_set)
    return mutual_coherence, coherence_set


def calc_coherence_np(X):
    M, N = X.shape
    norm_col = np.sqrt(np.sum(np.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = np.conj(X_normlized).T @ X_normlized
    coherence_set = np.abs(Gram - np.eye(N))
    mutual_coherence = np.max(coherence_set)
    return mutual_coherence, coherence_set


def my_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# run
ts = time.time()
main()
te = time.time()
print(f"simulation time {te - ts} [s]")
