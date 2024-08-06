import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import subprocess


def main():
    """
    相互コヒーレンスの最小化
    minimize \mu(X)
    """

    # 乱数シードの設定
    seed = 0
    torch.manual_seed(seed)

    # Param
    opt_loss = "pnorm_coherence"  # "mutual_coherence", "total_coherence", "pnorm_coherence", "Log_Sum", "Exp_Sum"
    save_path = "../data/Frame/TEST/"  # N64_M256_Niter10000_ExpSum
    N, M = 64, 256
    N_iter = 500
    p = 3
    learning_rate = 1e-2  # 1e-4
    device = "cpu"  # "cpu", "mps", "cuda"
    my_mkdir(save_path)

    ##################
    # 使用するGPUを決定
    if device == "cuda":
        least_used_gpu = get_least_used_gpu()
        device = torch.device(f'cuda:{least_used_gpu}')
    else:
        device = torch.device(device)
    ##################

    ##################################
    # # データ生成
    X0 = torch.randn((N, M), dtype=torch.complex64, device=device)
    X_pow = (torch.sum(torch.abs(X0) ** 2)) ** (1 / 2)
    X = (X0 / X_pow)
    X.requires_grad = True
    X_ini = X.to('cpu').detach().numpy().copy()  # 初期フレームの保存

    # 最適化手法の選択
    # optimizer = torch.optim.Adam([X], lr=learning_rate)
    optimizer = torch.optim.SGD([X], lr=learning_rate)  # 未知パラメータxを登録

    # パラメータの更新
    # Loss_data = torch.zeros(N_iter, dtype=torch.float32, device=device)
    Loss_data = np.zeros(N_iter, dtype=np.float32)
    I_M = torch.eye(M, dtype=torch.complex64, device=device)
    I_M_real = torch.eye(M, dtype=torch.float32, device=device)
    for n_iter in range(N_iter):

        # コヒーレンスの計算
        mutual_coherence, coherence_set = calc_coherence(X, I_M)

        if opt_loss == "mutual_coherence":
            loss = mutual_coherence
        elif opt_loss == "total_coherence":
            total_coherence = torch.sum(coherence_set)
            loss = total_coherence
            # coherence_set = coherence_set.to('cpu').detach().numpy().copy()
            # plt.plot(coherence_set)
            # plt.show()
        elif opt_loss == "pnorm_coherence":
            loss = torch.norm(coherence_set, p=p)

        elif opt_loss == "Log_Sum":  # min Log-Sum = max Geo-Mean
            loss = torch.mean(torch.log(coherence_set + I_M_real))

        elif opt_loss == "Exp_Sum":
            loss = torch.mean(torch.exp(coherence_set))

        # パラメータの更新
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        loss.backward()
        optimizer.step()
        print(f"({n_iter}) loss = {loss:.4f}, MC = {mutual_coherence:.4f}")
        # print(f"x = {x}")

        # データ回収
        Loss_data[n_iter] = loss.to('cpu').detach().numpy().copy()

    # Welch bound
    Welch_bound = np.sqrt((M - N) / (N * (M - 1))) * np.ones(N_iter)
    print(f"Welch_bound = {Welch_bound[0]}")

    # Tensor -> Numpy
    X = X.to('cpu').detach().numpy().copy()
    X = (X / np.sqrt(np.sum(np.abs(X) ** 2))) * np.sqrt(N * M)  # 平均電力を1に制約
    mutual_coherence, coherence_set = calc_coherence_np(X)

    # save frame
    fp = os.path.join(save_path, "frame.pickle")
    with open(fp, "wb") as f:
        pickle.dump(X, f)
    ##################################

    ################
    # (plot) CDF_coherence, CDF_amplitude
    plot_csv_cdf(coherence_set.reshape(-1), save_path, filename="CDF_coherence")
    plot_csv_cdf(np.abs(X).reshape(-1), save_path, filename="CDF_amplitude")
    ################

    ################
    # (plot) Cmap_coherence, "Cmap_amplitude"
    plot_cmap(coherence_set, save_path, filename="Cmap_coherence", vmin=0, vmax=1)
    plot_cmap(np.abs(X), save_path, filename="Cmap_amplitude", vmin=0, vmax=None)
    ################

    ################
    # (plot) Loss_vs_iterations
    x = np.arange(N_iter) + 1
    plot_csv(x, Loss_data, save_path, filename="Loss_vs_iterations")
    plot_csv(x, Welch_bound, save_path, filename="Loss_vs_iterations_Welch_bound")
    ################


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


def calc_coherence(X, I_M):
    norm_col = torch.sqrt(torch.sum(torch.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = torch.conj(X_normlized).T @ X_normlized
    coherence_set = torch.abs(Gram - I_M)
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


def get_least_used_gpu():
    # nvidia-smiの出力を取得
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')

    # GPUの使用量と総メモリを解析
    memory_usage = []
    for line in output:
        used, total = map(int, line.split(', '))
        memory_usage.append((used, total))
    # 使用率（使用メモリ/総メモリ）が最も低いGPUを選択
    usage_ratios = [used / total for used, total in memory_usage]
    least_used_gpu = usage_ratios.index(min(usage_ratios))
    return least_used_gpu


t_1 = time.time()
main()
t_2 = time.time()
print(f"time = {t_2 - t_1}")
