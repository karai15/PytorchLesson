import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle


def main():
    # Param
    opt_Gauss_frame = 0
    save_path = "./output/N64_M256_Minimized_TC/"  # N64_M256_Niter1000
    input_path = "./input_frame/"
    input_filename = "N64_M256_Niter50000_Minimized_TC.pickle"  # N64_M256_SIDCO.pickle, N64_M256_Niter50000_Minimized.pickle, N64_M256_Niter50000_Minimized_TC.pickle
    my_mkdir(save_path)

    fp = os.path.join(input_path, input_filename)
    with open(fp, "rb") as f:
        X = pickle.load(f)
        print("load data: " + fp)
        N, M = X.shape
        X = (X / np.sqrt(np.sum(np.abs(X) ** 2))) * np.sqrt(N * M)  # 平均電力を1に制約

    if opt_Gauss_frame == 1:
        N, M = 64, 256
        N_rep = 100
        mutual_coherence_avg = 0
        for n_rep in range(N_rep):
            X = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
            X = (X / np.sqrt(np.sum(np.abs(X) ** 2))) * np.sqrt(N * M)  # 平均電力を1に制約
            mutual_coherence, coherence_set = calc_coherence(X)
            mutual_coherence_avg = mutual_coherence_avg + mutual_coherence

        mutual_coherence_avg = mutual_coherence_avg / N_rep
        print(f"mutual_coherence_avg = {mutual_coherence_avg}")

    N, M = X.shape
    Welch_bound = np.sqrt((M - N) / (N * (M - 1)))
    mutual_coherence, coherence_set = calc_coherence(X)
    print(f"mutual_coherence = {mutual_coherence}")
    print(f"Welch_bound = {Welch_bound}")

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


def calc_coherence(X):
    M, N = X.shape
    norm_col = np.sqrt(np.sum(np.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = np.conj(X_normlized).T @ X_normlized
    in_coherence = np.abs(Gram - np.eye(N))
    mutual_coherence = np.max(in_coherence)
    return mutual_coherence, in_coherence


def my_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


t_1 = time.time()
main()
t_2 = time.time()
print(f"time = {t_2 - t_1}")
