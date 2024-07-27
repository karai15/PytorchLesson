import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 乱数シードの設定
    seed = 0
    np.random.seed(seed)

    # Param
    K = 64  # Num. of sub-carriers (2^B)
    Q = 8  # Num. of pilots
    N_B = int(np.log2(K))
    N_ind = 100  # Num. of individual
    N_gen = 5000  # Num. of generations
    S_prob = 0.9  # SUSで選択する個体の数
    C_prob = 0.7  # 交叉率
    M_prob = 0.006  # 突然変異確率
    N_SUS = int(N_ind * S_prob)
    N_elite = N_ind - N_SUS

    ######################
    # param
    B = 1
    N_cp = 16
    G_tau = 32
    scs = B / K
    T_cp = (N_cp - 1) / B
    tau_grid = np.arange(0, T_cp, T_cp / G_tau)  # [-pi/2, pi/2]
    f_subs = np.arange(-B / 2, B / 2, scs)  # sub-carriers (K)
    ######################

    # バイナリ -> 整数 変換用 のマップ
    Map_binary = np.array([2 ** (N_B - b - 1) for b in range(N_B)])
    Map_binary = np.tile(Map_binary[None, :], (Q, 1))  # (Q, N_B)

    # 初期個体
    Phi_binary = np.random.randint(0, 2, N_B * Q * N_ind, dtype="bool")
    Phi_binary = Phi_binary.reshape(N_ind, Q, N_B)
    Phi = np.sum(Phi_binary * Map_binary[None, :, :], axis=2)  # (N_ind, Q)

    # 保存データ
    data_fitness = np.zeros((N_ind, N_gen))
    data_MC = np.zeros((N_ind, N_gen))
    data_TC = np.zeros((N_ind, N_gen))

    for n_gen in range(N_gen):
        # Fitnessの計算
        fitness, MC, TC = calc_fitness(Phi, f_subs, tau_grid)  # (N_ind)

        # # 親の選択（SUS : stochastic_universal_sampling）
        _id_SUS = stochastic_universal_sampling(fitness, N_SUS)  # (N_SUS)
        # _id_SUS = roulette_wheel_selection(fitness, N_SUS)  # (N_SUS)
        id_SUS = _id_SUS[np.random.permutation(N_SUS)]
        # Phi_SUS = Phi[id_SUS, :]  # (N_SUS, Q)
        Phi_binary_SUS = Phi_binary[id_SUS, :, :]  # (N_SUS, Q, N_B)

        # # 交叉
        Phi_binary_cross = np.copy(Phi_binary_SUS)
        for n_ind in range(int(N_SUS / 2)):
            parent_1 = Phi_binary_SUS[2 * n_ind, :, :]
            parent_2 = Phi_binary_SUS[2 * n_ind + 1, :, :]
            offspring = uniform_crossover(parent_1, parent_2, C_prob)  # (2, Q, N_B)
            Phi_binary_cross[(2 * n_ind):(2 * n_ind + 2), :, :] = offspring
        # Phi_cross = np.sum(Phi_binary_cross * Map_binary[None, :, :], axis=2)  # (N_SUS, Q)

        # # 突然変異
        flip_mask = np.random.rand(*Phi_binary_cross.shape) < M_prob  # 反転
        Phi_binary_mutation = np.where(flip_mask, 1 - Phi_binary_cross, Phi_binary_cross).astype("bool")
        Phi_mutation = np.sum(Phi_binary_mutation * Map_binary[None, :, :], axis=2)  # (N_SUS, Q)

        # # 個体の更新
        id_elite = np.argsort(fitness)[::-1][0:N_elite]  # Fitnessが大きい個体は残す
        Phi_update = Phi.copy()
        Phi_update[0:N_elite, :] = Phi[id_elite, :]
        Phi_update[N_elite:, :] = Phi_mutation
        Phi = Phi_update.copy()

        # # データ回収
        data_fitness[:, n_gen] = fitness
        data_MC[:, n_gen] = MC
        data_TC[:, n_gen] = TC

        # # 出力
        print(f"({n_gen}) max_fit = {np.max(fitness)}, mean_fit = {np.mean(fitness)}")

    #################
    # # 解の出力
    fitness, _, _ = calc_fitness(Phi, f_subs, tau_grid)  # (N_ind)
    id_max = np.argmax(fitness)
    pilot_pattern = Phi[id_max, :]
    #
    f_subs_pilot = f_subs[pilot_pattern]
    B_sensing = delay_factor(tau_grid, f_subs_pilot)
    coherence_set = calc_coherence_np(B_sensing)
    TC = np.sqrt(np.sum(coherence_set ** 2))
    MC = np.max(coherence_set)
    #################

    #################
    # データ保存
    save_path = "./csv_data"
    fp = os.path.join(save_path, "GA_allocation.pickle")
    param_pilot = {}
    param_pilot["Q"] = Q
    param_pilot["id_subs_pilot"] = pilot_pattern
    with open(fp, "wb") as f:
        pickle.dump(param_pilot, f)
    print(f"save data : {save_path}")
    #################

    ###########################
    # # PLOT
    # Optimal pilot pattern
    ones_pilot = np.zeros(K)
    ones_pilot[pilot_pattern] = 1
    plt.figure()
    plt.plot(ones_pilot, "x", label=f"MC = {MC:.3f}, TC = {TC:.3f}")
    plt.legend()
    #
    # Fitness vs. Iterations, MC, TC
    max_fit = np.max(data_fitness, axis=0)
    min_MC = np.min(data_MC, axis=0)
    min_TC = np.min(data_TC, axis=0)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(max_fit, "x", label="max_fitness")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(min_MC, "x", label="min_MC")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(min_TC, "x", label="min_TC")
    plt.legend()
    plt.show()
    ###########################




def calc_fitness(Phi, f_subs, tau_grid):
    N_ind, Q = Phi.shape

    fit = np.zeros(N_ind, dtype=np.float64)
    MC = np.zeros(N_ind, dtype=np.float64)
    TC = np.zeros(N_ind, dtype=np.float64)
    for n_ind in range(N_ind):
        phi = Phi[n_ind, :]

        if np.unique(phi).size != phi.size:  # 重複がある場合
            _fit = 1 / 1000
            total_coherence = 1e12
            mutual_coherence = 1e12
        else:
            f_subs_pilot = f_subs[phi]
            B_sensing = delay_factor(tau_grid, f_subs_pilot)
            coherence_set = calc_coherence_np(B_sensing)
            total_coherence = np.sqrt(np.sum(coherence_set ** 2))
            mutual_coherence = np.max(coherence_set)
            _fit = 1 / total_coherence
            # _fit = 1 / mutual_coherence

        fit[n_ind] = _fit
        MC[n_ind] = mutual_coherence
        TC[n_ind] = total_coherence

    return fit, MC, TC


def calc_coherence_np(X):
    M, N = X.shape
    norm_col = np.sqrt(np.sum(np.abs(X) ** 2, axis=0))
    X_normlized = X / norm_col
    Gram = np.conj(X_normlized).T @ X_normlized
    coherence_set = np.abs(Gram - np.eye(N))
    return coherence_set


def delay_factor(tau_v, f_subs):
    if type(tau_v) != np.ndarray: tau_v = np.array([tau_v])
    B = np.exp(-1j * 2 * np.pi * f_subs[:, None] * tau_v[None, :])
    return B  # (K, L)


import numpy as np


def stochastic_universal_sampling(fitness_values, num_to_select):
    """
    Parameters:
        fitness_values (list or array): 各個体の適応度
        num_to_select (int): 選択する個体の数
    Returns:
        selected_indices (list): 選択された個体のインデックス
    """
    total_fitness = np.sum(fitness_values)
    pointers = np.linspace(0, total_fitness, num_to_select, endpoint=False)
    start_point = np.random.uniform(0, total_fitness / num_to_select)
    pointers += start_point

    cumulative_fitness = np.cumsum(fitness_values)
    selected_indices = np.searchsorted(cumulative_fitness, pointers)
    return selected_indices


def roulette_wheel_selection(fitness_values, num_to_select):
    """
    Parameters:
        fitness_values (list or np.ndarray): 各個体の適応度
        num_to_select (int): 選択する個体の数
    Returns:
        selected_indices (np.ndarray): 選択された個体のインデックス
    """

    cumulative_fitness = np.cumsum(fitness_values)
    total_fitness = cumulative_fitness[-1]

    selection_points = np.random.rand(num_to_select) * total_fitness
    selected_indices = np.searchsorted(cumulative_fitness, selection_points)
    return selected_indices


# 一様交差
def uniform_crossover(parent_1, parent_2, P_cross):
    Q, N_B = parent_1.shape

    offspring = np.zeros((2, Q, N_B), dtype="bool")
    if np.random.rand() < P_cross:  # 交差確率　P_cross
        crossover_mask = np.random.rand(Q, N_B) < 0.5
        offspring[0, :, :] = np.where(crossover_mask, parent_2[:, :], parent_1[:, :])
        offspring[1, :, :] = np.where(crossover_mask, parent_1[:, :], parent_2[:, :])
    else:
        offspring[0, :, :] = parent_1.copy()
        offspring[1, :, :] = parent_2.copy()

    return offspring

def calc_Weltch_bound(N, M):
    return np.sqrt((M - N) / (N * (M - 1)))





# run
ts = time.time()
main()
te = time.time()
print(f"simulation time {te - ts} [s]")