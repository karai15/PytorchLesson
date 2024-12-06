import torch
import numpy as np
import matplotlib.pyplot as plt

# import time
# import os
# import pickle
# import subprocess
# import sys


def main():
    # 乱数シードの設定
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    #################
    # parameters
    learning_rate = 1e-4
    p_boj = 3
    w_alpha = 1e0 * 0
    N_iter = 5000
    #
    N = 40
    G_theta = 2 * N
    #################

    # ベースのグリッド　　
    delta_theta = 2 / G_theta
    cor_ant = torch.arange(N) / 2
    theta_grid_base = torch.arange(-1, 1, 2 / G_theta)  # [-pi/2, pi/2]
    # 目的変数
    alpha = torch.zeros(G_theta, dtype=torch.float32)
    # alpha = torch.randn(G_theta, dtype=torch.float32) / 100
    alpha.requires_grad = True

    # optimizer = torch.optim.Adam([alpha], lr=learning_rate)
    optimizer = torch.optim.SGD([alpha], lr=learning_rate)

    data_loss = np.zeros(N_iter, dtype=np.float32)
    zeros_G = torch.zeros(G_theta, dtype=torch.float32)
    for n_iter in range(N_iter):

        # アレイ応答
        theta_grid = theta_grid_base + alpha
        Ar = array_factor_ts(theta_grid[None, :], cor_ant)[0, :, :]

        # Loss計算
        coherence = torch.sum(torch.abs(torch.conj(Ar).T @ Ar) ** p_boj) ** (1 / p_boj)
        P_alpha = torch.sum(torch.maximum(torch.abs(alpha) - delta_theta/2, zeros_G)) * w_alpha
        loss = coherence + P_alpha
        data_loss[n_iter] = loss

        ##########################
        # Update
        optimizer.zero_grad()  # 勾配を０に初期化．これをしないと，ステップするたびに勾配が足し合わされる
        loss.backward(retain_graph=False)  # retain_graph=False: backward後に計算グラフを開放（デフォルトでFalse）
        optimizer.step()
        ##########################

        print(f"({n_iter}) loss = {loss:.4f}, coherence = {coherence:.4f}, P_alpha = {P_alpha:.4f}")

    ##########################
    # PLOT
    alpha_np = alpha.to('cpu').detach().numpy().copy()
    theta_grid_base_np = theta_grid_base.to('cpu').detach().numpy().copy()
    plt.figure()
    plt.plot(data_loss, "x", label="loss")
    plt.legend()
    #
    plt.figure()
    plt.plot(alpha_np, "x", label="alpha")
    plt.axhline(y=delta_theta/2, color='red', linestyle='--', label=r'$|\Delta/2|$')
    plt.axhline(y=-delta_theta/2, color='red', linestyle='--')
    plt.legend()
    #
    plt.figure()
    # plt.plot(alpha_np + theta_grid_base_np, np.zeros(G_theta), "x", label="theta_grid")
    plt.plot(alpha_np + theta_grid_base_np, "x", label="theta_grid")
    plt.legend()
    plt.show()
    ##########################

    test = 1


def array_factor_ts(sin_theta, yn):
    Phi = torch.einsum('n,dl->dnl', yn, sin_theta)  # (N_data, N, L)
    A_ff = torch.exp(1j * 2 * torch.pi * Phi)
    return A_ff  # (N, L)

    test = 1


if __name__ == '__main__':
    main()
