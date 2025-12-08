# 通常のカルマンフィルタと定常カルマンフィルタを比較

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# システム設定
# ==========================================
# システム行列
A = np.array([[0, -0.7], [1, -1.5]])
C = np.array([[0, 1]])

# ノイズ設定
Q_scalar = 0.1  # プロセスノイズ分散
R_scalar = 0.5  # 観測ノイズ分散

# プロセスノイズが加わる場所 (B_v)
B_v = np.array([[0.5], [1.0]])

# 制御入力の係数 (B_u)。例題6.4に合わせ B_v と同じ係数を使用
B_u = B_v.copy()

# カルマンフィルタ用のQ行列変換: Q_kf = B_v * Q * B_v.T
Q_kf = B_v @ np.array([[Q_scalar]]) @ B_v.T
R_kf = np.array([[R_scalar]])

# シミュレーション長
N = 50

# 制御入力: サイン波 u[k] = sin(0.1 k)
u = 5 * np.sin(0.15 * np.arange(N))

# ==========================================
# 2. 定常カルマンゲインの計算
# ==========================================
# 代数リッカチ方程式 (DARE) を解く
#   scipy.linalg.solve_discrete_are(A, B, Q, R)
#   制御との双対性: A -> A.T, B -> C.T
P_ss_minus = scipy.linalg.solve_discrete_are(A.T, C.T, Q_kf, R_kf)

# 定常ゲイン K_ss を計算
# K = P C^T (C P C^T + R)^-1
S_ss = C @ P_ss_minus @ C.T + R_kf
K_ss = P_ss_minus @ C.T @ np.linalg.inv(S_ss)

print("=== Steady State Calculation ===")
print("Converged Covariance P_ss:\n", P_ss_minus)
print("Steady State Gain K_ss:\n", K_ss)

# ==========================================
# 3. シミュレーション実行
# ==========================================
# 真値データの生成
np.random.seed(42)
x_true = np.zeros((2, N))
y_obs = np.zeros(N)
x = np.zeros((2, 1))

for k in range(N):
    x_true[:, k] = x.flatten()
    # 観測 y = Cx + w
    w = np.random.normal(0, np.sqrt(R_scalar))
    y_obs[k] = (C @ x).item() + w

    # 更新 x = Ax + B_u u + B_v v
    v = np.random.normal(0, np.sqrt(Q_scalar))
    x = A @ x + B_u * u[k] + B_v * v

# --- 比較用の2つのフィルタ ---
# Filter 1: Standard (通常のKF)
# x_iter = np.zeros((2, 1))
x_iter = 5 * np.ones((2, 1))  # 間違っている推定値からスタート
P_iter = np.eye(2) * 5.0  # 初期共分散
est_iter = np.zeros((2, N))
P_history = np.zeros((N, 2, 2))
K_history = []  # ゲインの履歴保存用

# Filter 2: Steady-state (定常KF)
x_stat = x_iter.copy()
est_stat = np.zeros((2, N))
# ※定常KFでは P の更新は不要

for k in range(N):
    y = y_obs[k]

    # === Filter 1: Standard KF (time-varying gain) ===
    # 1. Time Update
    x_pred_iter = A @ x_iter + B_u * u[k]
    P_pred_iter = A @ P_iter @ A.T + Q_kf

    # 2. Measurement Update
    S = C @ P_pred_iter @ C.T + R_kf
    K_k = P_pred_iter @ C.T @ np.linalg.inv(S)
    K_history.append(K_k.flatten())  # ログ保存

    innovation = y - C @ x_pred_iter
    x_iter = x_pred_iter + K_k @ innovation
    P_iter = (np.eye(2) - K_k @ C) @ P_pred_iter

    est_iter[:, k] = x_iter.flatten()
    P_history[k] = P_iter

    # === Filter 2: Steady-state KF ===
    # 行列逆演算なし！
    # 1. Time Update
    x_pred_stat = A @ x_stat + B_u * u[k]

    # 2. Measurement Update (固定ゲイン K_ss を使用)
    innovation_stat = y - C @ x_pred_stat
    x_stat = x_pred_stat + K_ss @ innovation_stat

    est_stat[:, k] = x_stat.flatten()

# ==========================================
# 4. 結果の可視化
# ==========================================
plt.figure(figsize=(12, 14))

# 制御入力 u
plt.subplot(5, 1, 1)
plt.plot(u, label="Input u(k)")
plt.title("Control Input (sine wave)")
plt.ylabel("u")
plt.grid()
plt.legend(loc="upper right")

# 結果比較 (x1)
plt.subplot(5, 1, 2)
sigma_x1 = np.sqrt(P_history[:, 0, 0])
plt.plot(x_true[0, :], "k-", label="True State", alpha=0.6)
plt.plot(est_iter[0, :], "b--", label="Standard KF", linewidth=2)
plt.plot(est_stat[0, :], "r:", label="Steady-state KF (Fast)", linewidth=2)
plt.fill_between(
    range(N),
    est_iter[0, :] - 2 * sigma_x1,
    est_iter[0, :] + 2 * sigma_x1,
    color="b",
    alpha=0.06,
    label="Standard KF 95% CI",
)
plt.title("State Estimation Comparison (x1)")
plt.ylabel("x1")
plt.legend()
plt.grid()

# 結果比較 (x2) + 観測値 y
plt.subplot(5, 1, 3)
sigma_x2 = np.sqrt(P_history[:, 1, 1])
plt.plot(x_true[1, :], "k-", label="True State", alpha=0.6)
plt.plot(est_iter[1, :], "b--", label="Standard KF", linewidth=2)
plt.plot(est_stat[1, :], "r:", label="Steady-state KF (Fast)", linewidth=2)
plt.plot(y_obs, "g.", label="Observation y", alpha=0.7, markersize=6)
plt.fill_between(
    range(N),
    est_iter[1, :] - 2 * sigma_x2,
    est_iter[1, :] + 2 * sigma_x2,
    color="b",
    alpha=0.06,
    label="Standard KF 95% CI",
)
plt.title("State Estimation Comparison (x2)")
plt.ylabel("x2 / y")
plt.legend()
plt.grid()

# 誤差の比較（x1成分：逐次推定 - 定常推定）
plt.subplot(5, 1, 4)
diff = np.abs(est_iter[0, :] - est_stat[0, :])
plt.plot(diff, "m-")
plt.title("x1 estimation diff: |(standard KF) - (steady-state KF)|")
plt.ylabel(r"x1 estimation diff |$\hat{x}_{std}-\hat{x}_{ss}$|")
plt.xlabel("Step")
plt.grid()

# ゲインの収束確認
plt.subplot(5, 1, 5)
K_hist_arr = np.array(K_history)
plt.plot(K_hist_arr[:, 0], label=r"$K_1$", color="tab:blue")
plt.plot(K_hist_arr[:, 1], label=r"$K_2$", color="tab:orange")
plt.plot(
    np.full_like(K_hist_arr[:, 0], K_ss[0, 0]),
    label=r"$K_{1,\infty}$",
    linestyle=":",
    color="tab:blue",
)
plt.plot(
    np.full_like(K_hist_arr[:, 1], K_ss[1, 0]),
    label=r"$K_{2,\infty}$",
    linestyle=":",
    color="tab:orange",
)
plt.title("Kalman gain convergence", fontsize=13)
plt.xlabel("Iteration k", fontsize=12)
plt.ylabel("Kalman gain value", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "steady_state_kalman_filter_plot.png"
plt.savefig(output_path, dpi=400)
plt.show()
