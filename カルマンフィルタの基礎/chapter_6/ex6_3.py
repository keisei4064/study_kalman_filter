# 例題6.3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_kalman_filter import LinearKalmanFilter

# --- シミュレーション設定 ---
N = 50
dt = 1.0

# システム係数
A = np.array(
    [
        [0, -0.7],
        [1, -1.5],
    ]
)
B = np.array(
    [
        [0.5],
        [1],
    ]
)
C = np.array([[0, 1]])

# ノイズパラメータ
Q = 1.0  # v(k) の分散（各状態に同じ分散の独立ノイズを加える）
R = 0.1  # w(k) の分散

# 初期状態
x_true = np.zeros((N, 2))  # 真値 [x1, x2]
y_obs = np.zeros(N)  # 観測値

# 真値データの生成
u = 0  # 入力は今回0とする => 正規白色雑音
np.random.seed(42)
v = np.random.normal(0, np.sqrt(Q), size=(N, 2))  # システム雑音（各状態独立）
w = np.random.normal(0, np.sqrt(R), N)  # 観測雑音
for k in range(1, N):
    x_prev = x_true[k - 1].reshape(-1, 1)
    x_true[k] = (A @ x_prev + B * u).flatten() + v[k - 1]

y_obs[0] = (C @ x_true[0].reshape(-1, 1)).squeeze() + w[0]
for k in range(1, N):
    y_obs[k] = (C @ x_true[k].reshape(-1, 1)).squeeze() + w[k]

# --- 推定実行 ---
x_0 = np.zeros((2, 1))  # 初期推定状態はゼロベクトル
P_0 = np.eye(2)  # 初期共分散は単位行列

kf = LinearKalmanFilter(A, B, C, Q, R, P_0, x_0)

# 推定値とログの保存
x_est = np.zeros((N, 2))
P_history = np.zeros((N, 2, 2))
x_minus_history = np.zeros((N, 2))
x_est[0] = x_0.squeeze()
P_history[0] = P_0
x_minus_history[0] = x_0.squeeze()

for k in range(1, N):
    x_hat, P_hat, x_minus = kf.step(y_obs[k])
    x_est[k] = x_hat
    P_history[k] = P_hat
    x_minus_history[k] = x_minus

# カルマンゲインを配列化（2 成分を分けて可視化）
K_history = np.vstack(kf.K_log)

# --- プロット ---
plt.figure(figsize=(12, 10))

# x1, x2 の推移
plt.subplot(3, 1, 1)
plt.plot(x_true[:, 0], "k-", label="True x1", alpha=0.6)
plt.plot(x_minus_history[:, 0], "b.", markersize=8, alpha=0.7, label="Prior x1")
plt.plot(x_est[:, 0], "r--", label="Estimate x1")
plt.legend()
plt.ylabel("x1")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(x_true[:, 1], "k-", label="True x2", alpha=0.6)
plt.plot(x_minus_history[:, 1], "b.", markersize=8, alpha=0.7, label="Prior x2")
plt.plot(y_obs, "g.", label="Observation (y)", alpha=0.8, markersize=8)
plt.plot(x_est[:, 1], "r--", label="Estimate x2")
plt.legend()
plt.ylabel("x2 / y")
plt.grid()

# カルマンゲインの推移
plt.subplot(3, 1, 3)
steps = np.arange(1, N)
plt.plot(steps, K_history[:, 0], "m-", label="K1")
plt.plot(steps, K_history[:, 1], "m--", label="K2")
plt.legend()
plt.title("Kalman Gain Evolution")
plt.xlabel("Step")
plt.ylabel("Gain K")
plt.grid()

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "ex6_3_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()
