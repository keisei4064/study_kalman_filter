import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_kalman_filter import LinearKalmanFilter


# 例題6.4: システム制御のためのカルマンフィルタ
N = 50  # サンプル数（書籍のMATLABコードに合わせる）

# システム行列
A = np.array([[0.0, -0.7], [1.0, -1.5]])
B_w = np.array([[0.5], [1.0]])
B_u = B_w  # 制御入力uの係数
C = np.array([[0.0, 1.0]])

# 雑音パラメータ
Q = 0.1  # プロセスノイズ分散
R = 1.0  # 観測ノイズ分散

# 入力信号（PRBS 風の±1信号）
rng = np.random.default_rng(42)
u = rng.choice([-1.0, 1.0], size=N)

# 真値と観測の生成
x_true = np.zeros((N, 2))
y_obs = np.zeros(N)
process_noise = rng.normal(0, np.sqrt(Q), size=(N, 2))
measurement_noise = rng.normal(0, np.sqrt(R), size=N)

for k in range(1, N):
    x_prev = x_true[k - 1].reshape(-1, 1)
    x_true[k] = (A @ x_prev + B_u * u[k - 1]).flatten() + process_noise[k - 1]

y_obs[0] = (C @ x_true[0].reshape(-1, 1)).squeeze() + measurement_noise[0]
for k in range(1, N):
    y_obs[k] = (C @ x_true[k].reshape(-1, 1)).squeeze() + measurement_noise[k]

# フィルタ初期値
x_0 = np.zeros((2, 1))
P_0 = np.eye(2)

kf = LinearKalmanFilter(A, B_w, C, Q, R, P_0, x_0, B_u=B_u)

# 推定結果の保持
x_est = np.zeros((N, 2))
x_minus_history = np.zeros((N, 2))
P_history = np.zeros((N, 2, 2))

x_est[0] = x_0.squeeze()
x_minus_history[0] = x_0.squeeze()
P_history[0] = P_0

for k in range(1, N):
    x_hat, P_hat, x_minus = kf.step(y_obs[k], u[k - 1])
    x_est[k] = x_hat
    x_minus_history[k] = x_minus
    P_history[k] = P_hat

# 可視化 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# u(k)
axes[0].step(range(N), u, where="post", label="u(k)")
axes[0].set_ylabel("u")
axes[0].grid()
axes[0].legend(loc="upper right")

# x1
axes[1].plot(x_true[:, 0], "k-", label="True x1", alpha=0.6)
axes[1].plot(x_minus_history[:, 0], "b.", markersize=6, alpha=0.7, label="Prior x1")
axes[1].plot(x_est[:, 0], "r--", label="Estimate x1")
axes[1].set_ylabel("x1")
axes[1].grid()
axes[1].legend(loc="upper right")

# x2 + observation
axes[2].plot(x_true[:, 1], "k-", label="True x2", alpha=0.6)
axes[2].plot(x_minus_history[:, 1], "b.", markersize=6, alpha=0.7, label="Prior x2")
axes[2].plot(y_obs, "g.", label="Observation y", alpha=0.7, markersize=6)
axes[2].plot(x_est[:, 1], "r--", label="Estimate x2")
axes[2].set_ylabel("x2 / y")
axes[2].set_xlabel("Step")
axes[2].grid()
axes[2].legend(loc="upper right")

fig.suptitle("Kalman Filter with Control Input (Example 6.4)")
fig.tight_layout(rect=[0, 0, 1, 0.97])

output_path = Path(__file__).resolve().parent / "ex6_4_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()
