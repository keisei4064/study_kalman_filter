import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. システム設定 (例題6.4) ---
A = np.array([[0, -0.7], [1, -1.5]])
C = np.array([[0, 1]])  # 1x2 Matrix

# ノイズ共分散
Q_scalar = 0.01
R_scalar = 0.01

# プロセスノイズ行列 Q_kf = B_v * Q * B_v.T
B_v = np.array([[0.5], [1.0]])
Q_kf = B_v @ np.array([[Q_scalar]]) @ B_v.T

# 観測ノイズ行列 (1x1)
R_kf = np.array([[R_scalar]])

# --- 2. Iterative Solution (for loop) ---
# 時間発展させて収束値を見つける
P = np.eye(2) * 1.0  # 初期値
K_history = []

for k in range(50):
    # 予測誤差共分散の更新 (Time Update)
    P_minus = A @ P @ A.T + Q_kf

    # イノベーション共分散
    S = C @ P_minus @ C.T + R_kf

    # カルマンゲイン
    K = P_minus @ C.T @ np.linalg.inv(S)
    K_history.append(K.flatten())

    # 共分散の更新 (Measurement Update)
    P = (np.eye(2) - K @ C) @ P_minus

print(f"Iterative Result (k=50):\n K = {K_history[-1]}")


# --- 3. 解析解：代数リカッチ方程式 ---
# scipy.linalg.solve_discrete_are(A, B, Q, R)
# 制御との双対性により、A -> A.T, B -> C.T として渡す
P_inf_minus = scipy.linalg.solve_discrete_are(A.T, C.T, Q_kf, R_kf)

# 求まった P_inf (事前共分散) から 定常ゲイン K_ss を計算
S_inf = C @ P_inf_minus @ C.T + R_kf
K_ss = P_inf_minus @ C.T @ np.linalg.inv(S_inf)

print(f"\nAnalytical Result (DARE):\n K = {K_ss.flatten()}")


# --- ゲイン推移を可視化 ---
K_history = np.array(K_history)
plt.figure(figsize=(9, 5.0))
plt.plot(K_history[:, 0], label=r"$K_1$")
plt.plot(K_history[:, 1], label=r"$K_2$")
plt.plot(
    np.full_like(K_history[:, 0], K_ss.flatten()[0]),
    label=r"$K_{1,\infty}$",
    linestyle=":",
    color="tab:blue",
)
plt.plot(
    np.full_like(K_history[:, 1], K_ss.flatten()[1]),
    label=r"$K_{2,\infty}$",
    linestyle=":",
    color="tab:orange",
)
plt.xlabel("Iteration k", fontsize=12)
plt.ylabel("Kalman gain value", fontsize=12)
plt.title("Kalman gain convergence", fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
output_path = Path(__file__).resolve().parent / "steady_state_kalman_gain_plot.png"
plt.savefig(output_path, dpi=400)
plt.show()

# --- 検証 ---
error = np.linalg.norm(K_history[-1] - K_ss.flatten())
print(f"\nDifference: {error:.10e}")
if error < 1e-8:
    print(">> Success: Numerical and Analytical solutions match!")
else:
    print(">> Warning: Solutions do not match.")
