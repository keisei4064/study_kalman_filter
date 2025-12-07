# 例題6.1

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_kalman_filter import LinearKalmanFilter


# ==========================================
# 2. シミュレーション設定（書籍の問題設定）
# ==========================================
# システムパラメータ
A = 1.0
B = 1.0
C = 1.0
Q = 1.0  # プロセスノイズの分散
R = 10.0  # 観測ノイズの分散

# シミュレーション設定
N = 150  # ステップ数
u = 0  # 入力は今回0とする => 正規白色雑音

# 真のシステムのデータ生成 (Generative Process)
np.random.seed(42)  # 再現性のため
v = np.random.normal(0, np.sqrt(Q), N)  # プロセスノイズ
w = np.random.normal(0, np.sqrt(R), N)  # 観測ノイズ

x_true = np.zeros(N)
y_obs = np.zeros(N)

# 初期値
x_true[0] = 0
y_obs[0] = C * x_true[0] + w[0]

# 真値の生成ループ
for k in range(1, N):
    x_true[k] = A * x_true[k - 1] + B * u + v[k - 1]
    y_obs[k] = C * x_true[k] + w[k]

# ==========================================
# 3. 推定実行
# ==========================================
# フィルタの初期化（今回は0からスタート）
P_0 = 0.0  # 初期共分散
x_0 = 0.0  # 初期推定値

kf = LinearKalmanFilter(A, B, C, Q, R, P_0, x_0)

# 結果保存用
x_est = np.zeros(N)
P_history = np.zeros(N)
x_minus_history = np.zeros(N)
x_est[0] = x_0
P_history[0] = P_0
x_minus_history[0] = x_0

# フィルタリングループ
for k in range(1, N):
    x_hat, P_hat, x_minus = kf.step(y_obs[k])
    x_minus_history[k] = x_minus  # 事前推定値（更新前の状態）
    x_est[k] = x_hat  # 事後推定値
    P_history[k] = P_hat  # 事後共分散

# ==========================================
# 4. 可視化
# ==========================================
plt.figure(figsize=(12, 8))

# 状態推定のプロット
plt.subplot(2, 1, 1)
plt.plot(x_true, "k-", label="True State (x)", alpha=0.6)  # 真値
plt.plot(
    x_minus_history, "b.", markersize=3, alpha=0.6, label="Prior (x_minus)"
)  # 事前推定値
plt.plot(y_obs, "g.", label="Observation (y)", alpha=0.8, markersize=3)  # 観測値
plt.plot(x_est, "r--", label="KF Estimate (x_hat)")  # 推定値

# 不確実性（標準偏差）のバンドを描画
#   2σ範囲: 95.5%の確率でその範囲に値が存在
sigma = np.sqrt(P_history)  # 分散 -> 標準偏差履歴
plt.fill_between(
    range(N),
    x_est - 2 * sigma,
    x_est + 2 * sigma,
    color="r",
    alpha=0.1,
    label="95% Confidence",
)

plt.title("Kalman Filter Simulation (Example 6.1)")
plt.legend()
plt.grid()

# カルマンゲインの推移
plt.subplot(2, 1, 2)
plt.plot(kf.K_log, "r-")
plt.title("Kalman Gain Evolution")
plt.xlabel("Step")
plt.ylabel("Gain K")
plt.grid()

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "ex6_1_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()
