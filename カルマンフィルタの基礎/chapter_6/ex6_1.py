# 例題6.1

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ScalarKalmanFilter:
    def __init__(
        self,
        A,
        B,
        C,
        Q,  # プロセスノイズの分散
        R,  # 観測ノイズの分散
        P_init,
        x_init,
    ):
        # numpy配列に統一（スカラで来ても2次元配列にする）
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))

        # 行列・スカラをメンバ変数として保持
        self.P = np.atleast_2d(np.asarray(P_init, dtype=float))  # 事後誤差共分散
        self.x = np.atleast_2d(np.asarray(x_init, dtype=float))  # 事後推定値

        # カルマンゲインの履歴
        self.K_log = []

    def step(self, y, u):
        """
        1ステップ分のカルマンフィルタ演算を行う
        y: 観測値 (Observation)
        u: 入力 (Input)
        return: 更新後の推定値 x, 共分散 P
        """

        # --- [STEP 1: 時間更新 (Time Update / Prediction)] ---

        # 事前推定値 (x_minus) を計算
        x_minus = self.A @ self.x

        # 事前誤差共分散 (P_minus) を計算
        #   ダイナミクスA と，プロセスノイズQ により不確実性が増大する
        #   Bが掛かるのは足立本の流派．よく見る式では只のQ．
        P_minus = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T

        # --- [STEP 2: 観測更新 (Measurement Update / Correction)] ---
        # カルマンゲイン (K) を計算
        # ヒント: 分母は (C * P_minus * C + R) です。逆行列(割り算)の扱いに注意。
        # K = ...
        y_arr = np.atleast_2d(np.asarray(y, dtype=float))
        S = self.C @ P_minus @ self.C.T + self.R
        K = (P_minus @ self.C) @ np.linalg.inv(S)

        # ログ保存
        self.K_log.append(K.squeeze())

        # イノベーション（観測残差）を計算
        innovation = y_arr - self.C @ x_minus

        # 事後推定値 (self.x) を更新
        # ヒント: 予測値 + ゲイン * 誤差
        self.x = x_minus + K * innovation

        # 事後誤差共分散 (self.x) を更新
        # ヒント: (1 - KC)P_minus の形です
        self.P = (np.eye(K.shape[0]) - K @ self.C) @ P_minus

        return self.x.squeeze(), self.P.squeeze()


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
N = 300  # ステップ数
u = 0  # 入力は今回0とする => 正規白色雑音

# 真のシステムのデータ生成 (Generative Process)
np.random.seed(42)  # 再現性のため
v = np.random.normal(0, np.sqrt(Q), N)  # システムノイズ
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

kf = ScalarKalmanFilter(A, B, C, Q, R, P_0, x_0)

# 結果保存用
x_est = np.zeros(N)
P_history = np.zeros(N)
x_est[0] = x_0
P_history[0] = P_0

# フィルタリングループ
for k in range(1, N):
    x_hat, P_hat = kf.step(y_obs[k], u)
    x_est[k] = x_hat
    P_history[k] = P_hat

# ==========================================
# 4. 可視化
# ==========================================
plt.figure(figsize=(12, 8))

# 状態推定のプロット
plt.subplot(2, 1, 1)
plt.plot(x_true, "k-", label="True State (x)", alpha=0.6)  # 真値
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
plt.plot(kf.K_log, "b-")
plt.title("Kalman Gain Evolution")
plt.xlabel("Step")
plt.ylabel("Gain K")
plt.grid()

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "ex6_1_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()
