import numpy as np


def run_multivariate_least_squares():
    print("=== 第4章 最小二乗推定法（多変数・ベクトル）の実装 ===")

    # ---------------------------------------------------------
    # 0. 次元の設定 (p.69)
    # ---------------------------------------------------------
    # x: n次元 (例: 位置と速度 n=2)
    # y: m次元 (例: GPSと速度計 m=2) ※教科書はn個の観測値としているが，一般には m != n でもOK
    n = 2
    m = 2

    # ---------------------------------------------------------
    # 1. 設定：物理量と観測の統計的性質
    # ---------------------------------------------------------

    # [式(4.39)] 事前知識 (Prior Belief)
    #   E[x] = x_bar (平均ベクトル)
    #   E[(x - x_bar)(x - x_bar)^T] = Sigma_x (共分散行列)

    # 例: [位置, 速度] = [10, 2] だと思っている
    x_bar = np.array([[10.0], [2.0]])

    # 分散共分散行列(自信のなさ)
    Sigma_x = np.array(
        [
            [5.0, 0.0],  # 位置は結構自信ない
            [0.0, 1.0],  # 速度はそこそこ自信ある
        ]
    )

    # [式(4.37)]
    # 観測方程式 y = Cx + w
    #   C: 観測行列 (n x m)
    #   今回は「状態がそのまま観測される」単位行列とする
    C = np.eye(m, n)

    # [式(4.40)]
    # 観測雑音 w の統計的性質
    #   E[w] = w_bar
    #   E[(w - w_bar)(w - w_bar)^T] = Sigma_w
    w_bar = np.zeros((m, 1))  # バイアスなし

    # センサーの共分散
    Sigma_w = np.array(
        [
            [0.5, 0.0],  # センサー1は優秀(0.5)
            [0.0, 4.0],  # センサー2はポンコツ(4.0) としてみる
        ]
    )

    # ---------------------------------------------------------
    # 2. シミュレーション
    # ---------------------------------------------------------
    np.random.seed(123)

    # 真の値 x_true を生成 (多変量正規分布から)
    x_true = np.random.multivariate_normal(x_bar.flatten(), Sigma_x).reshape(n, 1)

    # 観測値 y を生成 [式(4.37)]
    w_noise = np.random.multivariate_normal(w_bar.flatten(), Sigma_w).reshape(m, 1)
    y = C @ x_true + w_noise

    print("--- 真の値と観測値 ---")
    print(f"真値 x_true\t: {x_true.flatten()}")
    print(f"事前予測 x_bar\t: {x_bar.flatten()}")
    print(f"観測値 y\t: {y.flatten()}")
    print()

    # ---------------------------------------------------------
    # 3. 推定アルゴリズム
    # ---------------------------------------------------------

    # --- Step 1: 最適ゲイン行列 F を求める ---

    # [式(4.50)]
    #   A = (C * Sigma_x * C^T + Sigma_w)  : イノベーションの共分散 A
    A = C @ Sigma_x @ C.T + Sigma_w
    A_inv = np.linalg.inv(A)
    B_T = Sigma_x @ C.T

    # [式(4.55)]
    #   F = Sigma_x * C^T * (C * Sigma_x * C^T + Sigma_w)^-1
    F = B_T @ A_inv

    print("--- 推定ゲイン F [式(4.55)] ---")
    print(F)
    print("\t> センサー1(精度高)の成分をより重視している")

    # --- Step 2: 推定値 x_hat を計算する ---

    # [式(4.56)]
    #   x_hat = x_bar + F @ { y - (C @ x_bar + w_bar) }

    # 「予測誤差（イノベーション，サプライズ）」を計算
    pred_error = y - (C @ x_bar + w_bar)

    # 更新
    x_hat = x_bar + F @ pred_error

    print()
    print("--- 推定結果 ---")
    print(f"予測誤差 y - (c @ x_bar + w_bar): {pred_error.flatten()}")
    print(
        f"事前予測 x_bar:{x_bar.flatten()}\t\t| 真値との誤差ノルム: {np.linalg.norm(x_bar - x_true):.4f}"
    )
    print(
        f"事後推定 x_hat:{x_hat.flatten()}\t| 真値との誤差ノルム: {np.linalg.norm(x_hat - x_true):.4f}"
    )
    # print(f"真値との誤差ノルム: {np.linalg.norm(x_hat - x_true):.4f}")

    # --- Step 3: 推定分散共分散 P を計算する ---

    print()
    print("--- 事後誤差共分散行列 P の検証 ---")

    # [式(4.57)]
    # 方法A: 引き算形式
    #   P = Sigma_x - F @ C @ Sigma_x
    P_A = Sigma_x - F @ C @ Sigma_x

    print("形式A (引き算) [式(4.57)]:\n", P_A)

    # # [式(4.59)]
    # 方法B: Woodbury形式 (逆行列の和)
    #   P = (Sigma_x^-1 + C^T @ Sigma_w^-1 @ C)^-1
    #   「精度の加算」の形！

    Precision_prior = np.linalg.inv(Sigma_x)  # 事前精度
    Precision_sensory = np.linalg.inv(Sigma_w)  # 観測精度

    # 精度の更新: 事後精度 = 事前精度 + 感覚精度 (Cで射影)
    Precision_posterior = Precision_prior + C.T @ Precision_sensory @ C

    # P は事後精度の逆行列
    P_B = np.linalg.inv(Precision_posterior)

    print("形式B (Woodbury/精度加算) [式(4.59)]:\n", P_B)

    # 確認
    diff = np.linalg.norm(P_A - P_B)
    print(f"\nWoodburyの公式による一致確認 (誤差): {diff:.10f}\n")
    P = P_A

    # 対角成分（分散）が小さくなったか確認
    print("--- 分散の減少 ---")
    print(f"事前:\n{Sigma_x}")
    print(f"事後:\n{P_A}")
    print()

    # --- Step 4: 推定ゲイン F を P を用いても計算してみる ---
    print("--- 推定ゲイン F (Pを用いて) [式(4.62)] ---")
    F_usingP = P @ C.T @ np.linalg.inv(Sigma_w)
    print(f"F (using P calc):\n{F_usingP}")
    print(f"F - F_usingP norm: {np.linalg.norm(F - F_usingP)}")

if __name__ == "__main__":
    run_multivariate_least_squares()
