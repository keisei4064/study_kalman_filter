import numpy as np


def run_scalar_least_squares():
    print("=== 第4章 最小二乗推定法（スカラー）の実装 ===")

    # ---------------------------------------------------------
    # 1. 設定：物理量と観測の統計的性質 (p.61)
    # ---------------------------------------------------------

    # [式(4.1)]
    # 真の値 x の統計的性質
    # xに関する事前知識としての平均,分散
    #   E[x] = x_bar
    #   E[(x - x_bar)^2] = sigma_x_2
    x_bar = 25.0  # 平均: 例「今の室温はだいたい25度」
    sigma_x_2 = 9.0  # 分散: 「プラマイ3度くらいはずれるかも」

    # [式(4.2)]
    # 観測の方程式 y = cx + w
    c = 1.0  # 物理量から観測量への変換係数 (電圧への変換など)

    # [式(4.3)]
    # 観測雑音(センサーノイズ) w の統計的性質
    #   E[w] = w_bar
    #   E[(w - w_bar)^2] = sigma_w_2
    w_bar = 0.0  # 平均
    sigma_w_2 = 4.0  # 分散

    # ---------------------------------------------------------
    # 2. シミュレーション
    # ---------------------------------------------------------
    np.random.seed(42)

    # 真の値 x_true を生成 (分布 N(x_bar, sigma_x_2) からサンプリング)
    x_true = x_bar + np.random.normal(0, np.sqrt(sigma_x_2))

    # [式(4.2)]
    # 観測値 y を生成
    #   y = cx + w
    w_noise = w_bar + np.random.normal(0, np.sqrt(sigma_w_2))  # ノイズwをサンプリング
    y = c * x_true + w_noise

    print(f"真の値 (x_true): {x_true:.4f}")
    print(f"観測値 (y)     : {y:.4f}")
    print("-" * 30)

    # ---------------------------------------------------------
    # 3. 推定アルゴリズム
    # ---------------------------------------------------------

    # --- Step 1: 最適なゲイン alpha を求める ---
    # 教科書 p.62 [式(4.12)]
    numerator = c * sigma_x_2
    denominator = (c**2 * sigma_x_2) + sigma_w_2
    alpha_opt = numerator / denominator

    # 「カルマンゲイン」の原型 - 「自分の予測の分散(sigma_x)」と「センサーの分散(sigma_w)」のバランスで決まっている
    print(f"最適ゲイン (alpha): {alpha_opt:.4f}")

    # --- Step 2: 推定値 x_hat を計算する ---
    # 教科書 p.63 [式(4.14)]
    #   x_hat = x_bar + alpha * { y - (c * x_bar + w_bar) }

    # 「予測誤差（イノベーション，サプライズ）」を計算
    #   (c * x_bar + w_bar) は「観測の予測値」
    pred_error = y - (c * x_bar + w_bar)

    # 更新則を適用
    x_hat = x_bar + alpha_opt * pred_error

    print(f"事前知識 (x_bar)  : {x_bar:.4f} | 予測誤差: {abs(x_bar - x_true)}")
    print(f"推定値 (x_hat)    : {x_hat:.4f} | 予測誤差: {abs(x_hat - x_true)}")
    print("-" * 30)

    # ---------------------------------------------------------
    # 4. 事後評価：不確実性は減ったか？ (Posterior Variance)
    # ---------------------------------------------------------

    # [式(4.16)]
    # 事後推定誤差分散 sigma_hat_2
    sigma_hat_2 = 1.0 / ((1.0 / sigma_x_2) + (c**2 / sigma_w_2))

    print(f"事前の不確実性 (sigma_x^2): {sigma_x_2:.4f}")
    print(f"事後の不確実性 (sigma^2)  : {sigma_hat_2:.4f}")


if __name__ == "__main__":
    run_scalar_least_squares()
