import numpy as np


class LinearKalmanFilter:
    def __init__(
        self,
        A,
        B,
        C,
        Q,
        R,
        P_init,
        x_init,
    ):
        """
        線形カルマンフィルタ
        """
        # スカラが渡っても行列計算できるように 2 次元配列へ正規化する
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))  # プロセスノイズ共分散
        self.R = np.atleast_2d(np.asarray(R, dtype=float))  # 観測ノイズ共分散

        self.P = np.atleast_2d(np.asarray(P_init, dtype=float))  # 事後誤差共分散
        self.x = np.atleast_2d(np.asarray(x_init, dtype=float))  # 事後推定値

        self.K_log = []

    def step(self, y):
        """
        1 ステップのカルマンフィルタ更新を行う。
        y: 観測値
        return: 事後推定値 x, 共分散 P, 事前推定値 x_minus
        """
        y_arr = np.atleast_2d(np.asarray(y, dtype=float))

        # 時間更新（事前推定）
        x_minus = self.A @ self.x
        P_minus = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T

        # 観測更新
        S = self.C @ P_minus @ self.C.T + self.R
        K = (P_minus @ self.C.T) @ np.linalg.inv(S)  # カルマンゲイン
        self.K_log.append(K.squeeze())

        innovation = y_arr - self.C @ x_minus
        self.x = x_minus + K @ innovation
        self.P = (np.eye(self.A.shape[0]) - K @ self.C) @ P_minus

        return self.x.squeeze(), self.P.squeeze(), x_minus.squeeze()
