import numpy as np


def KF_M_estimator_based(X, SIGMA, Z_input, R_input):
    """
    加权迭代卡尔曼滤波，（没用扩展因为是线性）
    :param X: 输入待融合数据X，行向量
    :param SIGMA: X与真实值间的误差的分布的协方差矩阵
    :param Z_input: 输入待融合数据Z，行向量
    :param R_input: X与真实值间的误差的分布的协方差矩阵
    :return: 融合结果行向量
    """

    def psi(x):
        threshold = 1.345
        if x == 0:
            return 1
        return np.clip(x, -threshold, threshold) / x

    def PSI(X, X_now):
        zeta = X - X_now
        PSI = np.diag([psi(_) for _ in zeta.flat])
        return np.matrix(PSI)

    X_pre = np.matrix(X).T
    Z = np.matrix(Z_input).T
    SIGMA_pre = np.matrix(SIGMA)
    R = np.matrix(R_input)
    PSI_X_now = PSI(X_pre, X_pre)
    PSI_Z_now = PSI(Z, Z)

    def K_now():
        K_now = np.pow(SIGMA_pre, 1 / 2) * PSI_X_now.I * np.pow(SIGMA_pre, 1 / 2) * (
                np.pow(SIGMA_pre, 1 / 2) * PSI_X_now.I * np.pow(SIGMA_pre, 1 / 2) +
                np.pow(R, 1 / 2) * PSI_Z_now.I * np.pow(R, 1 / 2)).I
        return K_now

    X_now = X_pre + K_now() * (Z - X_pre)

    while (1):
        X_last = X_now
        PSI_X_now = PSI(X_pre, X_now)
        PSI_Z_now = PSI(Z, X_now)
        X_now = X_pre + K_now() * (Z - X_pre)
        zeta = np.abs(X_last - X_now)
        enough = zeta[zeta > 1e-6].size == 0
        if enough:
            break

    return X_now