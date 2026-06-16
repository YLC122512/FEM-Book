import numpy as np

def ldlt_factor(K):
    """
    对对称正定矩阵 K 进行 LDL^T 分解，返回 L 和 D。
    使用标准的 LDL^T 分解公式（未修改原矩阵 K）。
    K: n x n 对称正定矩阵（numpy array）
    返回:
        L: 单位下三角矩阵 (n x n)
        D: 一维数组，对角元
    若遇到非正主元，抛出 ValueError。
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        # 计算 D[j]
        d = K[j, j]
        for k in range(j):
            d -= L[j, k]**2 * D[k]
        D[j] = d
        if D[j] <= 0:
            raise ValueError(f"非正主元：第 {j+1} 个主元 = {D[j]}，矩阵非正定")

        # 计算 L[i, j] for i > j
        for i in range(j+1, n):
            l = K[i, j]
            for k in range(j):
                l -= L[i, k] * D[k] * L[j, k]
            L[i, j] = l / D[j]

    return L, D

def ldlt_solve(L, D, b):
    """
    求解 LDL^T x = b
    L: 单位下三角矩阵（n x n）
    D: 对角元一维数组（长度 n）
    b: 右端项（长度 n）
    返回 x
    """
    n = len(b)
    # 前代：L y = b
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s

    # 对角求解：D z = y
    z = y / D

    # 回代：L^T x = z
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s += L[j, i] * x[j]   # L^T 的 (i,j) 元素 = L[j,i]
        x[i] = z[i] - s
    return x