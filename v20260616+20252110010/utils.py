import numpy as np
import time

def residual_norm(K, x, b):
    """计算相对残差 ||b - K x|| / ||b||"""
    r = b - K @ x
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    return norm_r, norm_r / norm_b if norm_b > 0 else np.inf

def condition_number(K, p=2):
    """条件数（基于特征值）"""
    eigvals = np.linalg.eigvalsh(K)
    return eigvals.max() / eigvals.min()

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end-start:.6f} 秒")
        return result
    return wrapper