import numpy as np
from ldlt import ldlt_factor, ldlt_solve

def solve_ldlt(K, rhs):
    """使用 LDL^T 求解 K x = rhs (K 对称正定)"""
    L, D = ldlt_factor(K)
    return ldlt_solve(L, D, rhs)

def solve_sparse(K, rhs, method='scipy'):
    """调用稀疏求解器（scipy 或 pypardiso）"""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    if method == 'scipy':
        K_sp = sp.csr_matrix(K)
        return spsolve(K_sp, rhs)
    elif method == 'pardiso':
        # 需要安装 pypardiso: pip install pypardiso
        try:
            from pypardiso import spsolve as pypardiso_solve
            return pypardiso_solve(K, rhs)
        except ImportError:
            raise ImportError("pypardiso 未安装，请使用 'scipy' 方法")
    else:
        raise ValueError(f"未知方法: {method}")