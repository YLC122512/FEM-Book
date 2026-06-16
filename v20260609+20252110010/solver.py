# solver.py
"""
方程求解：缩减法处理位移边界条件，计算位移和反力
"""
import numpy as np


class Solver:
    def __init__(self, model, K, F):
        self.model = model
        self.K = K
        self.F = F

    def solve_reduced(self):
        """
        缩减法求解：
        1. 分离已知位移自由度和未知位移自由度
        2. 求解 K_FF * d_F = F_F - K_FE * d_E
        3. 还原完整位移向量，计算反力 R = K * d - F
        """
        model = self.model
        fixed_dofs = model.fixed_dof
        fixed_vals = model.fixed_val
        free_dofs = model.get_free_dofs()

        # 分块矩阵
        K_FF = self.K[np.ix_(free_dofs, free_dofs)]
        K_FE = self.K[np.ix_(free_dofs, fixed_dofs)]
        K_EF = self.K[np.ix_(fixed_dofs, free_dofs)]
        K_EE = self.K[np.ix_(fixed_dofs, fixed_dofs)]

        F_F = self.F[free_dofs]
        F_E = self.F[fixed_dofs]

        d_E = fixed_vals
        # 求解未知位移
        RHS = F_F - K_FE @ d_E
        d_F = np.linalg.solve(K_FF, RHS)

        # 组装完整位移向量
        d = np.zeros(model.ndof_total)
        d[free_dofs] = d_F
        d[fixed_dofs] = d_E

        # 计算所有自由度的反力: R = K*d - F
        reactions = self.K @ d - self.F

        return d, reactions

    def solve_penalty(self, penalty=1e12):
        """
        罚函数法处理边界条件 (附加实现，用于对比)
        Kp = K + penalty * I_fixed, Fp = F + penalty * d_fixed
        """
        Kp = self.K.copy()
        Fp = self.F.copy()
        fixed_dofs = self.model.fixed_dof
        fixed_vals = self.model.fixed_val
        for i, dof in enumerate(fixed_dofs):
            Kp[dof, dof] += penalty
            Fp[dof] += penalty * fixed_vals[i]
        d = np.linalg.solve(Kp, Fp)
        reactions = Kp @ d - Fp
        return d, reactions