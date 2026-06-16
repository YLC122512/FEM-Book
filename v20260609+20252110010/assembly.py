# assembly.py
"""
总体刚度矩阵组装：生成LM矩阵，直接组装K和F
"""
import numpy as np
from element import element_ke


class Assembly:
    def __init__(self, model):
        self.model = model
        self.LM = None          # 对号矩阵，shape (nel, nen*ndof)
        self._build_LM()

    def _build_LM(self):
        """根据IEN和ndof生成对号矩阵LM (0-indexed全局自由度编号)"""
        model = self.model
        nel = model.nel
        nen = model.nen
        ndof = model.ndof
        self.LM = np.zeros((nel, nen * ndof), dtype=int)
        for e in range(nel):
            n1, n2 = model.IEN[e]
            # 节点n1的自由度起始编号
            start1 = n1 * ndof
            start2 = n2 * ndof
            if ndof == 1:
                self.LM[e, :] = [start1, start2]
            else:  # ndof == 2
                self.LM[e, :] = [start1, start1+1, start2, start2+1]

    def assemble(self):
        """
        组装总体刚度矩阵 K 和力向量 F
        返回 (K, F)
        """
        model = self.model
        ndof_total = model.ndof_total
        K = np.zeros((ndof_total, ndof_total))
        F = np.zeros(ndof_total)

        # 施加节点载荷
        F[model.force_dof] = model.force_val

        # 遍历所有单元，直接组装
        for e in range(model.nel):
            E = model.E[e]
            A = model.A[e]
            L = model.element_lengths[e]
            c = model.element_cos[e]
            s = model.element_sin[e]
            Ke = element_ke(model.nsd, E, A, L, c, s)

            # 获取该单元的LM行
            lm = self.LM[e, :]
            # 直接组装 (散射累加)
            for i in range(len(lm)):
                for j in range(len(lm)):
                    K[lm[i], lm[j]] += Ke[i, j]

        return K, F