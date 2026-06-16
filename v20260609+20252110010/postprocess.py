# postprocess.py
"""
后处理：计算每个单元的长度、方向余弦、应力、轴力，并输出
"""
import numpy as np
from element import element_stress


class PostProcessor:
    def __init__(self, model, d):
        self.model = model
        self.d = d          # 全局位移向量
        self.results = []   # 存储每个单元的结果字典

    def compute_all_elements(self):
        model = self.model
        for e in range(model.nel):
            n1, n2 = model.IEN[e]
            # 提取单元位移
            ndof = model.ndof
            if ndof == 1:
                de = np.array([self.d[n1*ndof], self.d[n2*ndof]])
            else:
                de = np.array([self.d[n1*ndof], self.d[n1*ndof+1],
                               self.d[n2*ndof], self.d[n2*ndof+1]])

            L = model.element_lengths[e]
            c = model.element_cos[e]
            s = model.element_sin[e]
            E = model.E[e]
            A = model.A[e]

            sigma = element_stress(model.nsd, E, L, c, s, de)
            force = sigma * A

            self.results.append({
                'element': e+1,
                'length': L,
                'cos': c,
                'sin': s,
                'stress': sigma,
                'axial_force': force,
                'displacements': de
            })

    def print_results(self):
        print("\n===== 单元结果 =====")
        for res in self.results:
            print(f"\n单元 {res['element']}:")
            print(f"  长度 = {res['length']:.6f}")
            print(f"  方向余弦 (c,s) = ({res['cos']:.6f}, {res['sin']:.6f})")
            print(f"  应力 sigma = {res['stress']:.6f}")
            print(f"  轴力 N = {res['axial_force']:.6f}")
            if self.model.ndof == 2:
                print(f"  单元位移: u1={res['displacements'][0]:.6f}, v1={res['displacements'][1]:.6f}, u2={res['displacements'][2]:.6f}, v2={res['displacements'][3]:.6f}")