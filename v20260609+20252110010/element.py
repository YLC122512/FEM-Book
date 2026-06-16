# element.py
"""
单元分析：长度、方向余弦、单元刚度矩阵、应力计算
"""
import numpy as np


def element_ke(nsd, E, A, L, c, s):
    """
    计算单元刚度矩阵
    对于一维杆: nsd=1, c=1, s=0
    对于二维桁架: nsd=2
    返回 (nen*ndof, nen*ndof) 的矩阵
    """
    k = E * A / L
    if nsd == 1:
        # 局部自由度顺序 [u1, u2]
        Ke = k * np.array([[1, -1],
                           [-1, 1]])
    else:  # nsd == 2
        # 局部自由度顺序 [u1, v1, u2, v2]
        c2 = c * c
        s2 = s * s
        cs = c * s
        Ke = k * np.array([[c2, cs, -c2, -cs],
                           [cs, s2, -cs, -s2],
                           [-c2, -cs, c2, cs],
                           [-cs, -s2, cs, s2]])
    return Ke


def element_stress(nsd, E, L, c, s, de):
    """
    计算单元轴向应力 sigma = E/L * [-c, -s, c, s] * de
    de: 单元节点位移向量 (顺序 u1,v1,u2,v2 或 u1,u2)
    返回应力值 (标量)
    """
    if nsd == 1:
        B = np.array([-1, 1]) / L
    else:
        B = np.array([-c, -s, c, s]) / L
    sigma = E * B @ de
    return sigma