import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    # x1, x2: list or array of [x, y, z]
    # returns L, direction_cosines, Ke
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("Error: Degenerate element, two nodes coincide.")
    cx, cy, cz = delta / L
    # transformation matrix T (6x6) for truss: 方向余弦阵
    # 实际上对于杆单元，刚度矩阵 Ke = k * T, where T = [cx^2, cx*cy, cx*cz, -cx^2, -cx*cy, -cx*cz; ...]
    # 更常见的是 Ke = (E*A/L) * [C, -C; -C, C] where C = [cx^2, cx*cy, cx*cz; cx*cy, cy^2, cy*cz; cx*cz, cy*cz, cz^2]
    C = np.array([[cx*cx, cx*cy, cx*cz],
                  [cx*cy, cy*cy, cy*cz],
                  [cx*cz, cy*cz, cz*cz]])
    k = E * A / L
    Ke = k * np.block([[C, -C], [-C, C]])
    return L, (cx, cy, cz), Ke

def truss3d_element_stress(x1, x2, E, A, de):
    # de: list or array of [u1, v1, w1, u2, v2, w2]
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    de = np.array(de, dtype=float)
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("Error: Degenerate element, two nodes coincide.")
    cx, cy, cz = delta / L
    # 轴向应变 epsilon = (delta_u)/L = ( (u2-u1)*cx + (v2-v1)*cy + (w2-w1)*cz ) / L
    # 或者通过 B matrix: B = 1/L * [-cx, -cy, -cz, cx, cy, cz]
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = np.dot(B, de)
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N

# 验证算例1
print("===== 算例1: 沿x轴的一维杆单元 =====")
x1 = [0,0,0]
x2 = [2,0,0]
E = 200e9  # Pa
A = 1.0e-4 # m^2
L, (cx,cy,cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
print(f"单元长度 L = {L} m")
print(f"方向余弦: cx={cx}, cy={cy}, cz={cz}")
print("刚度矩阵 Ke (6x6):")
print(Ke)
de = [0,0,0, 1.0e-3,0,0]
epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
print(f"轴向应变 epsilon = {epsilon}")
print(f"轴向应力 sigma = {sigma/1e6} MPa")
print(f"轴力 N = {N} N")
# 验证期望
print("\n验证结果:")
print(f"长度应为2m: {L} -> {'OK' if abs(L-2)<1e-9 else 'FAIL'}")
print(f"方向余弦应为(1,0,0): ({cx},{cy},{cz}) -> OK" if abs(cx-1)<1e-9 and abs(cy)<1e-9 and abs(cz)<1e-9 else "FAIL")
print(f"应变应为5e-4: {epsilon} -> OK" if abs(epsilon-5e-4)<1e-9 else "FAIL")
print(f"应力应为100MPa: {sigma/1e6} MPa -> OK" if abs(sigma-100e6)<1e6 else "FAIL")
print(f"轴力应为1e4N: {N} N -> OK" if abs(N-10000)<1e-9 else "FAIL")

# 算例2
print("\n===== 算例2: 空间任意方向杆单元 =====")
x1 = [0,0,0]
x2 = [1,2,2]
E = 210e9
A = 2.0e-4
L, (cx,cy,cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
print(f"单元长度 L = {L} m")
print(f"方向余弦: cx={cx}, cy={cy}, cz={cz}")
print("刚度矩阵 Ke (6x6):")
print(Ke)
# 检查对称性
sym_error = np.linalg.norm(Ke - Ke.T)
print(f"Ke对称性检查 (范数差): {sym_error} (应为0)")
# 特征值
eigvals = np.linalg.eigvalsh(Ke)
print(f"Ke的特征值: {eigvals}")
print(f"最小特征值: {np.min(eigvals)} (应 >=0)")
# 刚体平移测试
de_translation = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]  # 所有节点相同位移
eps_trans, sig_trans, N_trans = truss3d_element_stress(x1, x2, E, A, de_translation)
print(f"刚体平移位移产生的应变: {eps_trans} (应为0)")
de = [0,0,0, 1.0e-3, 2.0e-3, 2.0e-3]
epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
print(f"轴向应变 epsilon = {epsilon}")
print(f"轴向应力 sigma = {sigma/1e6} MPa")
print(f"轴力 N = {N} N")
print("\n验证结果:")
print(f"长度应为3m: {L} -> OK" if abs(L-3)<1e-9 else "FAIL")
print(f"方向余弦应为(1/3,2/3,2/3): ({cx},{cy},{cz}) -> OK" if abs(cx-1/3)<1e-9 and abs(cy-2/3)<1e-9 and abs(cz-2/3)<1e-9 else "FAIL")
print(f"应变应为1e-3: {epsilon} -> OK" if abs(epsilon-1e-3)<1e-9 else "FAIL")
print(f"应力应为210MPa: {sigma/1e6} MPa -> OK" if abs(sigma-210e6)<1e6 else "FAIL")
print(f"轴力应为4.2e4N: {N} N -> OK" if abs(N-42000)<1e-9 else "FAIL")
print(f"刚体平移应变接近0: {eps_trans} -> OK" if abs(eps_trans)<1e-12 else "FAIL")

# 任务4：刚度矩阵物理意义
print("\n===== 任务4: 刚度矩阵物理意义验证 =====")
# 选自由度j=1 (第一个自由度位移为1)
de_j = np.zeros(6)
de_j[0] = 1.0  # u1=1, 其他为0
Fe = Ke @ de_j
print(f"Ke的第1列: {Ke[:,0]}")
print(f"Fe = Ke * de (de=[1,0,0,0,0,0]) = {Fe}")
print("解释: Fe即为刚度矩阵的第1列。k_{i1}表示当第1个自由度(u1)产生单位位移时，在第i个自由度上需要施加的节点力。")