import numpy as np
import scipy.sparse as sp


def assemble_poisson_q4(nx, ny, f_func):
    """
    在单位正方形 [0,1]x[0,1] 上使用双线性 Q4 单元求解 -Δu = f，零 Dirichlet 边界。
    nx, ny: 每个方向的单元数（网格为 nx * ny 个单元）
    f_func: 右端项函数 f(x,y)
    返回稀疏刚度矩阵 K 和载荷向量 F（未处理边界）
    """
    n_nodes = (nx + 1) * (ny + 1)
    # 节点坐标 (x,y)
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    # 单元连接（局部节点顺序：左下、右下、右上、左上）
    IEN = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            IEN.append([n0, n1, n2, n3])

    # 高斯积分点（2x2）
    gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    gauss_w = [1.0, 1.0]

    # 单元刚度矩阵和载荷向量
    K = sp.lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    for e, nodes in enumerate(IEN):
        xe = coords[nodes, 0]
        ye = coords[nodes, 1]
        Ke = np.zeros((4, 4))
        Fe = np.zeros(4)

        for xi_idx, xi in enumerate(gauss_pts):
            for eta_idx, eta in enumerate(gauss_pts):
                w = gauss_w[xi_idx] * gauss_w[eta_idx]
                # 形函数及导数（对于双线性单元）
                N = np.array(
                    [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
                dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
                dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
                # Jacobian
                dx_dxi = np.dot(dN_dxi, xe)
                dy_dxi = np.dot(dN_dxi, ye)
                dx_deta = np.dot(dN_deta, xe)
                dy_deta = np.dot(dN_deta, ye)
                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
                invJ = np.array([[dy_deta, -dx_deta], [-dy_dxi, dx_dxi]]) / detJ
                # 全局导数
                dN_dx = np.zeros(4)
                dN_dy = np.zeros(4)
                for i in range(4):
                    dN_dx[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
                    dN_dy[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
                # 刚度贡献
                Ke += (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)) * detJ * w
                # 载荷贡献
                xp = np.dot(N, xe)
                yp = np.dot(N, ye)
                Fe += N * f_func(xp, yp) * detJ * w

        # 组装
        for i in range(4):
            gi = nodes[i]
            F[gi] += Fe[i]
            for j in range(4):
                gj = nodes[j]
                K[gi, gj] += Ke[i, j]

    # 处理零 Dirichlet 边界（边界节点自由度置0）
    boundary_nodes = []
    for i in range(n_nodes):
        x, y = coords[i]
        if x == 0 or x == 1 or y == 0 or y == 1:
            boundary_nodes.append(i)
    # 强制边界位移为0（修改 K 和 F）
    for node in boundary_nodes:
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1.0
        F[node] = 0.0

    return K.tocsr(), F, coords, boundary_nodes