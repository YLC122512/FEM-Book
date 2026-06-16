import numpy as np
import json
from ldlt import ldlt_factor, ldlt_solve
from solver_interface import solve_ldlt, solve_sparse
from utils import residual_norm, condition_number, timer
import poisson


def example_2_3_truss1():
    """复用 2.3 算例1：一维杆结构"""
    print("\n===== 2.3 算例1：一维杆结构 =====")
    K_full = np.array([[100, -100, 0], [-100, 300, -200], [0, -200, 200]], dtype=float)
    # 已知 d1=0, 载荷 F3=10
    free_dofs = [1, 2]  # 0-based: d2, d3
    fixed_dofs = [0]
    d_known = np.array([0.0])
    F_full = np.array([0, 0, 10.0])
    K_FF = K_full[np.ix_(free_dofs, free_dofs)]
    K_FE = K_full[np.ix_(free_dofs, fixed_dofs)]
    rhs = F_full[free_dofs] - K_FE @ d_known
    # LDLT 求解
    x_F = solve_ldlt(K_FF, rhs)
    d = np.zeros(3)
    d[free_dofs] = x_F
    d[fixed_dofs] = d_known
    print("位移解:", d)
    # 反力
    reactions = K_full @ d - F_full
    print("反力:", reactions)
    # 残差
    r_norm, rel_res = residual_norm(K_FF, x_F, rhs)
    print(f"相对残差: {rel_res:.2e}")


def example_2_3_truss2():
    """复用 2.3 算例2：二维桁架"""
    print("\n===== 2.3 算例2：二维桁架 =====")
    # 从 2.3 组装结果已知的 K_FF 和 rhs（为简化，直接使用预期矩阵）
    # 实际应调用 2.3 的组装模块，这里直接给出数值
    K_FF = np.array([[0.35355339, 0.35355339],
                     [0.35355339, 1.35355339]])
    rhs = np.array([10.0, 0.0])
    x = solve_ldlt(K_FF, rhs)
    print("节点3位移 (u, v):", x)
    # 应力等可复用 2.3 后处理，略


def example_tridiagonal(n):
    """三对角对称正定矩阵算例"""
    print(f"\n===== 三对角矩阵 n={n} =====")
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0: K[i, i - 1] = -1.0
        if i < n - 1: K[i, i + 1] = -1.0
    x_exact = np.ones(n)
    b = K @ x_exact
    # LDLT 求解
    x = solve_ldlt(K, b)
    error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    r_norm, rel_res = residual_norm(K, x, b)
    cond = condition_number(K)
    print(f"相对误差: {error:.2e}, 相对残差: {rel_res:.2e}, 条件数: {cond:.2e}")


def test_non_positive_definite():
    """非正定矩阵检测"""
    print("\n===== 非正定矩阵测试 =====")
    K = np.array([[1, 2], [2, 1]])
    b = np.array([1, 1])
    try:
        solve_ldlt(K, b)
    except ValueError as e:
        print("正确捕获错误:", e)


def ill_conditioned_example():
    """病态矩阵误差分析（任务2）"""
    print("\n===== 病态矩阵误差分析 =====")
    K = np.array([[1.0, 1.0], [1.0, 1.0001]])
    x_exact = np.array([1.0, 1.0])
    b = K @ x_exact
    cond = condition_number(K)
    print(f"条件数: {cond:.2e}")

    # 双精度求解
    x_double = solve_ldlt(K, b)
    err_double = np.linalg.norm(x_double - x_exact) / np.linalg.norm(x_exact)
    r_double, rel_r_double = residual_norm(K, x_double, b)
    print(f"双精度: 相对误差={err_double:.2e}, 相对残差={rel_r_double:.2e}")

    # 模拟4位有效数字舍入（人为降低精度）
    K4 = np.round(K, 4)
    b4 = np.round(b, 4)
    try:
        x4 = solve_ldlt(K4, b4)
        err4 = np.linalg.norm(x4 - x_exact) / np.linalg.norm(x_exact)
        r4, rel_r4 = residual_norm(K4, x4, b4)
        print(f"4位有效数字: 相对误差={err4:.2e}, 相对残差={rel_r4:.2e}")
    except ValueError:
        print("4位精度下矩阵可能非正定")

    # 说明：残差小但误差大是因为条件数大，即使微小舍入也会放大误差
    print("结论：条件数大时，小残差不能保证解准确。")


def large_poisson(nx, ny, solver='scipy'):
    """大规模 Poisson 方程有限元求解（任务3）"""
    print(f"\n===== Poisson 方程 Q4 单元网格 {nx}x{ny} =====")

    def f_func(x, y):
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    K, F, coords, _ = poisson.assemble_poisson_q4(nx, ny, f_func)
    print(f"矩阵规模: {K.shape[0]}, 非零元数: {K.nnz}")
    # 调用稀疏求解器
    import time
    start = time.perf_counter()
    if solver == 'scipy':
        from scipy.sparse.linalg import spsolve
        u = spsolve(K, F)
    elif solver == 'pardiso':
        from pypardiso import spsolve as pypardiso_solve
        u = pypardiso_solve(K, F)
    else:
        u = solve_ldlt(K.toarray(), F)  # 仅小规模可用
    end = time.perf_counter()
    print(f"求解耗时: {end - start:.3f} 秒")
    # 计算理论解
    x = coords[:, 0];
    y = coords[:, 1]
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    # 误差
    max_err = np.max(np.abs(u - u_exact))
    l2_err = np.sqrt(np.sum((u - u_exact) ** 2) / np.sum(u_exact ** 2))
    rel_res = residual_norm(K, u, F)[1]
    print(f"最大节点误差: {max_err:.4e}, 相对L2误差: {l2_err:.4e}, 相对残差: {rel_res:.2e}")
    return u, u_exact, coords


if __name__ == "__main__":
    # 算例0: 2.3 算例
    example_2_3_truss1()
    example_2_3_truss2()
    # 算例1: 三对角
    for n in [10, 100]:
        example_tridiagonal(n)
    # 算例2: 非正定检测
    test_non_positive_definite()
    # 任务2: 病态矩阵
    ill_conditioned_example()
    # 任务3: 大规模 Poisson（小规模测试，可增大nx,ny）
    large_poisson(20, 20, solver='scipy')