import json
import numpy as np
from model import Model
from assembly import Assembly
from solver import Solver
from postprocess import PostProcessor

def main():
    # 切换算例时仅需修改此处文件名
    input_file = "input_2d.json"   # 或 "input_1d.json"

    with open(input_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    model = Model(data)
    model.print_summary()

    assembler = Assembly(model)
    K, F = assembler.assemble()
    print("\n总体刚度矩阵 K (组装后):")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"K是否对称: {np.allclose(K, K.T)}")

    solver = Solver(model, K, F)
    d, reactions = solver.solve_reduced()
    print("\n求解结果:")
    print("节点位移:")
    model.print_displacements(d)
    print("约束反力:")
    model.print_reactions(reactions)

    post = PostProcessor(model, d)
    post.compute_all_elements()
    post.print_results()

    print("\n刚度矩阵性质分析:")
    print(f"  行列式(处理边界前): {np.linalg.det(K):.2e}")
    free_dofs = model.get_free_dofs()
    if len(free_dofs) > 0:
        K_FF = K[np.ix_(free_dofs, free_dofs)]
        print(f"  缩减后行列式: {np.linalg.det(K_FF):.2e}")
    print("  对角元非负: ", np.all(np.diag(K) >= 0))
    print("  稀疏性: 非零元素比例 {:.2%}".format(np.count_nonzero(K) / K.size))

if __name__ == "__main__":
    main()