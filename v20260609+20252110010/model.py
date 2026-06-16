# model.py
"""
模型数据管理：节点、单元、材料、边界条件、载荷
"""
import numpy as np


class Model:
    def __init__(self, data):
        self.title = data.get("Title", "Truss Model")
        self.nsd = data["nsd"]          # 空间维度 (1 或 2)
        self.ndof = data["ndof"]        # 每节点自由度数 (应与 nsd 一致)
        self.nnp = data["nnp"]          # 节点总数
        self.nel = data["nel"]          # 单元总数
        self.nen = data["nen"]          # 每单元节点数 (此处恒为2)

        # 材料与截面
        self.E = np.array(data["E"])          # 弹性模量，长度 nel
        self.A = np.array(data["CArea"])      # 截面积，长度 nel

        # 节点坐标
        self.x = np.array(data["x"])           # shape (nnp,)
        if self.nsd == 1:
            self.coords = self.x.reshape(-1, 1)
        else:
            self.y = np.array(data["y"])
            self.coords = np.vstack((self.x, self.y)).T   # shape (nnp, nsd)

        # 单元连接关系 (1-indexed)
        self.IEN = np.array(data["IEN"]) - 1   # 转为 0-indexed，shape (nel, nen)

        # 边界条件与载荷 (1-indexed)
        fixed_dof = np.array(data["fixed_dof"]) - 1
        fixed_val = np.array(data["fixed_value"])
        self.fixed_dof = fixed_dof
        self.fixed_val = fixed_val

        force_dof = np.array(data["force_dof"]) - 1
        force_val = np.array(data["force_value"])
        self.force_dof = force_dof
        self.force_val = force_val

        # 总自由度数
        self.ndof_total = self.nnp * self.ndof

        # 预计算单元长度和方向余弦 (供组装和后处理使用)
        self.element_lengths = np.zeros(self.nel)
        self.element_cos = np.zeros(self.nel)
        self.element_sin = np.zeros(self.nel)
        for e in range(self.nel):
            n1, n2 = self.IEN[e]
            coord1 = self.coords[n1]
            coord2 = self.coords[n2]
            dx = coord2 - coord1
            self.element_lengths[e] = np.linalg.norm(dx)
            if self.nsd == 1:
                self.element_cos[e] = 1.0
                self.element_sin[e] = 0.0
            else:
                L = self.element_lengths[e]
                self.element_cos[e] = dx[0] / L if L > 0 else 0
                self.element_sin[e] = dx[1] / L if L > 0 else 0

    def get_free_dofs(self):
        """返回所有未知自由度的全局编号 (0-indexed)"""
        all_dofs = np.arange(self.ndof_total)
        free_dofs = np.setdiff1d(all_dofs, self.fixed_dof)
        return free_dofs

    def print_summary(self):
        print(f"\n===== 模型信息: {self.title} =====")
        print(f"维度: {self.nsd}D, 节点数: {self.nnp}, 单元数: {self.nel}")
        print(f"总自由度数: {self.ndof_total}")

    def print_displacements(self, d):
        """打印节点位移"""
        for node in range(self.nnp):
            start = node * self.ndof
            end = start + self.ndof
            disp = d[start:end]
            if self.nsd == 1:
                print(f"  节点 {node+1}: u = {disp[0]:.6f}")
            else:
                print(f"  节点 {node+1}: u = {disp[0]:.6f}, v = {disp[1]:.6f}")

    def print_reactions(self, reactions):
        """打印约束反力 (仅输出 fixed_dof 上的值)"""
        for idx, dof in enumerate(self.fixed_dof):
            node = dof // self.ndof
            local = dof % self.ndof
            if self.nsd == 1:
                print(f"  自由度 {dof+1} (节点{node+1}): R = {reactions[dof]:.6f}")
            else:
                dir_name = "u" if local == 0 else "v"
                print(f"  自由度 {dof+1} (节点{node+1},{dir_name}): R = {reactions[dof]:.6f}")