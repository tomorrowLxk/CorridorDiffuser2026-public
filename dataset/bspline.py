import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# -------------------- 基础：B样条评估（不依赖 SciPy） --------------------

def _open_uniform_knot_vector(n_ctrl, degree):
    """
    生成开式均匀结向量（clamped），区间规范化到 [0, 1]
    n_ctrl: 控制点个数
    degree: 阶数（3=三次）
    """
    k = degree
    m = n_ctrl + k + 1  # knots 数
    # 端点重复 k+1 次
    knots = np.zeros(m)
    knots[-(k+1):] = 1.0
    if m > 2*(k+1):
        inner_count = m - 2*(k+1)
        if inner_count > 0:
            inner = np.linspace(0, 1, inner_count+2)[1:-1]
            knots[k+1: k+1+inner_count] = inner
    return knots

def _cox_de_boor_basis(i, k, u, knots):
    """
    Cox–de Boor 递推，返回标量 N_{i,k}(u)
    """
    if k == 0:
        return 1.0 if (knots[i] <= u < knots[i+1] or (u == knots[-1] and knots[i] <= u <= knots[i+1])) else 0.0
    denom1 = knots[i+k] - knots[i]
    denom2 = knots[i+k+1] - knots[i+1]
    term1 = 0.0
    term2 = 0.0
    if denom1 > 0:
        term1 = (u - knots[i]) / denom1 * _cox_de_boor_basis(i, k-1, u, knots)
    if denom2 > 0:
        term2 = (knots[i+k+1] - u) / denom2 * _cox_de_boor_basis(i+1, k-1, u, knots)
    return term1 + term2

def _bspline_eval(points, degree, knots, u_vals):
    """
    评估二维 B样条：points [N,2], degree k, knots [N+k+1], u_vals [M]
    返回形状 [M,2]
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    M = len(u_vals)
    out = np.zeros((M, 2), dtype=float)
    for j, u in enumerate(u_vals):
        # 稀疏非零基函数范围：i in [0, n-1]
        val = np.zeros(2, dtype=float)
        for i in range(n):
            Nij = _cox_de_boor_basis(i, degree, u, knots)
            if Nij != 0.0:
                val += Nij * points[i]
        out[j] = val
    return out

# -------------------- 实用函数 --------------------
def _first_sample_collision(samples, obstacles_polys):
    """
    逐样本快速检测第一个碰撞点的索引；无碰撞返回 None
    """
    for j, s in enumerate(samples):
        if j == 0:
            continue
        S = Point(s)
        for poly in obstacles_polys:
            if poly.contains(S):
                return j
    return None

# -------------------- 主函数：生成无碰撞 B样条中心线 --------------------

def generate_Bspline_route(new_map, topo_route, new_gvd_map, gvd_route,
                           degree=3,
                           sample_ds=0.3,
                           max_insertions=25,
                           corridor_margin=0.1):

    # 1) 从 topo_route 提取初始控制点
    nodes = new_gvd_map['nodes']
    old_point_indices = [0]  # ← 这就是你要的新列表
    ctrl = [np.array(nodes[topo_route[0]])]
    # for nid in topo_route:
    #     if nid not in nodes:
    #         raise KeyError(f"Node id {nid} not in new_gvd_map['nodes']")
    #     ctrl.append(np.array(nodes[nid], dtype=float))

    # 构建原始线段的 LineString 列表，用于判断采样点在哪一段上
    segments_length = [0]
    for i in range(len(gvd_route) - 1):
        seg = LineString([gvd_route[i], gvd_route[i + 1]])
        segments_length.append(seg.length+segments_length[-1])

    # 再额外添加等间距点作为控制点
    all_line = LineString(gvd_route)
    total_length = all_line.length
    distances = np.arange(0.1, total_length, 4)
    sampled_points = []
    for distance in distances:
        segment_index = None
        for i in range(len(segments_length) - 1):
            if segments_length[i] <= distance < segments_length[i + 1]:
                segment_index = i  # 落在第 i 段，即从旧点 i 到 i+1
                break
        if segment_index is None:
            segment_index = len(segments_length) - 2
        old_point_indices.append(segment_index)
        point = all_line.interpolate(distance)
        sampled_points.append([point.x, point.y])
    for point in sampled_points:
        ctrl.append(np.array(point, dtype=float))
    ctrl.append(np.array(nodes[topo_route[-1]]))
    control_points = np.vstack(ctrl)

    # 2) 预备 shapely 障碍物
    obstacles_polys = [Polygon(obs) for obs in new_map.obstacles]
    if corridor_margin > 1e-6:
        obstacles_polys = [poly.buffer(corridor_margin) for poly in obstacles_polys]
    obstacles_union = unary_union(obstacles_polys) if len(obstacles_polys) > 0 else None

    # 3) 迭代修复
    insertions = 0
    while True:
        n_ctrl = len(control_points)
        if n_ctrl < 2:
            return control_points.copy(), control_points.copy()

        k = min(degree, n_ctrl - 1)
        knots = _open_uniform_knot_vector(n_ctrl, k)

        # 采样曲线
        ctrl_len = np.sum(np.linalg.norm(control_points[1:] - control_points[:-1], axis=1))
        M = max(50, int(math.ceil(ctrl_len / max(sample_ds, 1e-2))) + 1)
        u_vals = np.linspace(0.0, 1.0, M)
        samples = _bspline_eval(control_points, k, knots, u_vals)

        # 碰撞检测
        if obstacles_union is None:
            break
        idx_hit = _first_sample_collision(samples, obstacles_polys)
        if idx_hit is None:
            break

        # 4) 碰撞 -> 从 gvd_route 找最近的**未使用**点作为新控制点
        p_hit = samples[idx_hit]
        gvd_pts = np.array(gvd_route)
        dists = np.linalg.norm(gvd_pts - p_hit, axis=1)
        sorted_indices = np.argsort(dists)  # 按距离升序排列索引

        new_cp = None
        gvd_idx = None
        insert_idx = None

        # 尝试每一个候选点（从最近开始）
        for cand_idx in sorted_indices:
            candidate = gvd_pts[cand_idx]

            # 跳过已存在的控制点
            if any(np.allclose(candidate, cp, atol=1e-6) for cp in control_points):
                continue

            # 找到第一个未使用的候选点
            new_cp = candidate
            gvd_idx = cand_idx

            # ---- 找插入位置（保持 gvd_route 顺序）----
            insert_idx = None
            for d in range(len(old_point_indices)):
                if old_point_indices[d]>=gvd_idx:
                    insert_idx = d
                    break
            if insert_idx is None:
                insert_idx = len(old_point_indices)
            break

        # 如果没有找到可用的新控制点
        if new_cp is None:
            insertions += 1
            if insertions >= max_insertions:
                break
            else:
                continue  # 继续下一次迭代，也许后续采样能避开

        # 插入新控制点
        control_points = np.insert(control_points, insert_idx, new_cp, axis=0)
        old_point_indices = np.insert(old_point_indices, insert_idx, gvd_idx)
        insertions += 1

        if insertions >= max_insertions:
            break

    # 5) 生成最终中心线
    n_ctrl = len(control_points)
    k = min(degree, n_ctrl - 1)
    knots = _open_uniform_knot_vector(n_ctrl, k)
    ctrl_len = np.sum(np.linalg.norm(control_points[1:] - control_points[:-1], axis=1))
    M = max(50, int(math.ceil(ctrl_len / max(sample_ds, 1e-2))) + 1)
    u_vals = np.linspace(0.0, 1.0, M)
    centerline = _bspline_eval(control_points, k, knots, u_vals)

    return np.array(centerline), control_points

