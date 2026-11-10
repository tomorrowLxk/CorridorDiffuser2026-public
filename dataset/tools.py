import numpy as np
import math
from copy import deepcopy
from shapely.geometry import Polygon, box, Point, LineString
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
from scipy.spatial import Voronoi
import heapq


# TODO 地图转换部分
# ---------------------------------------------------------------
# 将标准地图转化为栅格地图
def ConMap2RasMap(map_obj, resolution=0.2, dtype=np.uint8):
    """
    将 ContinuousMap 转换为二值栅格图
    输出: occupancy_grid[H, W], True 表示被占据（障碍物或边界）
    """
    W = int(np.round(map_obj.width / resolution))
    H = int(np.round(map_obj.height / resolution))
    grid = np.zeros((H, W), dtype=dtype)

    # 1. 填充边界（margin 区域视为障碍）
    margin_px = int(np.round(map_obj.margin / resolution))
    grid[:margin_px, :] = 1  # 上下
    grid[-margin_px:, :] = 1
    grid[:, :margin_px] = 1  # 左右
    grid[:, -margin_px:] = 1

    # 2. 光栅化每个障碍物多边形
    for obs in map_obj.obstacles:
        # 将多边形顶点转换为像素坐标
        px_coords = [(int(np.round(x / resolution)), int(np.round(y / resolution))) for x, y in obs]
        # 创建多边形对象（用于精确填充）
        poly1 = Polygon(obs)
        poly2 = Polygon(px_coords)

        # 获取包围盒
        minx, miny, maxx, maxy = map(int, poly2.bounds)
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, W - 1)
        maxy = min(maxy, H - 1)

        # 扫描线填充：检查每个像素中心是否在多边形内
        for px in range(minx, maxx + 1):
            for py in range(miny, maxy + 1):
                # 像素中心的世界坐标
                wx = (px + 0.5) * resolution
                wy = (py + 0.5) * resolution
                if poly1.contains(Point(wx, wy)):
                    grid[py, px] = 1  # 注意：numpy 是 row-major，py 对应 H

    grid = np.clip(grid, 0, 255).astype(dtype)
    return grid


# 构建起终点的栅格地图
def Point2Raster(map_obj, veh, start_pose, goal_pose, resolution=0.2, dtype=np.uint8):
    """
    将起点/终点位姿栅格化，考虑车辆尺寸和朝向，用等腰三角形表示。
    start_pose, goal_pose: (x, y, theta) in world frame
    veh: dict, {"length": L, "width": W}
    输出: grid[H, W, 2]，通道0=起点，通道1=终点
    """
    if map_obj is None:
        W, H = 256, 256
    else:
        W = int(np.round(map_obj.width / resolution))
        H = int(np.round(map_obj.height / resolution))
    grid = np.zeros((H, W, 2), dtype=dtype)

    def draw_vehicle_pose(grid, pose, channel):
        x, y, theta = pose

        # 三角形三个顶点（世界坐标）
        # 顶点（指向航向角）
        tip = np.array([x + (veh.L + veh.front_overhang) * np.cos(theta),
                        y + (veh.L + veh.front_overhang) * np.sin(theta)])
        # 后端两个角
        left = np.array([x - veh.rear_overhang * np.cos(theta) - (veh.width / 2) * np.sin(theta),
                         y - veh.rear_overhang * np.sin(theta) + (veh.width / 2) * np.cos(theta)])
        right = np.array([x - veh.rear_overhang * np.cos(theta) + (veh.width / 2) * np.sin(theta),
                          y - veh.rear_overhang * np.sin(theta) - (veh.width / 2) * np.cos(theta)])

        tri = Polygon([tip, left, right])

        # 转换到像素坐标
        minx, miny, maxx, maxy = tri.bounds
        minx = max(int(np.round(minx / resolution)), 0)
        miny = max(int(np.round(miny / resolution)), 0)
        maxx = min(int(np.round(maxx / resolution)), W - 1)
        maxy = min(int(np.round(maxy / resolution)), H - 1)

        for px in range(minx, maxx + 1):
            for py in range(miny, maxy + 1):
                wx = (px + 0.5) * resolution
                wy = (py + 0.5) * resolution
                if tri.contains(Point(wx, wy)):
                    grid[py, px, channel] = 1  # numpy row=py, col=px

    # 绘制起点和终点
    draw_vehicle_pose(grid, start_pose, 0)
    draw_vehicle_pose(grid, goal_pose, 1)

    grid = np.clip(grid, 0, 255).astype(dtype)

    return grid


# 构建输出速度安全走廊的栅格地图
def Corridor2Raster(map_obj, speed_corridor, resolution=0.2, sigma_factor=0.5):
    """
    将速度走廊栅格化，走廊内每个栅格用速度值填充。

    map_obj: ContinuousMap
    speed_corridor: dict_keys(['t','x','y','yaw','kappa','v','a','delta','r'])
    resolution: 栅格分辨率
    sigma_factor: 高斯权重因子，越大表示平滑范围越宽

    输出: grid[H, W], dtype=float
    """
    W = int(np.round(map_obj.width / resolution))
    H = int(np.round(map_obj.height / resolution))
    grid = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    # 遍历每个走廊圆
    for x, y, v, kappa, r in zip(
            speed_corridor["x"],
            speed_corridor["y"],
            speed_corridor["v"],
            speed_corridor["kappa"],
            speed_corridor["r"]
    ):
        # 找到覆盖范围对应的栅格坐标
        px = int(np.round(x / resolution))
        py = int(np.round(y / resolution))
        rad_px = int(np.ceil(r / resolution))

        # 局部搜索窗口
        x_min, x_max = max(0, px - rad_px), min(W - 1, px + rad_px)
        y_min, y_max = max(0, py - rad_px), min(H - 1, py + rad_px)

        for gx in range(x_min, x_max + 1):
            for gy in range(y_min, y_max + 1):
                # 像素中心坐标
                wx = (gx + 0.5) * resolution
                wy = (gy + 0.5) * resolution
                d = np.hypot(wx - x, wy - y)
                if d <= r:
                    # 高斯权重（平滑速度分布）
                    sigma = sigma_factor * r
                    w = np.exp(-0.5 * (d / sigma) ** 2)
                    grid[gy, gx] += v * w
                    weight[gy, gx] += w

    # 归一化
    mask = weight > 1e-6
    grid[mask] = grid[mask] / weight[mask]
    grid = np.clip(grid, 0, 10)
    grid_uint8 = (grid / 10.0 * 255).astype(np.uint8)
    return grid_uint8


# TODO 数据生成部分
# ---------------------------------------------------------------
# 用于膨胀地图中的障碍物
def obstacle_inflate(map_obj, veh_obj, clearance: float = 0.2, buffer_resolution: int = 8):
    """
    返回一个新的 ContinuousMap，障碍物都膨胀了 radius = veh_obj.width/2 + clearance。
    - map_obj: ContinuousMap 实例（含 .width, .height, .margin, .obstacles: list[np.ndarray]）
    - veh_obj: 具有 .width 属性的 Vehicle-like 对象
    - clearance: 额外安全距离（m）
    - buffer_resolution: shapely.buffer 的圆角近似分辨率（越大越圆滑，慢些）
    """
    # 延申半径（按你指定）
    radius = float(veh_obj.width / 2.0 + clearance)

    # 动态 import ContinuousMap 类（与你工程里定义的位置一致）
    try:
        from sim.ContinuousMap import ContinuousMap
    except Exception:
        # 若路径不同，可直接从 map_obj.__class__ 创建新实例
        ContinuousMap = map_obj.__class__

    # 新地图容器
    new_map = ContinuousMap(width=map_obj.width, height=map_obj.height, margin=map_obj.margin + radius)

    # 有效边界（膨胀后仍应在 margin 内）
    bbox = box(new_map.margin, new_map.margin, new_map.width - new_map.margin, new_map.height - new_map.margin)

    polys = []
    for obs in getattr(map_obj, "obstacles", []):
        # obs 预计为 ndarray (N,2)
        try:
            poly = Polygon(obs)
        except Exception:
            # 若 obs 非法，跳过
            continue
        if poly.is_empty:
            continue
        # 膨胀并裁剪到地图有效 bbox
        inflated = poly.buffer(radius, resolution=buffer_resolution, join_style=1)  # round joins
        clipped = inflated.intersection(bbox)
        if clipped.is_empty:
            continue
        # 合并先收集（后做 unary_union）
        # clipped 可能是 Polygon 或 MultiPolygon
        if clipped.geom_type == "Polygon":
            polys.append(clipped)
        elif clipped.geom_type == "MultiPolygon":
            # MultiPolygon
            for part in clipped.geoms:
                if not part.is_empty:
                    polys.append(part)

    if not polys:
        # 没有任何障碍在膨胀后仍留存，直接返回空地图
        return new_map

    # 合并所有膨胀多边形 —— 去除重叠并把相连部分合并成独立连通块
    merged = unary_union(polys)

    # 处理合并结果，可能是 Polygon 或 MultiPolygon
    parts = []
    if merged.is_empty:
        parts = []
    # 统一处理 Polygon 和 MultiPolygon
    if hasattr(merged, 'geoms'):
        parts = list(merged.geoms)
    else:
        parts = [merged]
    parts = [p for p in parts if not p.is_empty and p.area >= 1e-6]

    # 将每个 part 的外边界转为 numpy array 并加入 new_map
    for part in parts:
        coords = np.array(part.exterior.coords)  # shapely 回龙点最后重复第一个
        if len(coords) >= 2:
            # 删除最后重复点（若重复）
            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            # 保证至少三点
            if coords.shape[0] >= 3:
                # Optional: ensure CCW orientation (some downstream funcs expect CCW)
                def polygon_area(poly):
                    x = poly[:, 0]
                    y = poly[:, 1]
                    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

                if polygon_area(coords) < 0:
                    coords = coords[::-1]
                new_map.add_obstacle(coords.copy())

    return new_map


# 用于生成Voronoi图
def generate_Voronoi(map_obj):
    """
    Generate a General Voronoi Diagram skeleton and return an undirected weighted graph.
    Input:
        map_obj: ContinuousMap instance (has .width, .height, .margin, .obstacles list of np.ndarray (N,2))
    Return:
        bone_graph: dict with keys:
            'nodes' : {nid: np.array([x,y]), ...}
            'edges' : {nid: [(nbr_id, length), ...], ...}
    Notes:
        - Requires scipy.spatial.Voronoi.
        - This function is vector-based (no rasterization).
        - Clearance filtering threshold, sampling step, merge eps are chosen conservatively; change inside if needed.
    """

    # Try to reuse your project's distance util if available
    from sim.tools import point_to_polygon_distance_tongyi as _pt2poly_dist

    # --- parameters (tunable) ---
    edge_sample_step = 0.5  # sampling step when generating obstacle edge points (m)
    boundary_sample_step = 2.0  # sampling along the inner map frame
    clearance_threshold = 1e-5  # minimum clearance (m) to keep a Voronoi edge (default small; recommend inflated obstacles earlier)
    segment_check_samples = 2  # number of sample points along a Voronoi segment to check clearance
    merge_eps = 1e-3  # merge nodes closer than this (m)

    # -----------------------------

    # Liang-Barsky clipping of a segment to bbox
    def _clip_segment_to_aabb(p1, p2, xmin, ymin, xmax, ymax):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < 1e-12:
                if qi < 0:
                    return None
            else:
                t = qi / pi
                if pi < 0:
                    if t > u2: return None
                    if t > u1: u1 = t
                else:
                    if t < u1: return None
                    if t < u2: u2 = t
        np1 = np.array([x1 + u1 * dx, y1 + u1 * dy])
        np2 = np.array([x1 + u2 * dx, y1 + u2 * dy])
        return np1, np2

    # sample obstacle polygon edges to generate Voronoi sites
    def _sample_obstacle_edges(map_obj, step=edge_sample_step):
        pts = []
        for poly in getattr(map_obj, "obstacles", []):
            poly = np.asarray(poly)
            for i in range(len(poly)):
                a = poly[i]
                b = poly[(i + 1) % len(poly)]
                seg = b - a
                L = float(np.linalg.norm(seg))
                n = max(1, int(np.ceil(L / step)))
                for k in range(n + 1):
                    pts.append(a * (1 - k / n) + b * (k / n))
        if len(pts) == 0:
            # fallback: sample 4 corners inside margin
            pts = [
                np.array([map_obj.margin, map_obj.margin]),
                np.array([map_obj.width - map_obj.margin, map_obj.margin]),
                np.array([map_obj.margin, map_obj.height - map_obj.margin]),
                np.array([map_obj.width - map_obj.margin, map_obj.height - map_obj.margin]),
            ]
        return np.array(pts)

    # sample inner boundary frame (margin rectangle)
    def _sample_boundary_frame(map_obj, step=boundary_sample_step):
        x0, x1 = map_obj.margin, map_obj.width - map_obj.margin
        y0, y1 = map_obj.margin, map_obj.height - map_obj.margin
        xs = np.arange(x0, x1 + 1e-6, step)
        ys = np.arange(y0, y1 + 1e-6, step)
        pts = []
        for x in xs:
            pts.append([x, y0])
            pts.append([x, y1])
        for y in ys:
            pts.append([x0, y])
            pts.append([x1, y])
        return np.array(pts)

    # segment clearance: sample points along segment and compute min distance to obstacles and margins
    def _segment_clearance(p1, p2, map_obj, samples=segment_check_samples):
        ts = np.linspace(0.0, 1.0, samples)
        dmin = float('inf')
        for t in ts:
            p = p1 * (1.0 - t) + p2 * t
            # distance to each obstacle (point->polygon distance)
            for obs in getattr(map_obj, "obstacles", []):
                d = _pt2poly_dist(p, obs)
                if d < dmin:
                    dmin = d
            # also distance to margin/border
            dmin = min(dmin,
                       p[0] - map_obj.margin,
                       p[1] - map_obj.margin,
                       map_obj.width - map_obj.margin - p[0],
                       map_obj.height - map_obj.margin - p[1])
            # early exit
            if dmin < clearance_threshold:
                return dmin
        return dmin

    # 1) build sites
    obs_sites = _sample_obstacle_edges(map_obj, step=edge_sample_step)
    boundary_sites = _sample_boundary_frame(map_obj, step=boundary_sample_step)
    sites = np.vstack([obs_sites, boundary_sites])

    # 2) compute Voronoi
    vor = Voronoi(sites)

    xmin, ymin = map_obj.margin, map_obj.margin
    xmax, ymax = map_obj.width - map_obj.margin, map_obj.height - map_obj.margin

    # 3) collect finite segments clipped to bbox
    segments = []
    # iterate ridge_vertices and ridge_points pairs
    for (pidx, qidx), (u, v) in zip(vor.ridge_points, vor.ridge_vertices):
        if u >= 0 and v >= 0:
            p1 = vor.vertices[u]
            p2 = vor.vertices[v]
            clipped = _clip_segment_to_aabb(p1, p2, xmin, ymin, xmax, ymax)
            if clipped is not None:
                segments.append(clipped)
        else:
            # infinite ridge: build a far line and clip
            p = vor.points[pidx]
            q = vor.points[qidx]
            # perpendicular direction
            dir_vec = np.array([p[1] - q[1], q[0] - p[0]], dtype=float)
            if np.linalg.norm(dir_vec) < 1e-9:
                continue
            center = (p + q) / 2.0
            far1 = center + dir_vec * 1e5
            far2 = center - dir_vec * 1e5
            clipped = _clip_segment_to_aabb(far1, far2, xmin, ymin, xmax, ymax)
            if clipped is not None:
                segments.append(clipped)

    # 4) filter segments by clearance
    kept_segments = []
    for p1, p2 in segments:
        if _segment_clearance(p1, p2, map_obj, samples=segment_check_samples) >= clearance_threshold:
            kept_segments.append((p1, p2))

    # 5) assemble nodes + adjacency (merge close nodes)
    nodes = {}  # id -> coord (np.array)
    edges = {}  # id -> list of (nbr_id, length)

    def _add_node(pt):
        # merge if close
        for nid, xy in nodes.items():
            if np.linalg.norm(xy - pt) <= merge_eps:
                return nid
        nid = len(nodes)
        nodes[nid] = np.array(pt, dtype=float)
        edges[nid] = []
        return nid

    for p1, p2 in kept_segments:
        a = _add_node(p1)
        b = _add_node(p2)
        dist = float(np.linalg.norm(nodes[a] - nodes[b]))
        # prevent duplicate edge
        if not any(nid == b for (nid, _) in edges[a]):
            edges[a].append((b, dist))
        if not any(nid == a for (nid, _) in edges[b]):
            edges[b].append((a, dist))

    bone_graph = {'nodes': nodes, 'edges': edges}
    return bone_graph


# 用于生成GVD图，其实没用，因为和Voronoi图一样
def generate_GVD(voronoi_graph, map_obj):
    """
    从 Voronoi 结果中过滤障碍物内部/穿越障碍物的边，得到 GVD。
    修改后的 Voronoi 已没有错误的边
    输入:
        voronoi_graph: dict, 包含 'nodes' 和 'edges'
        map_obj: dict, 包含 'obstacles' (膨胀后的障碍物 Polygon 列表)
    输出:
        bone_graph: dict, 格式同 voronoi_graph
    """
    nodes = voronoi_graph['nodes']
    edges = voronoi_graph['edges']
    obstacles = map_obj.obstacles

    # 预处理：把障碍物转为 shapely Polygon
    obstacle_polys = [Polygon(o) for o in obstacles]

    def is_free(pt):
        """判断点是否在自由空间"""
        p = Point(pt)
        return not any(poly.contains(p) for poly in obstacle_polys)

    bone_nodes = {}
    bone_edges = {}

    for nid, pt in nodes.items():
        if is_free(pt):  # 节点必须在自由空间
            bone_nodes[nid] = pt

    for nid, nbrs in edges.items():
        if nid not in bone_nodes:
            continue
        valid_nbrs = []
        p1 = Point(nodes[nid])
        for nbr_id, w in nbrs:
            if nbr_id not in bone_nodes:
                continue
            p2 = Point(nodes[nbr_id])
            seg = LineString([p1, p2])

            # 如果线段与任何障碍物内部相交，则丢弃
            if any(poly.intersects(seg) and not poly.touches(seg) for poly in obstacle_polys):
                continue
            valid_nbrs.append((nbr_id, w))

        if valid_nbrs:
            bone_edges[nid] = valid_nbrs

    bone_graph = {'nodes': bone_nodes, 'edges': bone_edges}

    return bone_graph


# 用于生成拓扑图
def gvd_to_topo(gvd_map):
    """
    简化 GVD 为拓扑图，保留原始节点 id：
      - 保留所有度 >= 3 的节点；
      - 保留那些度 == 1 且可延伸到某个度 >= 3 节点的端点（其 id 不变）；
      - 中间度 == 2 的节点被合并（不保留），边权为链上权重的累加。

    输入:
        gvd_map: {'nodes': {nid: np.array([x,y])}, 'edges': {nid: [(nbr_id, weight), ...], ...}}

    返回:
        topo_map: {'nodes': {nid: coord, ...}, 'edges': {nid: [(nbr_id, total_weight), ...], ...}}
        （节点 id 保持原样）
    """
    nodes = gvd_map['nodes']
    edges = gvd_map['edges']

    # degree
    degree = {nid: len(edges.get(nid, [])) for nid in nodes.keys()}

    # helper: weight lookup
    def edge_weight(a, b):
        for nid, w in edges.get(a, []):
            if nid == b:
                return w
        return None

    topo_nodes_set = set(nid for nid, d in degree.items() if d >= 3)  # 初始关键节点集合
    topo_edges = {}  # nid -> list of (nbr_id, total_weight)
    added_pairs = set()  # 用 frozenset({a,b}) 防止重复加入边
    visited_original = set()  # 记录原始边是否已被遍历 (frozenset)

    # 初始化 topo_edges 对于关键节点（以后会添加端点）
    for nid in topo_nodes_set:
        topo_edges.setdefault(nid, [])

    # 遍历每个关键节点的每个邻边，向外延伸直到遇到关键节点或端点
    for start in list(topo_nodes_set):
        for nbr, w in edges.get(start, []):
            pair0 = frozenset((start, nbr))
            if pair0 in visited_original:
                continue

            # 开始沿这条方向走
            total_w = float(w)
            prev = start
            curr = nbr
            visited_original.add(pair0)  # 标记已访问该原始段

            path_nodes = [start, curr]  # 可选：记录链上经过节点（包含端点）

            # 沿着链继续走，直到碰到 degree>=3 或 degree==1（端点）
            while True:
                dcurr = degree.get(curr, 0)
                if dcurr >= 3:
                    end = curr
                    break
                if dcurr == 1:
                    end = curr
                    break
                # dcurr == 2 的情况：找下一个节点
                next_candidates = [nn for nn, _ in edges.get(curr, []) if nn != prev]
                if not next_candidates:
                    # 不该发生，但防御性处理
                    end = curr
                    break
                nxt = next_candidates[0]
                wseg = edge_weight(curr, nxt)
                if wseg is None:
                    # 数据异常，跳出
                    end = curr
                    break
                total_w += float(wseg)
                prev, curr = curr, nxt
                path_nodes.append(curr)
                visited_original.add(frozenset((prev, curr)))

            # 现在 end 是度>=3 或度==1
            # 保留端点只有在 end 与某关键节点相连（这里我们是从 start(关键) 出发，故总是满足）
            # 所以我们要把 start 与 end 作为 topo 的两个端点（start 已是关键节点）
            if degree.get(end, 0) >= 3 or degree.get(end, 0) == 1:
                a = start
                b = end
                pair_key = tuple(sorted((a, b)))
                if pair_key in added_pairs:
                    continue
                added_pairs.add(pair_key)

                # 确保 topo_nodes_set 包含 end（如果是端点则应加入）
                topo_nodes_set.add(a)
                topo_nodes_set.add(b)

                topo_edges.setdefault(a, []).append((b, total_w))
                topo_edges.setdefault(b, []).append((a, total_w))

    # 最后，移除那些不应存在的孤立条目（如果有）
    # 只保留 topo_nodes_set 中节点，并确保每个 topo_nodes_set 节点都有 edges 键
    final_nodes = {nid: nodes[nid] for nid in topo_nodes_set}
    final_edges = {}
    for nid in topo_nodes_set:
        neighs = topo_edges.get(nid, [])
        # 过滤掉连接到未保留节点的边（理论上不存在）
        neighs_filtered = [(m, w) for (m, w) in neighs if m in topo_nodes_set]
        final_edges[nid] = neighs_filtered

    return {'nodes': final_nodes, 'edges': final_edges}


# 用于将起始点映射到拓扑图
def proj_point2topo(start_pose, goal_pose, topo_map, map_obj, angle_range=np.pi / 6):
    def candidates_for_pose(pose, forward=True, cross_flag=True):
        x, y, theta = pose
        if not forward:
            theta = theta % (2 * np.pi) - np.pi  # 反向角度
        candidates = []

        for nid, coord in topo_map['nodes'].items():
            if len(topo_map['edges'].get(nid, [])) < 3:
                continue

            vec = coord - np.array([x, y])
            distance = np.linalg.norm(vec)
            if distance < 1e-6:
                candidates.append(nid)
                continue

            # 检查角度条件
            angle_to_node = math.atan2(vec[1], vec[0])
            if abs((angle_to_node - theta + np.pi) % (2 * np.pi) - np.pi) > angle_range:
                continue

            # 直线：起点到候选点
            line = LineString([(x, y), tuple(coord)])

            # 1. 检查是否穿过障碍物
            collision = False
            for obs in map_obj.obstacles:
                poly = Polygon(obs)
                if line.intersects(poly):
                    collision = True
                    break
            if collision:
                continue

            # 2. 检查是否和拓扑边交叉（除非拓扑边包含该候选点）
            if cross_flag:
                cross_topo = False
                for n1, neighbors in topo_map['edges'].items():
                    for n2_info in neighbors:
                        if n1 >= n2_info[0]:  # 避免重复
                            continue
                        # 如果拓扑边不包含当前候选点，才需要检查
                        if nid not in (n1, n2_info[0]):
                            edge_line = LineString([
                                tuple(topo_map['nodes'][n1]),
                                tuple(topo_map['nodes'][n2_info[0]])
                            ])
                            if line.intersects(edge_line):
                                cross_topo = True
                                break
                    if cross_topo:
                        break
                if cross_topo:
                    continue

            # 如果通过所有条件，就接受这个候选点
            candidates.append(nid)

        return candidates

    start_candidates = candidates_for_pose(start_pose, forward=True)
    goal_candidates = candidates_for_pose(goal_pose, forward=False)

    if len(start_candidates) == 0:
        start_candidates = candidates_for_pose(start_pose, forward=True, cross_flag=False)

    if len(goal_candidates) == 0:
        goal_candidates = candidates_for_pose(goal_pose, forward=False, cross_flag=False)

    # 只保留一个最佳候选（角度最小）
    def best_candidate(pose, candidates, forward=True):
        x, y, theta = pose
        best = None
        best_angle_diff = float("inf")
        for nid in candidates:
            coord = topo_map['nodes'][nid]
            vec = coord - np.array([x, y])
            angle_to_node = math.atan2(vec[1], vec[0])
            if forward:
                angle_diff = abs((angle_to_node - theta + np.pi) % (2 * np.pi) - np.pi)
            else:
                theta = theta % (2 * np.pi) - np.pi  # 反向角度
                angle_diff = abs((angle_to_node - theta + np.pi) % (2 * np.pi) - np.pi)
            if angle_diff < best_angle_diff:
                best = nid
                best_angle_diff = angle_diff
        return best

    start_nid = best_candidate(start_pose, start_candidates, forward=True)
    goal_nid = best_candidate(goal_pose, goal_candidates, forward=False)

    return start_nid, goal_nid, start_candidates, goal_candidates


# 用于将起点(-1)与终点(-2)添加到地图字典中
def update_map(map_dict, start_pose, goal_pose, start_candidates, goal_candidates):
    new_map_dict = deepcopy(map_dict)
    # 添加起点终点
    new_map_dict["nodes"][-1] = np.array(start_pose[0:2])
    new_map_dict["nodes"][-2] = np.array(goal_pose[0:2])

    for sc in start_candidates:
        dist1 = math.hypot(start_pose[0] - new_map_dict["nodes"][sc][0],
                           start_pose[1] - new_map_dict["nodes"][sc][1])
        if -1 not in new_map_dict["edges"].keys():
            new_map_dict["edges"][-1] = [(sc, dist1)]
        else:
            new_map_dict["edges"][-1].append((sc, dist1))
        new_map_dict["edges"][sc].append((-1, dist1))

    for gc in goal_candidates:
        dist2 = math.hypot(goal_pose[0] - new_map_dict["nodes"][gc][0],
                           goal_pose[1] - new_map_dict["nodes"][gc][1])
        if -2 not in new_map_dict["edges"].keys():
            new_map_dict["edges"][-2] = [(gc, dist2)]
        else:
            new_map_dict["edges"][-2].append((gc, dist2))
        new_map_dict["edges"][gc].append((-2, dist2))

    return new_map_dict


# 用于在拓扑地图上搜索可行路径
def generate_topo_route(new_gvd_map, start_nids, goal_nids):
    """
    枚举 new_gvd_map 中从 -1 到 -2 的所有简单路径（不重复节点）。
    输入:
        new_gvd_map: dict
            {
                "nodes": {nid: np.array([x, y]), ...},
                "edges": {nid: [(nbr_id, weight), ...], ...}
            }
    输出:
        topo_routes: list of list
            [[nid1, nid2, ..., -2], [...], ...]
    """
    start, goal = -1, -2
    edges = new_gvd_map["edges"]
    nodes = new_gvd_map["nodes"]
    topo_routes = []

    def dfs(current, path, visited):
        if current == goal:
            topo_routes.append(path[:])
            return
        for nbr, _ in edges.get(current, []):
            if nbr not in visited:  # 避免成环
                visited.add(nbr)
                path.append(nbr)
                dfs(nbr, path, visited)
                path.pop()
                visited.remove(nbr)

    dfs(start, [start], {start})

    routes = []
    for route in topo_routes:
        # 找到从头开始第2个元素之后的 start_nid 出现
        new_route = route[:]

        for i in range(len(new_route) - 2, 1, -1):
            if new_route[i] in start_nids:
                p1 = nodes[new_route[i]] - nodes[new_route[i - 1]]
                p2 = nodes[new_route[i - 1]] - nodes[new_route[i - 2]]
                if np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)) >= 0.5:
                    new_route = [-1] + new_route[i:]
                break

        for i in range(1, len(new_route) - 2):
            if new_route[i] in goal_nids:
                p1 = nodes[new_route[i + 1]] - nodes[new_route[i]]
                p2 = nodes[new_route[i + 2]] - nodes[new_route[i + 1]]
                if np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)) >= 0.5:
                    new_route = new_route[:i + 1] + [-2]
                break

        w = 0
        for i in range(len(new_route) - 1):
            n1 = new_route[i]
            n2 = new_route[i + 1]
            flag = False
            for n3 in new_gvd_map['edges'][n2]:
                if n3[0] == n1:
                    w += n3[1]
                    flag = True
                    break
            if not flag:
                w += math.hypot(new_gvd_map['nodes'][n1][0] - new_gvd_map['nodes'][n2][0],
                                new_gvd_map['nodes'][n1][1] - new_gvd_map['nodes'][n2][1])

        if (new_route, w) not in routes:
            routes.append((new_route, w))
    routes.sort(key=lambda x: x[1])

    return routes


# 辅助函数
def dijkstra_path(graph, start, goal):
    """
    单次 Dijkstra，返回从 start 到 goal 的节点 ID 路径
    """
    edges = graph['edges']
    dist = {nid: float('inf') for nid in edges}
    prev = {nid: None for nid in edges}
    dist[start] = 0
    pq = [(0, start)]  # (距离, 节点ID)

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist[u]:
            continue
        for v, w in edges.get(u, []):
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # 回溯路径
    path = []
    u = goal
    if prev[u] is not None or u == start:
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
    return path


# 用于在GVD地图上搜索对应的路线
def generate_gvd_route(new_gvd_map, topo_route):
    """
    根据 topo_route 展开得到全局坐标路径
    """
    full_node_path = []
    for i in range(len(topo_route) - 1):
        a, b = topo_route[i], topo_route[i + 1]
        sub_path = dijkstra_path(new_gvd_map, a, b)
        if i > 0:
            sub_path = sub_path[1:]  # 避免重复拼接
        full_node_path.extend(sub_path)

    # 映射到坐标
    coords = [new_gvd_map['nodes'][nid] for nid in full_node_path]
    return np.array(coords)


# 用于根据贝塞尔曲线生成可行空间
def generate_drivable_area(new_map, bs_route, max_radius=8.0, step=0.4):
    """
    沿着B样条路径生成可行驶的凸空间（圆形表示），采用自适应采样间隔

    参数:
        new_map: 地图对象，需包含 new_map.obstacles (list of shapely.Polygon)
        bs_route: list of (x,y)，B样条路径
        max_radius: 每个凸空间的最大半径
        step: 逐渐缩小半径的步长
        min_radius: 最小允许半径，小于该值则舍弃该点

    返回:
        drivable_area: list of dict [{"type": "circle", "center": (x, y), "radius": r}, ...]
    """

    line = LineString(bs_route)
    drivable_area = []
    all_obs = [Polygon(obs) for obs in new_map.obstacles]
    all_obs.append(Polygon(np.array([[0, 0, new_map.width, new_map.width], [0, new_map.margin, new_map.margin, 0]]).T))
    all_obs.append(Polygon(np.array([[0, 0, new_map.width, new_map.width],
                                     [new_map.height - new_map.margin, new_map.height, new_map.height,
                                      new_map.height - new_map.margin]]).T))
    all_obs.append(Polygon(np.array([[0, 0, new_map.margin, new_map.margin],
                                     [new_map.margin, new_map.height - new_map.margin, new_map.height - new_map.margin,
                                      new_map.margin]]).T))
    all_obs.append(Polygon(np.array([[new_map.width - new_map.margin, new_map.width - new_map.margin, new_map.width,
                                      new_map.width],
                                     [new_map.margin, new_map.height - new_map.margin, new_map.height - new_map.margin,
                                      new_map.margin]]).T))
    s = 0.0
    total_len = line.length
    prev_radius = 4

    while s < total_len:
        p = line.interpolate(s)
        cx, cy = p.x, p.y
        radius = max_radius

        # 找到最大可行半径
        while radius > 0:
            circle = Point(cx, cy).buffer(radius, resolution=16)
            collision = any(circle.intersects(obs) for obs in all_obs)
            if not collision:
                break
            if radius < 1:
                radius = radius - 0.05
            else:
                radius = np.round(radius - step, 2)

        if radius > 0:
            drivable_area.append({"type": "circle", "center": (cx, cy), "radius": radius})
            prev_radius = radius  # 更新上一圆半径
        else:
            prev_radius = 0.3

        # 自适应采样间隔
        sample_dist = max(prev_radius / 4.0, 0.1)
        s += sample_dist

    return drivable_area


# 用于计算bs_route曲率，剔除掉不符合条件的
def curvature_spline(bs_route, veh, s=0, ):
    x = bs_route[:, 0]
    y = bs_route[:, 1]
    n = len(x)

    if n < 3: raise ValueError("路径至少需要3个点")

    # 1. 计算累积弧长作为参数
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
    s_param = np.cumsum(segment_lengths)
    s_param = np.insert(s_param, 0, 0)  # 从0开始

    # 2. 使用弧长参数进行样条拟合
    tck, u = splprep([x, y], u=s_param, s=s, per=False, k=3)  # k=3: 三次样条

    # 3. 在原始参数点处求一阶和二阶导数
    dx_dt, dy_dt = splev(u, tck, der=1)  # 一阶导
    d2x_dt, d2y_dt = splev(u, tck, der=2)  # 二阶导

    # 4. 计算曲率 kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx_dt * d2y_dt - dy_dt * d2x_dt)
    denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5

    # 防止除零
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    curvature = numerator / (denominator + 1e-6)

    # 5. 曲率半径
    radius = 1.0 / curvature
    if min(radius) < 0.3 * veh.L / np.tan(veh.max_delta):
        return False, min(radius)
    else:
        return True, min(radius)
