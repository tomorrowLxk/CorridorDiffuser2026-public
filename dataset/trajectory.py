import numpy as np
import casadi as ca
from shapely.geometry import Point, Polygon, LineString
from scipy.interpolate import PchipInterpolator, interp1d
from scipy import integrate

def generate_opt_trajectory(opt_path, new_map, veh,
                            dt=0.1,
                            alpha_stop=0.8,
                            eps=1e-6,
                            solver_opts=None):
    """
    Time-parameterize a geometric path (opt_path) into a (x,y,v) trajectory sampled every dt,
    by optimizing node velocities v_i with CasADi/IPOPT subject to kino-dynamic constraints.

    Inputs:
        opt_path: dict with keys:
            - x (N,), y (N,), ds (N-1,), kappa (N-1,), yaw (N,)
        new_map: object with attribute `obstacles` (list of polygon vertex arrays (N,2))
        veh: vehicle object with attributes:
            max_v, max_acc (a_pos), min_acc (a_neg, negative number),
            max_jerk, L, max_delta, max_delta_dot
        dt: sampling interval for final trajectory
        alpha_stop: conservative factor in stopping-distance based speed limit
        eps: small numeric to avoid division by zero
        solver_opts: dict for IPOPT options
    Returns:
        dict: {
          'success': bool,
          'message': str,
          'v_nodes': array(N,),
          'node_time': array(N,),
          't': array(T,),
          'traj': array(T,3) columns x,y,v
        }
    """

    # --- 路径参数 ---
    N = len(opt_path['x'])
    x_nodes = np.asarray(opt_path['x'], dtype=float)  # length N, 原坐标x
    y_nodes = np.asarray(opt_path['y'], dtype=float)  # length N, 原坐标y
    yaw_nodes = np.asarray(opt_path['yaw'], dtype=float)  # length N, 原航向角yaw

    ds = np.asarray(opt_path['ds'], dtype=float)  # length N-1, 原分段长度
    kappa_seg = np.asarray(opt_path['kappa'], dtype=float)  # length N-1, 原分段曲率

    if N < 2: return {'success': False, 'message': 'path too short', 'traj': None}

    polys = [Polygon(obs) for obs in new_map.obstacles]  # 障碍物

    # --- 车辆参数 ---
    v_global_max = veh.max_v  # 最大速度
    a_pos = veh.max_acc  # 最大加速度
    a_neg = veh.min_acc  # 最大减速度
    max_jerk = veh.max_jerk  # 最大加加速度
    L = veh.L  # 轴距
    max_delta = veh.max_delta  # 最大转向角
    max_delta_dot = veh.max_delta_dot  # 最大转角速度

    # --- 计算 N 个节点处的曲率 ---
    kappa_node = np.zeros(N, dtype=float)
    for i in range(N):
        if N == 1:
            kappa_node[i] = 0.0
        elif i == 0:
            # kappa_node[i] = kappa_seg[0]
            kappa_node[i] = 0.0
        elif i == N - 1:
            kappa_node[i] = kappa_seg[-1]
        else:
            kappa_node[i] = (ds[i - 1] * kappa_seg[i - 1] + ds[i] * kappa_seg[i]) / (ds[i - 1] + ds[i])

    # --- 计算 N 个节点处的前轮转角 ---
    delta_node = np.arctan(kappa_node * L)

    # --- 计算每个点沿当前方向到障碍物的最短距离 ---
    dists = np.full(N, 1e6, dtype=float)
    if len(polys) > 0:
        for i in range(N):
            p = Point(x_nodes[i], y_nodes[i])
            ray = LineString([(x_nodes[i], y_nodes[i]),
                              (x_nodes[i] + 50.0 * np.cos(yaw_nodes[i]), y_nodes[i] + 50.0 * np.sin(yaw_nodes[i]))])
            md = 50
            for poly in polys:
                boundary = poly.exterior  # 获取多边形边界线段
                intersection = ray.intersection(boundary)  # 计算射线与多边形边界的交点
                if intersection.is_empty:  continue
                # 处理不同类型的交点（可能是点、多点、线段）
                if hasattr(intersection, 'geoms'):
                    # 多个交点（MultiPoint, GeometryCollection）
                    for geom in intersection.geoms:
                        if geom.geom_type in ['Point']:
                            d = p.distance(geom)
                            if d < md:
                                md = d
                        elif geom.geom_type in ['LineString']:
                            # 取线段起点（最近端）
                            pt = geom.boundary.geoms[0]  # 起点
                            d = p.distance(pt)
                            if d < md:
                                md = d
                elif intersection.geom_type == 'Point':
                    d = p.distance(intersection)
                    if d < md:
                        md = d
                elif intersection.geom_type == 'LineString':
                    pt = intersection.boundary.geoms[0]
                    d = p.distance(pt)
                    if d < md:
                        md = d
            dists[i] = md

    # --- 速度限制：来自横向加速度 ---
    a_lat_max = 3.0  # m/s^2 默认横向加速度上限
    v_curve = np.full(N, v_global_max)
    for i in range(N):
        ki = abs(kappa_node[i])
        if ki < 1e-9:
            v_curve[i] = v_global_max
        else:
            v_curve[i] = min(v_global_max, np.sqrt(max(eps, a_lat_max / ki)))

    # --- 速度限制：来自障碍物 ---
    v_obs = np.sqrt(np.maximum(0.0, 2.0 * abs(a_neg) * alpha_stop * dists))
    v_obs = np.minimum(v_obs, v_global_max)

    # --- 速度限制：完整限制 ---
    v_init_limit = np.minimum.reduce([v_curve, v_obs])

    # --- 构建优化问题：变量v ---
    v = ca.MX.sym('v', N)

    # --- 构建优化问题：不等式约束 空列表 ---
    g = []
    lbg = []
    ubg = []

    # --- 构建优化问题：不等式约束 起终点速度约束 ---
    g += [v[0] - 0.0, v[-1] - 0.0]
    lbg += [-eps, -eps]
    ubg += [eps, eps]

    # --- 构建优化问题：不等式约束 加速度范围约束 ---
    # ds_sym = [ca.MX(ds_i) for ds_i in ds]  # length N-1, 距离符号化
    for i in range(N - 1):
        a_seg = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds[i] + eps)
        g.append(a_seg)
        lbg.append(float(a_neg))
        ubg.append(float(a_pos))

    # --- 构建优化问题：不等式约束 加加速度范围约束 ---
    a_segs = []  # length N-1 路段加速度
    dt_segs = []  # length N-1 路段用时
    for i in range(N - 1):  # 先计算加速度与用时
        a_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds[i] + eps)
        a_segs.append(a_i)
        dt_i = 2.0 * ds[i] / (v[i] + v[i + 1] + eps)
        dt_segs.append(dt_i)
    # J_time = -sum(ds) / sum(dt_segs)  # 同时收集优化目标
    J_time = sum(dt_segs)

    square_jerk_2_segs = []
    for i in range(len(a_segs) - 1):
        jerk = (a_segs[i + 1] - a_segs[i]) / (dt_segs[i] + eps)
        square_jerk_2_segs.append(jerk ** 2)
        g.append(jerk)
        lbg.append(-float(max_jerk))
        ubg.append(float(max_jerk))
    J_jerk = sum(square_jerk_2_segs)  # 同时收集优化目标

    # --- 构建优化问题：不等式约束 前轮转角转速约束 ---
    square_delta_dot_segs = []
    for i in range(N - 1):
        delta_diff = float(delta_node[i + 1] - delta_node[i])
        delta_diff_dot = delta_diff / (dt_segs[i] + eps)
        square_delta_dot_segs.append(delta_diff_dot ** 2)
        g.append(delta_diff_dot)
        lbg.append(-float(max_delta_dot))
        ubg.append(float(max_delta_dot))
    J_delta_dot = sum(square_delta_dot_segs)  # 同时收集优化目标

    # --- 构建优化问题：优化目标 总目标 ---
    J = J_time  + 0.01 * J_jerk + 0.01 * J_delta_dot

    # --- 构建优化问题：整合 ---
    nlp = {'x': v, 'f': J, 'g': ca.vertcat(*g)}
    if solver_opts is None:
        solver_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1e6}
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)

    # --- 构建优化问题：计算初始解 ---
    v_init = v_init_limit.copy()
    v_init[0] = 0.0
    v_init[-1] = 0.0

    # --- 构建优化问题：上下界 ---
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    lbx = np.zeros(N)
    ubx = np.array([max(eps, v_init_limit[i]) for i in range(N)])

    # --- 构建优化问题：求解 ---
    try:
        sol = solver(x0=v_init/2, lbx=lbx, ubx=ubx, lbg=lbg_arr, ubg=ubg_arr)
        v_opt = np.array(sol['x']).flatten()
        success = solver.stats()['success']
        message = 'OK'
    except Exception as e:
        v_opt = v_init.copy()
        success = False
        message = str(e)

    path2traj={'x':x_nodes, 'y':y_nodes, 'yaw':yaw_nodes, 'ds':ds, 'v':v_opt, 'kappa':kappa_node, 'L':veh.L}
    opt_traj = resample_trajectory(path2traj)

    return {'success': success, 'message': message, 'opt_traj': opt_traj}


def resample_trajectory(path2traj, dt=0.1):
    """
    从最优路径点 (离散点) 生成时间轨迹 (dt 采样)
    输入:
        opt_path: dict {"x","y","yaw","kappa","ds","v"}  # 节点信息
        dt: 采样时间间隔
    输出:
        traj: dict {"t","x","y","yaw","kappa","v","a","delta"}
    """
    x_nodes = np.array(path2traj["x"])
    y_nodes = np.array(path2traj["y"])
    yaw_nodes = np.array(path2traj["yaw"])
    kappa_nodes = np.array(path2traj["kappa"])
    ds_nodes = np.array(path2traj["ds"])
    v_nodes = np.array(path2traj["v"])  # 优化得到的速度

    # === Step 1: 构造弧长坐标 ===
    s_nodes = np.zeros(len(x_nodes))
    s_nodes[1:] = np.cumsum(ds_nodes)

    # === Step 2: 用梯形法积分求时间节点 ===
    t_nodes = np.zeros(len(s_nodes))
    for k in range(1, len(s_nodes)):
        ds = s_nodes[k] - s_nodes[k-1]
        v_avg = 0.5 * (v_nodes[k] + v_nodes[k-1])
        t_nodes[k] = t_nodes[k-1] + ds / max(v_avg, 1e-3)

    # === Step 3: 构造时间插值器 ===
    f_x = interp1d(t_nodes, x_nodes, kind="cubic")
    f_y = interp1d(t_nodes, y_nodes, kind="cubic")
    f_yaw = interp1d(t_nodes, yaw_nodes, kind="cubic")
    f_kappa = interp1d(t_nodes, kappa_nodes, kind="cubic")
    f_v = interp1d(t_nodes, v_nodes, kind="cubic")

    # === Step 4: 按 dt 采样 ===
    T_total = t_nodes[-1]
    t_samples = np.arange(0, T_total, dt)
    x_samples = f_x(t_samples)
    y_samples = f_y(t_samples)
    yaw_samples = f_yaw(t_samples)
    kappa_samples = f_kappa(t_samples)
    v_samples = f_v(t_samples)

    # === Step 5: 数值微分得到加速度 ===
    a_samples = np.gradient(v_samples, dt)

    # === Step 6: 由曲率求前轮转角 delta ===
    L = path2traj.get("L", 2.8)  # 车辆轴距 (可传入 veh.L)
    delta_samples = np.arctan(L * kappa_samples)

    opt_traj = {
        "t": t_samples,
        "x": x_samples,
        "y": y_samples,
        "yaw": yaw_samples,
        "kappa": kappa_samples,
        "v": v_samples,
        "a": a_samples,
        "delta": delta_samples,
    }
    return opt_traj


def resample_trajectory_1(path2traj, dt=0.1):
    """
    从已有时域轨迹 (t 不等距) 重采样为均匀 dt 的轨迹。
    要求输入包含 "t" 字段。
    """
    required_keys = {"x", "y", "yaw", "kappa", "v"}
    if not required_keys.issubset(path2traj.keys()):
        raise ValueError(f"输入必须包含字段: {required_keys}")

    x_orig = np.array(path2traj["x"])
    y_orig = np.array(path2traj["y"])
    yaw_orig = np.array(path2traj["yaw"])
    kappa_orig = np.array(path2traj["kappa"])
    ds_orig = np.array(path2traj["ds"])
    v_orig = np.array(path2traj["v"])  # 优化得到的速度

    # === Step 1: 构造弧长坐标 ===
    s_nodes = np.zeros(len(x_orig))
    s_nodes[1:] = np.cumsum(ds_orig)

    # === Step 2: 用梯形法积分求时间节点 ===
    t_orig = np.zeros(len(s_nodes))
    for k in range(1, len(s_nodes)):
        ds = s_nodes[k] - s_nodes[k-1]
        v_avg = 0.5 * (v_orig[k] + v_orig[k-1])
        t_orig[k] = t_orig[k-1] + ds / max(v_avg, 1e-3)

    # 检查单调递增
    if not np.all(np.diff(t_orig) > 0):
        raise ValueError("时间戳 t 必须严格递增")

    # 构建 PCHIP 插值器（保形，避免过冲）
    interp_x = PchipInterpolator(t_orig, x_orig)
    interp_y = PchipInterpolator(t_orig, y_orig)
    interp_yaw = PchipInterpolator(t_orig, yaw_orig)
    interp_kappa = PchipInterpolator(t_orig, kappa_orig)
    interp_v = PchipInterpolator(t_orig, v_orig)

    # 均匀时间网格
    t_start, t_end = t_orig[0], t_orig[-1]
    t_new_ = np.arange(t_start, t_end + dt / 2, dt)

    # 重采样
    x_new_ = interp_x(t_new_)
    y_new_ = interp_y(t_new_)
    yaw_new_ = interp_yaw(t_new_)
    kappa_new_ = interp_kappa(t_new_)
    v_new_ = interp_v(t_new_)

    # 计算加速度 a = dv/dt（用中心差分 + 边界处理）
    a_new_ = np.gradient(v_new_, dt)

    # 计算前轮转角 delta ≈ arctan(wheelbase * kappa)
    L = path2traj.get("L", 2.8)  # 车辆轴距 (可传入 veh.L)
    delta_new_ = np.arctan(L * kappa_new_)

    return {
        "t": t_new_,
        "x": x_new_,
        "y": y_new_,
        "yaw": yaw_new_,
        "kappa": kappa_new_,
        "v": v_new_,
        "a": a_new_,
        "delta": delta_new_
    }


def resample_trajectory_2(path2traj, dt=0.1, v_min=0.01):
    """
    从路径 + 速度剖面 (无时间戳) 生成均匀 dt 的时间轨迹。
    输入应包含: x, y, yaw, kappa, v （可选 ds）
    """
    required_keys = {"x", "y", "yaw", "kappa", "v"}
    if not required_keys.issubset(path2traj.keys()):
        raise ValueError(f"输入必须包含字段: {required_keys}")

    x = np.asarray(path2traj["x"])
    y = np.asarray(path2traj["y"])
    yaw = np.asarray(path2traj["yaw"])
    kappa = np.asarray(path2traj["kappa"])
    v = np.asarray(path2traj["v"])

    if len(x) < 2:
        raise ValueError("至少需要两个点")

    # 1. 计算弧长 s
    if "ds" in path2traj:
        ds = np.asarray(path2traj["ds"])
        if len(ds) != len(x)-1:
            raise ValueError("ds 长度+1必须与 x 一致")
        s = np.concatenate([[0.0], np.cumsum(ds)])  # 确保 s[0]=0
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        ds_segments = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0.0], np.cumsum(ds_segments)])

    # 2. 平滑 v(s) 使用 PCHIP（避免振荡）
    v_clipped = np.clip(v, v_min, None)
    v_s_func = PchipInterpolator(s, v_clipped)

    # 3. 数值积分 t(s) = ∫ ds / v(s)
    s_fine = np.linspace(s[0], s[-1], max(1000, 4 * len(s)))
    v_fine = v_s_func(s_fine)
    v_fine[0], v_fine[-1] =  (v_fine[0]+ v_fine[1])/2, (v_fine[-1]+ v_fine[-2])/2
    dt_ds = 1.0 / v_fine
    t_fine = integrate.cumulative_trapezoid(dt_ds, s_fine, initial=0.0)

    # 4. 构建 s(t) 反函数
    s_t_func = interp1d(t_fine, s_fine, kind='linear', fill_value="extrapolate")

    # 5. 均匀时间网格
    T_end = t_fine[-1]
    t_new = np.arange(0.0, T_end + dt / 2, dt)

    # 6. 获取对应 s
    s_new = s_t_func(t_new)

    # 7. 插值原始路径量 (x(s), y(s), ...)
    x_s = PchipInterpolator(s, x)
    y_s = PchipInterpolator(s, y)
    yaw_s = PchipInterpolator(s, yaw)
    kappa_s = PchipInterpolator(s, kappa)

    x_new = x_s(s_new)
    y_new = y_s(s_new)
    yaw_new = yaw_s(s_new)
    kappa_new = kappa_s(s_new)
    v_new = v_s_func(s_new)

    # 8. 计算导出量
    a_new = np.gradient(v_new, dt)

    L = path2traj.get("L", 2.8)  # 车辆轴距 (可传入 veh.L)
    delta_new = np.arctan(L * kappa_new)

    return {
        "t": t_new,
        "x": x_new,
        "y": y_new,
        "yaw": yaw_new,
        "kappa": kappa_new,
        "v": v_new,
        "a": a_new,
        "delta": delta_new
    }