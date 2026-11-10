import numpy as np
import casadi as ca
from scipy.interpolate import interp1d


def generate_trajectory(opt_path,
                            dt=0.1,
                            eps=1e-6,
                            solver_opts=None):

    # --- 路径参数 ---
    N = len(opt_path['x'])
    x_nodes = np.asarray(opt_path['x'], dtype=float)  # length N, 原坐标x
    y_nodes = np.asarray(opt_path['y'], dtype=float)  # length N, 原坐标y
    yaw_nodes = np.asarray(opt_path['yaw'], dtype=float)  # length N, 原航向角yaw

    ds = np.asarray(opt_path['ds'], dtype=float)  # length N-1, 原分段长度
    kappa_seg = np.asarray(opt_path['kappa'], dtype=float)  # length N-1, 原分段曲率

    if N < 2: return {'success': False, 'message': 'path too short', 'traj': None}

    # --- 车辆参数 ---
    v_global_max = 10.0  # 最大速度
    a_pos = 5.0  # 最大加速度
    a_neg = -6.0  # 最大减速度
    max_jerk = 6.0  # 最大加加速度
    L = 2.4  # 轴距
    max_delta = 40 * np.pi / 180  # 最大转向角
    max_delta_dot = np.deg2rad(30.0)  # 最大转角速度

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

    # --- 速度限制：来自横向加速度 ---
    a_lat_max = 3.0  # m/s^2 默认横向加速度上限
    v_curve = np.full(N, v_global_max)
    for i in range(N):
        ki = abs(kappa_node[i])
        if ki < 1e-9:
            v_curve[i] = v_global_max
        else:
            v_curve[i] = min(v_global_max, np.sqrt(max(eps, a_lat_max / ki)))

    # --- 速度限制：完整限制 ---
    v_init_limit = v_curve

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

    path2traj={'x':x_nodes, 'y':y_nodes, 'yaw':yaw_nodes, 'ds':ds, 'v':v_opt, 'kappa':kappa_node, 'L':2.4}
    opt_traj = resample_trajectory(path2traj, dt)

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
    f_x = interp1d(t_nodes, x_nodes, kind="linear")
    f_y = interp1d(t_nodes, y_nodes, kind="linear")
    f_yaw = interp1d(t_nodes, yaw_nodes, kind="linear")
    f_kappa = interp1d(t_nodes, kappa_nodes, kind="linear")
    f_v = interp1d(t_nodes, v_nodes, kind="linear")

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
    L = path2traj.get("L", 2.4)  # 车辆轴距 (可传入 veh.L)
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