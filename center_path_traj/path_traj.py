import os
import numpy as np
from scipy.interpolate import splprep, splev
import casadi as ca
from scipy.interpolate import PchipInterpolator
from scipy import integrate
from tqdm import tqdm
import re
from matplotlib import pyplot as plt
from scipy import stats
import time

def safe_splprep(points, s=0.0, k=3, per=False):
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2 and points.shape[1] == 2, "points must be (N, 2)"
    assert points.shape[0] >= 2, "At least 2 points required"

    x, y = points[:, 0], points[:, 1]

    # 去除相邻重复点（避免 ds=0）
    diffs = np.diff(np.column_stack([x, y]), axis=0)
    keep = np.linalg.norm(diffs, axis=1) > 1e-8
    keep = np.concatenate([[True], keep])
    x, y = x[keep], y[keep]

    N = len(x)
    if N < 2:
        raise ValueError("Not enough unique points after deduplication.")

    # 自动调整 k
    k = min(k, N - 1)
    if k < 1:
        k = 1

    # 调用 splprep
    try:
        tck, u = splprep([x, y], s=s, k=k, per=per)
        return tck, u
    except Exception as e:
        print(f"splprep failed with N={N}, k={k}, s={s}")
        print("x:", x)
        print("y:", y)
        raise e


def compute_path_info(points, yaw_start=None, yaw_end=None):
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    if N < 2:
        raise ValueError("At least 2 points required.")

    x = points[:, 0]
    y = points[:, 1]
    if points.shape[1] == 3:
        v = points[:, 2]
    else:
        v=np.zeros_like(x)

    # --- 1. 计算 ds ---
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)  # (N-1,)

    # --- 2. 计算 yaw ---
    yaw = np.zeros(N)

    if N == 2:
        yaw[:] = np.arctan2(dy[0], dx[0])
    else:
        # 中间点：用前后向量和的方向（更平滑）
        yaw[1:-1] = np.arctan2(dy[:-1] + dy[1:], dx[:-1] + dx[1:])
        # 首尾
        yaw[0] = np.arctan2(dy[0], dx[0])
        yaw[-1] = np.arctan2(dy[-1], dx[-1])

    # 覆盖用户指定的首尾 yaw
    if yaw_start is not None:
        yaw[0] = yaw_start
    if yaw_end is not None:
        yaw[-1] = yaw_end

    # 归一化 yaw 到 [-pi, pi]
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    # --- 3. 计算曲率 kappa (长度为 N，边界点可设为 NaN 或用邻近值) ---
    kappa = np.full(N, np.nan)  # 初始化为 NaN

    if N >= 3:
        # 向量化计算三点曲率（i from 1 to N-2）
        x_prev, x_curr, x_next = x[:-2], x[1:-1], x[2:]
        y_prev, y_curr, y_next = y[:-2], y[1:-1], y[2:]

        # 向量 AB 和 AC
        abx = x_curr - x_prev
        aby = y_curr - y_prev
        acx = x_next - x_prev
        acy = y_next - y_prev

        # 有向面积 * 2
        signed_area2 = abx * acy - aby * acx  # = 2 * signed area of triangle

        # 边长
        len_ab = np.hypot(abx, aby)
        len_bc = np.hypot(x_next - x_curr, y_next - y_curr)
        len_ac = np.hypot(acx, acy)

        # 避免除零
        denom = len_ab * len_bc * len_ac
        valid = denom > 1e-8

        kappa_mid = np.zeros_like(signed_area2)
        kappa_mid[valid] = signed_area2[valid] / denom[valid]

        kappa[1:-1] = kappa_mid

        # 可选：边界点用最近邻填充（或保持 NaN）
        kappa[0] = kappa[1] if not np.isnan(kappa[1]) else 0.0
        kappa[-1] = kappa[-2] if not np.isnan(kappa[-2]) else 0.0

    else:
        # N == 2: 直线，曲率为 0
        kappa[:] = 0.0

    return {
        'x': x,
        'y': y,
        'ds': ds,
        'kappa': kappa,  # shape (N,)
        'yaw': yaw,  # shape (N,)
        'v': v
    }


def generate_trajectory(opt_path, dt=0.1, eps=1e-6, solver_opts=None):
    # --- 路径参数 ---
    N = len(opt_path['x'])
    x_nodes = np.asarray(opt_path['x'], dtype=float)  # length N, 原坐标x
    y_nodes = np.asarray(opt_path['y'], dtype=float)  # length N, 原坐标y
    yaw_nodes = np.asarray(opt_path['yaw'], dtype=float)  # length N, 原航向角yaw
    v_ref = np.asarray(opt_path.get('v'), dtype=float)
    v_ref = None

    ds = np.asarray(opt_path['ds'], dtype=float)  # length N-1, 原分段长度
    kappa_seg = np.asarray(opt_path['kappa'], dtype=float)  # length N-1, 原分段曲率

    if N < 2: return {'success': False, 'message': 'path too short', 'traj': None}

    # --- 车辆参数 ---
    v_global_max = 3.0  # 最大速度
    # a_pos = 5.0  # 最大加速度
    # a_neg = -6.0  # 最大减速度
    a_pos = 2.0
    a_neg = -2.0
    max_jerk = 3.0  # 最大加加速度
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
    J = J_time + 0.01 * J_jerk + 0.01 * J_delta_dot

    # --- 构建优化问题：整合 ---
    nlp = {'x': v, 'f': J, 'g': ca.vertcat(*g)}
    if solver_opts is None:
        solver_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1e4, 'ipopt.tol': 1e-3}
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)

    # --- 构建优化问题：计算初始解 ---
    v_init = v_init_limit.copy()
    v_init[0] = 0.0
    v_init[-1] = 0.0

    if v_ref is not None and len(v_ref) == N:
        # 热启动：直接使用外部参考速度作为初值
        v_init_guess = v_ref.copy()
        # 强制端点归零（确保满足硬约束，防止求解器因初始违反而报错）
        v_init_guess[0] = 0.0
        v_init_guess[-1] = 0.0
        # 保险起见，把超出物理极限的值截断到 [0, v_curve] 范围内
        v_init_guess = np.clip(v_init_guess, 0, v_init_limit)
    else:
        # 冷启动（兜底方案）：沿用你原来的几何推导
        v_init_guess = v_init_limit / 2
        v_init_guess[0] = 0.0
        v_init_guess[-1] = 0.0

    # --- 构建优化问题：上下界 ---
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    lbx = np.zeros(N)
    ubx = np.array([max(eps, v_init_limit[i]) for i in range(N)])

    # --- 构建优化问题：求解 ---
    try:
        t1 = time.time()
        sol = solver(x0=v_init_guess, lbx=lbx, ubx=ubx, lbg=lbg_arr, ubg=ubg_arr)
        time_cost = 1000*(time.time() - t1)
        v_opt = np.array(sol['x']).flatten()
        success = solver.stats()['success']
        message = 'OK'
    except Exception as e:
        v_opt = v_init.copy()
        success = False
        message = str(e)
        time_cost = 0.0

    traj = {'x': x_nodes, 'y': y_nodes, 'yaw': yaw_nodes, 'ds': ds, 'v': v_opt, 'kappa': kappa_node, 'L': 2.4}

    return {'success': success, 'message': message, 'traj': traj, "time_cost": time_cost}


def resample_trajectory_final(path2traj, dt=0.1, recompute_yaw_kappa=True, v_min=1e-2):
    """
    从路径节点（无时间戳）生成均匀 dt 的时间轨迹。

    输入要求:
        - x, y, v: 必须，长度 N
        - ds: 可选，长度应为 N-1（段长）；若缺失则自动计算
        - yaw, kappa: 可选；若 recompute_yaw_kappa=True 则忽略
        - L: 车辆轴距（默认 2.4）

    输出:
        dict with keys: t, x, y, yaw, kappa, v, a, delta （所有长度一致）
    """
    # --- 1. 提取必要字段 ---
    x = np.asarray(path2traj['x'], dtype=float)
    y = np.asarray(path2traj['y'], dtype=float)
    v = np.asarray(path2traj['v'], dtype=float)
    N = len(x)
    assert N >= 2, "At least 2 points required."
    assert len(v) == N, "v must have same length as x."

    # --- 2. 计算弧长 s (N,) ---
    if 'ds' in path2traj:
        ds = np.asarray(path2traj['ds'])
        if len(ds) == N - 1:
            s = np.concatenate([[0.0], np.cumsum(ds)])
        elif len(ds) == N:
            # 允许传入累积 s（但需归零起点）
            s = ds - ds[0]
        else:
            raise ValueError(f"ds 长度应为 {N - 1} 或 {N}，实际为 {len(ds)}")
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        ds_segments = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(ds_segments)])

    assert len(s) == N, "s 长度必须等于 N"

    # --- 3. 平滑速度剖面 v(s) ---
    v_clipped = np.clip(v, v_min, None)
    v_s_func = PchipInterpolator(s, v_clipped)

    # --- 4. 高精度积分 t(s) = ∫ ds / v(s) ---
    s_fine = np.linspace(s[0], s[-1], max(1000, 4 * N))
    v_fine = v_s_func(s_fine)
    dt_ds = 1.0 / v_fine
    t_fine = integrate.cumulative_trapezoid(dt_ds, s_fine, initial=0.0)
    T_end = t_fine[-1]

    if T_end <= 0:
        raise ValueError("Total time is zero. Check velocity values.")

    # --- 5. 构建 s(t) 反函数 ---
    s_t_func = PchipInterpolator(t_fine, s_fine)

    # --- 6. 均匀时间采样 ---
    t_new = np.arange(0.0, T_end + dt / 2, dt)
    s_new = s_t_func(t_new)

    # --- 7. 插值 x(s), y(s) ---
    x_s = PchipInterpolator(s, x)
    y_s = PchipInterpolator(s, y)
    x_new = x_s(s_new)
    y_new = y_s(s_new)
    v_new = v_s_func(s_new)

    # --- 8. 计算 yaw 和 kappa ---
    if recompute_yaw_kappa:
        # --- 1. 从 x/y 计算 yaw（几何方向，与参数化无关）---
        # 注意：不要用 dt！因为 yaw = atan2(dy/ds, dx/ds) ∝ atan2(dy, dx)
        dx_geom = np.gradient(x_new)
        dy_geom = np.gradient(y_new)
        yaw_new = np.arctan2(dy_geom, dx_geom)

        # --- 2. 使用三点圆拟合法计算曲率 kappa（带符号）---
        N = len(x_new)
        if N < 3:
            kappa_new = np.zeros(N)
        else:
            # 向量化计算中间点 (i=1 to N-2)
            x_prev, x_curr, x_next = x_new[:-4], x_new[2:-2], x_new[4:]
            y_prev, y_curr, y_next = y_new[:-4], y_new[2:-2], y_new[4:]

            # 向量 AB 和 AC
            abx = x_curr - x_prev
            aby = y_curr - y_prev
            acx = x_next - x_prev
            acy = y_next - y_prev

            # 有向面积 * 2 = cross(AB, AC)
            signed_area2 = abx * acy - aby * acx

            # 三边长度
            len_ab = np.hypot(abx, aby)
            len_bc = np.hypot(x_next - x_curr, y_next - y_curr)
            len_ac = np.hypot(acx, acy)

            # 曲率公式: kappa = (2 * Area) / (|AB| * |BC| * |AC|)
            denom = len_ab * len_bc * len_ac
            kappa_mid = np.zeros_like(signed_area2)
            valid = denom > 1e-8
            kappa_mid[valid] = signed_area2[valid] / denom[valid]

            # 构建完整 kappa 数组
            kappa_new = np.zeros(N)
            kappa_new[1:-1] = kappa_mid
            # 边界点：用最近邻值填充（也可设为0）
            kappa_new[0] = kappa_new[1]
            kappa_new[-1] = kappa_new[-2]
    else:
        # 使用原始 yaw/kappa（需用户确保其高质量）
        if 'yaw' not in path2traj or 'kappa' not in path2traj:
            raise ValueError("recompute_yaw_kappa=False requires 'yaw' and 'kappa' in input.")

        yaw_unwrapped = np.unwrap(np.asarray(path2traj['yaw']))
        kappa_orig = np.asarray(path2traj['kappa'])

        yaw_s = PchipInterpolator(s, yaw_unwrapped)
        kappa_s = PchipInterpolator(s, kappa_orig)

        yaw_new = np.angle(np.exp(1j * yaw_s(s_new)))  # wrap to [-pi, pi]
        kappa_new = kappa_s(s_new)

    # --- 9. 导出量 ---
    a_new = np.gradient(v_new, dt)
    L = path2traj.get('L', 2.4)
    delta_new = np.arctan(L * kappa_new)

    return {
        't': t_new,
        'x': x_new,
        'y': y_new,
        'yaw': yaw_new,
        'kappa': kappa_new,
        'v': v_new,
        'a': a_new,
        'delta': delta_new
    }


def load_path_data(directory_path):
    data_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            # 提取文件名中开头的数字部分（支持 4.npy, 123.npy 等）
            match = re.match(r'^(\d+)', filename)
            if match:
                key = int(match.group(1))  # 转为整数 key
                file_path = os.path.join(directory_path, filename)
                data = np.load(file_path, allow_pickle=True)
                data_dict[key] = data
    return data_dict


def compute_velocity_diff(smooth_traj, smooth_path):
    """
    对 smooth_path 中每个点，在 smooth_traj 上找 xy 最近的点，
    返回两者速度 v 的差值 (path_v - traj_v)。

    Parameters
    ----------
    smooth_traj : np.ndarray, shape (N, 3)  # [x, y, v]
    smooth_path : np.ndarray, shape (M, 3)  # [x, y, v]

    Returns
    -------
    v_diff : np.ndarray, shape (M,)  # path.v - nearest_traj.v
    nearest_indices : np.ndarray, shape (M,)  # 对应 smooth_traj 中的索引
    """
    # 仅用 x, y 两列计算欧氏距离
    traj_xy = smooth_traj[:, :2]   # (N, 2)
    path_xy = smooth_path[:, :2]   # (M, 2)

    # 广播计算所有两两距离: (M, N)
    dists = np.linalg.norm(path_xy[:, None, :] - traj_xy[None, :, :], axis=2)

    # 对每个 path 点取最近 traj 点的索引
    nearest_idx = np.argmin(dists, axis=1)  # (M,)

    # 取对应的 v 并算差值
    v_diff = smooth_path[:, 2] - smooth_traj[nearest_idx, 2]
    v_diff2 = 1-smooth_path[:, 2]/smooth_traj[nearest_idx, 2]

    return np.abs(v_diff), np.abs(v_diff2)

mean_errors = []
max_errors = []
min_errors = []
std_errors = []
time_cost = []
points_dict = load_path_data(fr"outputs/paths")
for k, path_old in tqdm(points_dict.items()):
    indices = np.linspace(0, len(path_old) - 1, min(len(path_old), 80), dtype=int)  # 实验用
    x_new, y_new, v_new = path_old[indices, 0], path_old[indices, 1], path_old[indices, 2]
    smooth_path = np.column_stack((x_new, y_new, v_new))
    path_info = compute_path_info(smooth_path)
    res = generate_trajectory(path_info)
    traj = res['traj']
    time_cost.append(res['time_cost'])
    opt_traj = resample_trajectory_final(traj, dt=0.1, recompute_yaw_kappa=False)
    smooth_traj = np.column_stack([opt_traj['x'], opt_traj['y'], opt_traj['v']])
    v_diff, v_diff2 = compute_velocity_diff(smooth_traj, smooth_path)
    mean_errors.append(np.mean(v_diff))
    max_errors.append(np.max(v_diff))
    min_errors.append(np.min(v_diff))
    std_errors.append(np.std(v_diff))

np.mean(time_cost)
# np.mean(mean_errors)
# np.mean(min_errors)
# np.mean(max_errors)
# np.mean(std_errors)
#
# np.mean(mean_errors) - 1.96 * stats.sem(mean_errors)
# np.mean(mean_errors) + 1.96 * stats.sem(mean_errors)
#
# np.mean(min_errors) - 1.96 * stats.sem(min_errors)
# np.mean(min_errors) + 1.96 * stats.sem(min_errors)
#
# np.mean(max_errors) - 1.96 * stats.sem(max_errors)
# np.mean(max_errors) + 1.96 * stats.sem(max_errors)
#
# np.mean(std_errors) - 1.96 * stats.sem(std_errors)
# np.mean(std_errors) + 1.96 * stats.sem(std_errors)

