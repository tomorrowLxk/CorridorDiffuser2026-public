import casadi as ca
import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString


def generate_opt_path(drivable_area, veh, start_pose, goal_pose, bs_route=None, M=None, v_min=1, sigma=0.05):
    eps = 1e-6
    L = veh.L
    delta_max = veh.max_delta  # veh.max_delta
    delta_dot_max = veh.max_delta_dot
    kappa_max = np.tan(delta_max) / L

    if M is None: M = int(LineString(bs_route).length / 0.25)

    x = ca.SX.sym("x", M + 1)
    y = ca.SX.sym("y", M + 1)
    yaw = ca.SX.sym("yaw", M + 1)
    ds = ca.SX.sym("ds", M)
    kappa = ca.SX.sym("kappa", M)

    w_vars, w0, lbw, ubw = [], [], [], []

    # === 初始猜测 ===
    if bs_route is not None:
        bs_route = np.array(bs_route)
        s = np.linspace(0, 1, len(bs_route))
        s_new = np.linspace(0, 1, M + 1)
        f_x = interp1d(s, bs_route[:, 0], kind='cubic', fill_value="extrapolate")
        f_y = interp1d(s, bs_route[:, 1], kind='cubic', fill_value="extrapolate")
        xs = f_x(s_new)
        ys = f_y(s_new)
        dx = np.diff(xs)
        dy = np.diff(ys)
        yaws = np.arctan2(dy, dx)
        yaws = np.append(yaws, yaws[-1])
        ds_init = np.hypot(dx, dy)
        d_theta = (np.diff(yaws) + np.pi) % (2 * np.pi) - np.pi
        kappa_init = d_theta / (ds_init + 1e-6)
        kappa_init = np.clip(kappa_init, -kappa_max, kappa_max)
    else:
        xs = np.linspace(start_pose[0], goal_pose[0], M + 1)
        ys = np.linspace(start_pose[1], goal_pose[1], M + 1)
        yaws = np.linspace(start_pose[2], goal_pose[2], M + 1)
        total_len = np.hypot(goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1])
        ds_init = np.full(M, total_len / M)
        kappa_init = np.zeros(M)

    # === 变量边界 ===
    for k in range(M + 1):
        w_vars += [x[k], y[k], yaw[k]]
        w0 += [xs[k], ys[k], yaws[k]]
        lbw += [-ca.inf] * 3
        ubw += [ca.inf] * 3

    for k in range(M):
        w_vars += [ds[k], kappa[k]]
        w0 += [ds_init[k], kappa_init[k]]
        lbw += [0.1, -kappa_max]
        ubw += [max(max(ds_init), 1.0), kappa_max]  # ds 上限放宽

    # === 约束 ===
    g, lbg, ubg = [], [], []

    g += [kappa[0]]
    lbg += [-np.tan(delta_dot_max * 0.1) / L]
    ubg += [np.tan(delta_dot_max * 0.1) / L]

    g += [x[0] - start_pose[0],
          y[0] - start_pose[1],
          ca.sin(yaw[0] - start_pose[2])]
    g += [x[M] - goal_pose[0],
          y[M] - goal_pose[1],
          ca.sin(yaw[M] - goal_pose[2])]
    lbg += [0] * 6
    ubg += [0] * 6

    for k in range(M):
        x_next = x[k] + ca.cos(yaw[k]) * ds[k]
        y_next = y[k] + ca.sin(yaw[k]) * ds[k]
        yaw_next = yaw[k] + kappa[k] * ds[k]
        g += [x[k + 1] - x_next,
              y[k + 1] - y_next,
              ca.sin(yaw[k + 1] - yaw_next)]
        lbg += [0, 0, -1e-6]
        ubg += [0, 0, 1e-6]

    for k in range(M - 1):
        # avg_kappa = (kappa[k] * ds[k] + kappa[k + 1] * ds[k + 1]) / (ds[k] + ds[k + 1])
        avg_kappa = (kappa[k] + kappa[k + 1]) / 2
        g += [2 * (kappa[k + 1] - kappa[k]) / (ds[k] + ds[k + 1] + eps) / (1 + (avg_kappa * L) ** 2)]
        lbg += [-delta_dot_max / v_min / L]
        ubg += [delta_dot_max / v_min / L]

    # === 目标函数 ===
    j1 = sum(ds_init) + eps
    J = 1.0 * ca.sum(ds) / j1  # 路径长度
    j2 = sum(kappa_init ** 2) + eps
    J += 4.0 * ca.sumsqr(kappa) / j2  # 平均曲率小
    j3 = sum(((kappa_init[1:] - kappa_init[:-1]) / (ds_init[1:] + ds_init[:-1])) ** 2) + eps
    J += 7.0 * ca.sumsqr((kappa[1:] - kappa[:-1]) / (ds[1:] + ds[:-1] + eps)) / j3  # 曲率变化平滑

    sm1 = []
    for k in range(M + 1):
        d_list = []
        for c in drivable_area:
            cx, cy = c["center"]
            r = c["radius"]
            dist = (x[k] - cx) ** 2 + (y[k] - cy) ** 2
            d_list.append(dist / r ** 2 - 1)
        sm = -sigma * ca.log(ca.sum1(ca.exp(-ca.vertcat(*d_list) / sigma)))
        sm1.append(sm)
        J += ca.fmax(sm + 0.64, 0) ** 2
    sm2 = sigma * ca.log(ca.sum1(ca.exp(ca.vertcat(*sm1) / sigma)))
    J += 10000 * ca.fmax(0, sm2) ** 2

    # === 求解 ===
    nlp = {"x": ca.vertcat(*w_vars), "f": J, "g": ca.vertcat(*g)}
    opts = {'expand': True,
            'ipopt.print_level': 0,
            'print_time': 0,
            # 'ipopt.max_iter': 1e6,
            'ipopt.max_iter': 3e3,
            'ipopt.max_cpu_time': 10,
            # 'ipopt.tol': 1e-4,
            'ipopt.tol': 1e-3,
            }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    if not solver.stats()['success']:
        print("Optimization failed:", solver.stats()['return_status'])
        return None

    if sol["f"] > 40:
        return None

    # === 解包 ===
    w_opt = sol["x"].full().flatten()
    opt_path = {"x": [], "y": [], "yaw": [], "kappa": [], "ds": []}
    idx = 0
    for _ in range(M + 1):
        opt_path["x"].append(w_opt[idx])
        idx += 1
        opt_path["y"].append(w_opt[idx])
        idx += 1
        opt_path["yaw"].append(w_opt[idx])
        idx += 1
    for _ in range(M):
        opt_path["ds"].append(w_opt[idx])
        idx += 1
        opt_path["kappa"].append(w_opt[idx])
        idx += 1

    """
    # === 目标函数 ===
    j1 = sum(ds_init)
    ca.sum(opt_path["ds"]) / j1  # 路径长度
    j2 = sum(kappa_init ** 2)
    ca.sumsqr(opt_path["kappa"]) / j2  # 平均曲率小
    j3 = sum(((kappa_init[1:] - kappa_init[:-1]) / (ds_init[1:] + ds_init[:-1])) ** 2)
    ca.sumsqr((np.array(opt_path["kappa"][1:]) - np.array(opt_path["kappa"][:-1])) / (np.array(opt_path["ds"][1:]) + np.array(opt_path["ds"][:-1]))) / j3  # 曲率变化平滑
    
    j4=0
    sm1 = []
    for k in range(M + 1):
        d_list = []
        for c in drivable_area:
            cx, cy = c["center"]
            r = c["radius"]
            dist = ca.sqrt((opt_path["x"][k] - cx) ** 2 + (opt_path["y"][k] - cy) ** 2 + 1e-8)
            d_list.append(dist / r - 1)
        sm = -sigma * ca.log(ca.sum1(ca.exp(-ca.vertcat(*d_list) / sigma)))
        sm1.append(sm)
        j4 += ca.fmax(sm+0.4, 0)** 2
    sm2 = sigma * ca.log(ca.sum1(ca.exp(ca.vertcat(*sm1) / sigma)))
    10000 * ca.fmax(0, sm2) ** 2
    """

    return opt_path
