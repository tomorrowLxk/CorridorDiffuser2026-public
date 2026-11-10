import numpy as np
from shapely.geometry import Point, Polygon
from copy import deepcopy


def generate_speed_corridor(new_map, opt_traj, drivable_area, r_min=0.3):
    """
    cont_map: ContinuousMap 对象，obstacles 是 Polygon 列表
    traj: dict with keys 'x','y','v','kappa' (1D arrays)
    drivable_area: 原可行空间，新的安全走廊是其子集
    r_max_speed: 最大膨胀半径
    r_min: 半径下限
    kappa_ref: 曲率参考值，用于曲率约束
    speed_ref, speed_power: 用于非线性速度半径映射
    返回: radii (1D array)，每个轨迹点对应圆的半径
    """

    speed_corridor = deepcopy(opt_traj)

    old_corridor = []
    for data in drivable_area:
        old_corridor.append([data['center'][0], data['center'][1], data['radius']])
    old_corridor = np.array(old_corridor)

    radii = []
    kappa_abs = abs(opt_traj['kappa'])
    obstacles = [Polygon(obs) for obs in new_map.obstacles]

    for x, y, v, kappa in zip(opt_traj['x'], opt_traj['y'], opt_traj['v'], opt_traj['kappa']):
        point = Point(x, y)

        # 1. 原空间约束，上限
        r_con = old_corridor[:, 2] - np.linalg.norm(np.array([x, y]) - old_corridor[:, :2], axis=1)
        r_con = [x for x in r_con if x >= 0]
        r_con.append(r_min)
        r_con = max(r_con)

        # 2. 环境半径 r_env，上限
        if new_map.obstacles:
            r_env = min([point.distance(obs) for obs in obstacles])
        else:
            r_env = r_con

        # 2. 速度约束半径 r_speed（非线性映射），速度越大半径越小，映射公式可调整
        r_speed = r_con / (1 + ((v - min(opt_traj['v'])) / (max(opt_traj['v']) - min(opt_traj['v']))) ** 1.5)
        # 3. 曲率约束半径 r_curv
        r_curv = r_con / (1 + ((abs(kappa)-min(kappa_abs)) / (max(kappa_abs)-min(kappa_abs))) ** 1.5)

        # 5. 最终半径
        r_final = max(min(r_speed, r_curv, r_env), r_min)
        radii.append(r_final)
    speed_corridor['r'] = radii
    return speed_corridor


