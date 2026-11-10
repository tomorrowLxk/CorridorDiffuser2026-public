import numpy as np
from matplotlib import pyplot as plt

from sim.tools import create_rectangle


class C:
    # vehicle config
    WD = 1.6  # [m] 前轮距/后轮距
    TR = 0.31  # [m] 轮胎半径
    TW = 0.185  # [m] 轮胎宽度

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.4 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


def draw_car(veh, x, y, yaw, steer, color='black'):
    car = np.array([[-veh.rear_overhang, -veh.rear_overhang, veh.front_overhang + veh.L, veh.front_overhang + veh.L,
                     -veh.rear_overhang],
                    [veh.width / 2, -veh.width / 2, -veh.width / 2, veh.width / 2, veh.width / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])

    Rot2 = np.array([[np.cos(steer), np.sin(steer)],
                     [-np.sin(steer), np.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[veh.L], [-C.WD / 2]])
    flWheel += np.array([[veh.L], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    Arrow(x, y, yaw, veh.L * 0.6, color)


def plot_init_case(init_map, veh=None, start_pose=None, goal_pose=None, init_route=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, init_map.width)
    ax.set_ylim(0, init_map.height)
    # ax.set_title("Initial conditions")
    ax.tick_params(labelsize=15)
    # 画出障碍物
    for obs in init_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, init_map.width, init_map.width],
            [0, init_map.margin, init_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, init_map.width, init_map.width],
            [init_map.height-init_map.margin, init_map.height, init_map.height, init_map.height-init_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, init_map.margin, init_map.margin],
            [init_map.margin, init_map.height-init_map.margin, init_map.height-init_map.margin, init_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([init_map.width-init_map.margin, init_map.width-init_map.margin, init_map.width, init_map.width],
            [init_map.margin, init_map.height - init_map.margin, init_map.height - init_map.margin, init_map.margin],
            color='black', alpha=0.9, label='margin')
    # RRT连接的路线
    px = [p[0] for p in init_route]
    py = [p[1] for p in init_route]
    ax.plot(px, py, linestyle='--', linewidth=1)
    # 起终点车辆位姿
    draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='red')
    draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig("init_case.png", bbox_inches='tight', dpi=600, pad_inches=0)
    plt.savefig("init_case.png", bbox_inches='tight', dpi=600)
    plt.show()


def plot_inflate_map(inflate_map, veh=None, start_pose=None, goal_pose=None, init_route=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.set_title("Map with inflated obstacles")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height-inflate_map.margin, inflate_map.height, inflate_map.height, inflate_map.height-inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height-inflate_map.margin, inflate_map.height-inflate_map.margin, inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width-inflate_map.margin, inflate_map.width-inflate_map.margin, inflate_map.width, inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # RRT连接的路线
    if init_route is not None:
        px = [p[0] for p in init_route]
        py = [p[1] for p in init_route]
        ax.plot(px, py, linestyle='--', linewidth=1)
    # 起终点车辆位姿
    if (veh is not None) and (start_pose is not None):
        draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='red')
    if (veh is not None) and (goal_pose is not None):
        draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    if (start_pose is not None) and (goal_pose is not None):
        ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    plt.show()


def plot_gvd(inflate_map, gvd_map, veh=None, start_pose=None, goal_pose=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("the GVD corresponding to the map")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 起终点车辆位姿
    if (veh is not None) and (start_pose is not None):
        draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='red')
    if (veh is not None) and (goal_pose is not None):
        draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    if (start_pose is not None) and (goal_pose is not None):
        ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    # 画出GVD图
    nodes = gvd_map['nodes']
    edges = gvd_map['edges']
    for nid, nbrs in edges.items():
        p1 = nodes[nid]
        for nbr_id, _ in nbrs:
            p2 = nodes[nbr_id]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.8, linewidth=1, )
    plt.savefig("gvd.png", bbox_inches='tight', dpi=600)
    plt.show()


def plot_topo_map(inflate_map, topo_map, veh=None, start_pose=None, goal_pose=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.set_title("Topological map")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 起终点车辆位姿
    if (veh is not None) and (start_pose is not None):
        draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='red')
    if (veh is not None) and (goal_pose is not None):
        draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    if (start_pose is not None) and (goal_pose is not None):
        ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    # 画出拓扑图的点
    nodes = topo_map['nodes']
    for nd in nodes.values():
        ax.plot(nd[0], nd[1], 'ob', alpha=0.6)
    # 画出拓扑图的边
    edges = topo_map['edges']
    for nid, nbrs in edges.items():
        p1 = nodes[nid]
        for nbr_id, _ in nbrs:
            p2 = nodes[nbr_id]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.6)
    plt.show()


def plot_topo_route(inflate_map, new_topo_map, topo_route, veh=None, start_pose=None, goal_pose=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 起终点车辆位姿
    if (veh is not None) and (start_pose is not None):
        draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='red')
    if (veh is not None) and (goal_pose is not None):
        draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    if (start_pose is not None) and (goal_pose is not None):
        ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    # 画出拓扑图的点
    nodes = new_topo_map['nodes']
    for nd in nodes.values():
        ax.plot(nd[0], nd[1], 'ob', alpha=0.6)
    # 画出拓扑图的边
    edges = new_topo_map['edges']
    for nid, nbrs in edges.items():
        p1 = nodes[nid]
        for nbr_id, _ in nbrs:
            p2 = nodes[nbr_id]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.6)
    # 画出拓扑路线
    for i in range(len(topo_route) - 1):
        ax.plot([nodes[topo_route[i]][0], nodes[topo_route[i + 1]][0]],
                [nodes[topo_route[i]][1], nodes[topo_route[i + 1]][1]], 'k', alpha=0.8, linewidth=2.0)
        # 在当前点（topo_route[i]）旁添加标注 "1"
        x, y = nodes[topo_route[i]]
        ax.text(x - 0.3, y + 0.3, topo_route[i], fontsize=10, color='red', weight='bold')  # 可调整偏移量
    plt.savefig("topo_map.png", bbox_inches='tight', dpi=600)
    plt.show()


def plot_gvd_route(inflate_map, topo_map, gvd_route, s_id, g_id):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 画出拓扑图的点
    nodes = topo_map['nodes']
    for nd in nodes.values():
        ax.plot(nd[0], nd[1], 'ob', alpha=0.6)
    # 画出拓扑图的边
    edges = topo_map['edges']
    for nid, nbrs in edges.items():
        p1 = nodes[nid]
        for nbr_id, _ in nbrs:
            p2 = nodes[nbr_id]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.8)
    nodes = topo_map['nodes']
    node_s = topo_map['nodes'][s_id]
    node_g = topo_map['nodes'][g_id]
    ax.plot(node_s[0], node_s[1], '*r', alpha=0.6, markersize=10)
    ax.plot(node_g[0], node_g[1], '*b', alpha=0.6, markersize=10)
    ax.plot(gvd_route[:, 0], gvd_route[:, 1], alpha=0.8, linewidth=2.0)
    plt.show()

def plot_bspline_route(inflate_map, topo_map, bs_route, control_points, s_id, g_id):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 画出控制点
    for cp in control_points:
        ax.plot(cp[0], cp[1], 'ob', alpha=0.6, markersize=3)
    # 画出路线
    node_s = topo_map['nodes'][s_id]
    node_g = topo_map['nodes'][g_id]
    ax.plot(node_s[0], node_s[1], '*r', alpha=0.6, markersize=12)
    ax.plot(node_g[0], node_g[1], '*b', alpha=0.6, markersize=12)
    ax.plot(bs_route[:, 0], bs_route[:, 1], alpha=0.8, linewidth=2.0)
    plt.savefig("bspline_route.png", bbox_inches='tight', dpi=600)
    plt.show()


def plot_drivable_area(inflate_map, drivable_area, veh=None, start_pose=None, goal_pose=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 起终点车辆位姿
    if (veh is not None) and (start_pose is not None):
        draw_car(veh, start_pose[0], start_pose[1], start_pose[2], steer=0, color='blue')
    if (veh is not None) and (goal_pose is not None):
        draw_car(veh, goal_pose[0], goal_pose[1], goal_pose[2], steer=0, color='blue')
    # 车后轴中心点坐标
    if (start_pose is not None) and (goal_pose is not None):
        ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
    # 可行空间
    for da in drivable_area:
        ax.fill(da['center'][0] + da['radius'] * np.cos(np.linspace(0, 2 * np.pi, 36)),
                da['center'][1] + da['radius'] * np.sin(np.linspace(0, 2 * np.pi, 36)),
                'r', edgecolor='none', alpha=0.5)
        ax.plot(da['center'][0], da['center'][1], 'bo',  markersize=3)
    plt.savefig("drivable_area.png", bbox_inches='tight', dpi=600)
    plt.show()


def plot_opt_path(inflate_map, drivable_area, opt_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 可行驶空间
    for da in drivable_area:
        ax.fill(da['center'][0] + da['radius'] * np.cos(np.linspace(0, 2 * np.pi, 36)),
                da['center'][1] + da['radius'] * np.sin(np.linspace(0, 2 * np.pi, 36)),
                'r', edgecolor='none', alpha=0.2)
    # 最优路径
    ax.plot(opt_path['x'], opt_path['y'], '--k', linewidth=2.0)
    # plt.savefig("opt_path.png", bbox_inches='tight', dpi=600)
    plt.show()

def plot_opt_traj(inflate_map, opt_traj):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")
    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 画出最优轨迹
    opt_traj = opt_traj['opt_traj']
    for i in range(len(opt_traj['x'])):
        ax.plot(opt_traj['x'][i], opt_traj['y'][i], 'or', markersize=opt_traj['v'][i]*2, alpha=0.2)
    # plt.savefig("opt_traj.png", bbox_inches='tight', dpi=600)
    plt.show()

def plot_speed_corridor(inflate_map, opt_traj, speed_corridor):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, inflate_map.width)
    ax.set_ylim(0, inflate_map.height)
    ax.tick_params(labelsize=15)
    # ax.set_title("?")

    # 画出障碍物
    for obs in inflate_map.obstacles:
        ax.fill(obs[:, 0], obs[:, 1],
                hatch='///', fill=True, color='silver', alpha=0.5,
                edgecolor='dimgray', linewidth=1.2, label='Obstacle')
    # 画出边界
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [0, inflate_map.margin, inflate_map.margin, 0],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.width, inflate_map.width],
            [inflate_map.height - inflate_map.margin, inflate_map.height, inflate_map.height,
             inflate_map.height - inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([0, 0, inflate_map.margin, inflate_map.margin],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    ax.fill([inflate_map.width - inflate_map.margin, inflate_map.width - inflate_map.margin, inflate_map.width,
             inflate_map.width],
            [inflate_map.margin, inflate_map.height - inflate_map.margin, inflate_map.height - inflate_map.margin,
             inflate_map.margin],
            color='black', alpha=0.9, label='margin')
    # 画出走廊
    for i in range(len(speed_corridor['x'])):
        ax.fill(speed_corridor['x'][i] + speed_corridor['r'][i] * np.cos(np.linspace(0, 2 * np.pi, 36)),
                speed_corridor['y'][i] + speed_corridor['r'][i] * np.sin(np.linspace(0, 2 * np.pi, 36)),
                'r', edgecolor='none', alpha=0.2)
    ax.plot(opt_traj['x'][::5], opt_traj['y'][::5], '.k', linewidth=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("speed_corridor.png", bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()
