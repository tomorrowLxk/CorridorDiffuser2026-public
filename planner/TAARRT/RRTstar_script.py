from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from tqdm import tqdm
from pathlib import Path
import re


def is_in_corridor(x, y, cor_arr):
    gx = int(np.floor(x * 5))
    gy = int(np.floor(y * 5))
    if gx < 0 or gy < 0 or gx >= cor_arr.shape[1] or gy >= cor_arr.shape[0]:
        return False
    return cor_arr[gy, gx] > 0


def line_collision_check(p1, p2, cor_arr):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 0.1:
        return True
    steps = int(dist * 10)
    for i in range(steps + 1):
        x = p1[0] + (p2[0] - p1[0]) * i / steps
        y = p1[1] + (p2[1] - p1[1]) * i / steps
        if not is_in_corridor(x, y, cor_arr):
            return False
    return True


class Node:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost


def sample_in_corridor(goal, goal_rate=0.05):
    if random.random() < goal_rate:
        return goal
    idx = random.randint(0, len(free_points) - 1)
    base = free_points[idx]
    jitter = (np.random.rand(2) - 0.5) * 0.2
    return (base[0] + jitter[0], base[1] + jitter[1])


def nearest(nodes, rnd):
    dlist = [math.hypot(n.x - rnd[0], n.y - rnd[1]) for n in nodes]
    min_idx = int(np.argmin(dlist))
    return nodes[min_idx]


def near(nodes, new_node, radius):
    neighbors = []
    for n in nodes:
        if math.hypot(n.x - new_node.x, n.y - new_node.y) <= radius:
            neighbors.append(n)
    return neighbors


def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path

def collect_samples(data_dir, traj_dir):
    data_path = Path(data_dir)
    traj_path = Path(traj_dir)

    # Step 1: 找出所有 _c.png 文件（作为 ID 的基准）
    c_files = list(data_path.glob('*_ALL.png'))

    samples = []
    for c_file in c_files:
        # 提取 ID：例如 '44_c.png' -> '44'
        stem = c_file.stem  # '44_c'
        match = re.match(r'(\d+)_ALL', stem)
        if not match:
            continue
        id_str = match.group(1)
        sample_id = int(id_str)

        # 构造其他文件名
        traj_file = traj_path / f"traj_{int(id_str):04d}_0.npy"  # 格式化为 0044

        samples.append({
            'id': sample_id,
            'c_path': str(c_file),
            'traj_path': str(traj_file)
        })

    # 可选：按 ID 排序
    samples.sort(key=lambda x: x['id'])
    return samples

def rrt_star(start, goal, cor_arr):
    nodes = [Node(start[0], start[1])]
    best_cost = float('inf')
    cost_history = []

    for i in range(max_iter):
        rnd = sample_in_corridor(goal)
        nearest_node = nearest(nodes, rnd)

        theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
        new_x = nearest_node.x + step_size * math.cos(theta)
        new_y = nearest_node.y + step_size * math.sin(theta)
        new_node = Node(new_x, new_y)

        if not is_in_corridor(new_x, new_y, cor_arr):
            continue
        if not line_collision_check((nearest_node.x, nearest_node.y),
                                    (new_node.x, new_node.y), cor_arr):
            continue

        # Step 1. 初始化父节点与代价
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + step_size

        # Step 2. 搜索邻居，尝试更优父节点
        neighbors = near(nodes, new_node, r_rewire)
        for nb in neighbors:
            new_cost = nb.cost + math.hypot(nb.x - new_node.x, nb.y - new_node.y)
            if new_cost < new_node.cost and line_collision_check((nb.x, nb.y), (new_node.x, new_node.y), cor_arr):
                new_node.parent = nb
                new_node.cost = new_cost

        nodes.append(new_node)

        # Step 3. 尝试重新连接邻居
        for nb in neighbors:
            potential_cost = new_node.cost + math.hypot(nb.x - new_node.x, nb.y - new_node.y)
            if potential_cost < nb.cost and line_collision_check((nb.x, nb.y), (new_node.x, new_node.y), cor_arr):
                nb.parent = new_node
                nb.cost = potential_cost

        # Step 4. 判断是否到达目标
        if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) < goal_threshold:
            total_cost = new_node.cost + math.hypot(new_node.x - goal[0], new_node.y - goal[1])
            if total_cost < best_cost:
                best_cost = total_cost
                goal_node = Node(goal[0], goal[1], new_node, total_cost)
                cost_history.append(best_cost)

        # Step 5. 可视化进度
        # if i % 200 == 0:
        #     plt.clf()
        #     plt.imshow(cor_arr, origin='lower', cmap='gray')
        #     plt.plot([n.x / 0.2 for n in nodes],
        #              [n.y / 0.2 for n in nodes], "y.", markersize=1)
        #     plt.plot(start[0] / 0.2, start[1] / 0.2, "go")
        #     plt.plot(goal[0] / 0.2, goal[1] / 0.2, "ro")
        #     plt.pause(0.001)

        # Step 6. 收敛判断
        if len(cost_history) >= 5:
            diff = np.abs(np.diff(cost_history[-10:])).mean()
            if diff < cost_threshold:
                # print(f"RRT* converged at iteration {i}, cost change {diff:.4f}")
                return nodes, goal_node

    print("RRT* ended, best cost =", best_cost)
    return nodes, goal_node if best_cost < float('inf') else None



step_size = 1.0
max_iter = 5000
goal_threshold = 0.3
r_rewire = 1.5  # 重连半径（影响收敛速度）
cost_threshold = 0.5  # 收敛判断用

jishu = 0
time_cost = []

data_dir = fr"outputs/corridors"
traj_dir = fr"data/trajectories"
samples = collect_samples(data_dir, traj_dir)
time_cost = []
for sample in tqdm(samples):
    idx = sample['id']
    speed_map = np.array(Image.open(sample['c_path']))[:,:,0]/255
    sg_dict = np.load(sample['traj_path'], allow_pickle=True).item()

    free_idx = np.argwhere(speed_map > 0.05)
    free_points = free_idx * 0.2 + 0.1
    free_points = np.flip(free_points, axis=1)

    start = (sg_dict['x'][0], sg_dict['y'][0])
    goal = (sg_dict['x'][-1], sg_dict['y'][-1])
    t1 = time.time()
    nodes, goal_node = rrt_star(start, goal, speed_map)
    if goal_node:
        path = extract_path(goal_node)
        t2 = time.time()
        time_cost.append(1000*(t2 - t1))
    else:
        jishu+=1

np.mean(time_cost)


