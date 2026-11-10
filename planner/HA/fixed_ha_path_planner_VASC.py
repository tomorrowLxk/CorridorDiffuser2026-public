import numpy as np
import heapq
import matplotlib.pyplot as plt
import math
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy import ndimage


RESOLUTION = 0.2
N_THETA = 36
L = 2.4
PHI_MAX = 40 / 180 * np.pi  # 40°
V_MAX = 10.0  # speed_map 最大值

# 转向动作
N_ACTION = 5
STEER_ACTIONS = [
    -PHI_MAX,
    -PHI_MAX / 2,
    0.0,
    PHI_MAX / 2,
    PHI_MAX
]


# ========================
# 工具函数
# ========================
def world_to_grid(x, y):
    i = int(np.floor(x / RESOLUTION))
    j = int(np.floor(y / RESOLUTION))
    return j, i


def grid_to_world(i, j):
    x = i * RESOLUTION + RESOLUTION / 2
    y = j * RESOLUTION + RESOLUTION / 2
    return y, x


def discretize_theta(theta):
    theta = theta % (2 * np.pi)
    return int(round(theta / (2 * np.pi) * N_THETA)) % N_THETA


def continuous_theta(idx):
    return idx / N_THETA * 2 * np.pi


def heuristic(i, j, gi, gj):
    dx = (j - gj) * RESOLUTION
    dy = (i - gi) * RESOLUTION
    return math.hypot(dx, dy) / V_MAX  # 时间下界


def my_hybrid_a_star(start, goal, mask_map, speed_map):
    H, W = mask_map.shape

    sx, sy, syaw = start
    gx, gy, gyaw = goal

    si, sj = world_to_grid(sx, sy)
    gi, gj = world_to_grid(gx, gy)
    stheta = discretize_theta(syaw)
    gtheta = discretize_theta(gyaw)

    open_set = []
    g_score = np.full((H, W, N_THETA), np.inf)
    closed = np.zeros((H, W, N_THETA), dtype=bool)
    parent = {}

    g_score[si, sj, stheta] = 0.0
    heapq.heappush(open_set, (0.0, si, sj, stheta))

    while open_set:
        _, i, j, k = heapq.heappop(open_set)
        if closed[i, j, k]:
            continue
        closed[i, j, k] = True

        # goal region
        if abs(i - gi) <= 0 and abs(j - gj) <= 0 and k == gtheta:
            return reconstruct_path(parent, (i, j, k), start), np.sum(closed)

        x, y = grid_to_world(i, j)
        theta = continuous_theta(k)
        v = max(speed_map[i, j] * 10, 0.1)
        step_length = 0.5

        o_theta = None
        if (i, j, k) in parent.keys():
            o_theta = continuous_theta(parent[(i, j, k)][2])

        for phi_idx in range(len(STEER_ACTIONS)):
            phi = STEER_ACTIONS[phi_idx]
            new_theta = theta + (step_length / L) * np.tan(phi)
            new_x = x + step_length * np.cos(theta)
            new_y = y + step_length * np.sin(theta)

            ni, nj = world_to_grid(new_x, new_y)
            if ni<0 or nj<0 or ni>=H or nj>=W:
                continue
            n_v = max(speed_map[ni, nj] * 10, 0.1)
            nk = discretize_theta(new_theta)

            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if not mask_map[ni, nj]:
                continue
            if closed[ni, nj, nk]:
                continue
            diff = 0.0
            if o_theta is not None:
                diff = ((o_theta-new_theta) + np.pi) % (2 * np.pi) - np.pi

            tentative_g = g_score[i, j, k] + 2 * step_length / (n_v + v) + 0.1*np.abs(diff)
            if tentative_g < g_score[ni, nj, nk]:
                g_score[ni, nj, nk] = tentative_g
                f = tentative_g + heuristic(ni, nj, gi, gj)
                heapq.heappush(open_set, (f, ni, nj, nk))
                parent[(ni, nj, nk)] = (i, j, k, (new_x, new_y, new_theta))

    return None, np.sum(closed)


def reconstruct_path(parent, goal_state, start):
    path = []
    cur = goal_state
    while cur in parent:
        pi, pj, pk, pose = parent[cur]
        path.append(pose)
        cur = (pi, pj, pk)
    path.reverse()
    return [start] + path


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


def dilate_occmap_circular(occ_map, margin=7):
    if margin <= 0:
        return occ_map.copy()
    y, x = np.ogrid[-margin:margin + 1, -margin:margin + 1]
    kernel = x ** 2 + y ** 2 <= margin ** 2
    dilated_bool = ndimage.binary_dilation(occ_map, structure=kernel)
    return dilated_bool.astype(occ_map.dtype)


H, W = 256, 256
jishu = 0
nums = []

data_dir = fr"outputs/corridors"
traj_dir = fr"data/trajectories"
samples = collect_samples(data_dir, traj_dir)

for sample in tqdm(samples):
    idx = sample['id']
    speed_map = np.array(Image.open(sample['c_path']))[:,:,0]/255
    sg_dict = np.load(sample['traj_path'], allow_pickle=True).item()
    mask_map = speed_map > 0.05
    start = (sg_dict['x'][0], sg_dict['y'][0], sg_dict['yaw'][0])
    goal = (sg_dict['x'][-1], sg_dict['y'][-1], sg_dict['yaw'][-1])
    c_path, num = my_hybrid_a_star(start, goal, mask_map, speed_map)
    if c_path is None:
        print(idx)
        jishu += 1
    else:
        nums.append(num)
        c_path = np.array(c_path)
        plt.imshow(speed_map, origin='lower')
        plt.plot(5 * c_path[:, 0], 5 * c_path[:, 1], 'w', alpha=0.5)
        # plt.savefig(fr'outputs/paths_no_vasc\output_{idx}.png', dpi=300, bbox_inches='tight')
        # np.save(fr"outputs/paths\{idx}.npy", c_path)
        plt.show()
        plt.clf()

print(jishu)
print(np.mean(nums))




# plt.imshow(speed_map, origin='lower')
# plt.plot(5 * path[:, 0], 5 * path[:, 1], 'w')
# plt.show()




