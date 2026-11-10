# 读取path，衡量path到mask中心线的距离
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy import stats

RESOLUTION = 0.2


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


def collect_samples_path(data_dir, traj_dir):
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
        traj_file = traj_path / f"{sample_id}.npy"  # 格式化为 0044

        samples.append({
            'id': sample_id,
            'corridor': str(c_file),
            'planned_path': str(traj_file)
        })

    # 可选：按 ID 排序
    samples.sort(key=lambda x: x['id'])
    return samples


mean_errors = []
max_errors = []
min_errors = []
std_errors = []

H, W = 256, 256

data_dir = fr"outputs/corridors"
traj_dir = fr"outputs/paths"
samples = collect_samples_path(data_dir, traj_dir)

for sample in tqdm(samples):
    break

for sample in tqdm(samples):
    idx = sample['id']
    try:
        planned_path = np.load(sample['planned_path'], allow_pickle=True)[:, :2]
    except FileNotFoundError:
        continue
    speed_map = np.array(Image.open(sample['corridor']))[:, :, 0] / 255
    mask_map = speed_map > 0.05
    skeleton = skeletonize(mask_map)
    centerline_coords = np.argwhere(skeleton)
    center_x, center_y = grid_to_world(centerline_coords[:, 0], centerline_coords[:, 1])
    center_path = np.array([center_x, center_y]).T
    tree = cKDTree(center_path)
    distances, _ = tree.query(planned_path)

    # 3. 统计结果
    mean_error = np.mean(distances)
    max_error = np.max(distances)
    min_error = np.min(distances)
    std_error = np.std(distances)

    mean_errors.append(mean_error)
    max_errors.append(max_error)
    min_errors.append(min_error)
    std_errors.append(std_error)

    # print(f"=== 点集到点集的最短距离统计 ===")
    # print(f"平均距离 (Mean): {mean_error:.3f}")
    # print(f"最大距离 (Max):  {max_error:.3f}")
    # print(f"最小距离 (Min):  {min_error:.3f}")

np.mean(mean_errors)
np.mean(min_errors)
np.mean(max_errors)
np.mean(std_errors)

np.mean(mean_errors) - 1.96 * stats.sem(mean_errors)
np.mean(mean_errors) + 1.96 * stats.sem(mean_errors)

np.mean(min_errors) - 1.96 * stats.sem(min_errors)
np.mean(min_errors) + 1.96 * stats.sem(min_errors)

np.mean(max_errors) - 1.96 * stats.sem(max_errors)
np.mean(max_errors) + 1.96 * stats.sem(max_errors)

np.mean(std_errors) - 1.96 * stats.sem(std_errors)
np.mean(std_errors) + 1.96 * stats.sem(std_errors)
