## 用于测试仿真器的脚本

from sim.Vehicle import Vehicle
from sim.ContinuousMap import map_generator
from sim.tools import sample_start_goal
from sim.tools import create_rectangle
from matplotlib import pyplot as plt
import numpy as np



MAP_W = 40
MAP_H = 40


veh = Vehicle(width=2.0, length=4.2)
m, added = map_generator(MAP_W, MAP_H, 2, 8, 4)
print(f"Generated {added} obstacles")
start_pose, goal_pose, path = sample_start_goal(m, veh, clearance=0.2)


# todo ---------- 可视化原地图、障碍物、起终点、测试路径 ----------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', 'box')
ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_title("Trajectory -> Safe Corridor (bubble/tube)")

for obs in m.obstacles:
    ax.fill(obs[:, 0], obs[:, 1], alpha=0.9)
px = [p[0] for p in path]
py = [p[1] for p in path]
ax.plot(px, py, linestyle='--', linewidth=1)
# start/goal rectangles
s_rect = create_rectangle((start_pose[0], start_pose[1]), start_pose[2], veh)
g_rect = create_rectangle((goal_pose[0], goal_pose[1]), goal_pose[2], veh)
ax.plot(np.append(s_rect[:, 0], s_rect[0, 0]), np.append(s_rect[:, 1], s_rect[0, 1]), linewidth=1.0)
ax.plot(np.append(g_rect[:, 0], g_rect[0, 0]), np.append(g_rect[:, 1], g_rect[0, 1]), linewidth=1.0)
ax.scatter([start_pose[0], goal_pose[0]], [start_pose[1], goal_pose[1]], marker='x')
plt.show()




