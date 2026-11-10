import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def plot_map(map_obj, start=None, goal=None, path=None, corridor_polygons=None, title="Map"):
    """
    Visualize the continuous map with obstacles, start/goal, path, and corridors.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw obstacles
    for obs in map_obj.obstacles:
        poly = Polygon(obs, closed=True, color='gray', alpha=0.8, edgecolor='k', linewidth=1)
        ax.add_patch(poly)

    # Draw map boundary
    ax.set_xlim(0, map_obj.width)
    ax.set_ylim(0, map_obj.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Draw start/goal
    if start is not None:
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start', zorder=5)
    if goal is not None:
        ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal', zorder=5)

    # Draw path
    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=3, label='Path', zorder=4)

    # Draw corridor polygons (e.g., safety bubbles)
    if corridor_polygons is not None:
        for cor in corridor_polygons:
            cor = np.array(cor)
            poly = Polygon(cor, closed=True, color='lightblue', alpha=0.5,
                           edgecolor='blue', linewidth=2, zorder=3)
            ax.add_patch(poly)

    plt.legend()
    plt.title(title)
    plt.show()


# todo. 绘制地图与栅格地图
def plot_raster_comparison(map_obj, grid, title="Raster vs Continuous"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 左图：原始连续地图
    for obs in map_obj.obstacles:
        poly = plt.Polygon(obs, color='gray', alpha=0.8, edgecolor='k')
        ax1.add_patch(poly)
    ax1.set_xlim(0, map_obj.width)
    ax1.set_ylim(0, map_obj.height)
    ax1.set_aspect('equal')
    ax1.set_title("Continuous Map")
    ax1.grid(True, alpha=0.3)

    # 右图：栅格地图
    ax2.imshow(grid, cmap='gray_r', origin='lower',
               extent=[0, map_obj.width, 0, map_obj.height])
    ax2.set_title("Raster Map")
    ax2.grid(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


