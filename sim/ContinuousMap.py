import numpy as np
from sim.tools import polygons_intersect
from sim.Obstacle import make_non_overlapping_obstacles, make_overlapping_obstacles
from sim.Obstacle import make_overlapping_walls, make_non_overlapping_walls


class ContinuousMap:
    def __init__(self, width=40.0, height=40.0, margin=1.0):
        self.width = width
        self.height = height
        self.margin = margin
        self.obstacles = []

    def in_bounds(self, poly):
        # check polygon within [margin, width-margin] x [margin, height-margin]
        if np.any(poly[:, 0] < self.margin - 1e-8) or np.any(poly[:, 0] > self.width - self.margin + 1e-8):
            return False
        if np.any(poly[:, 1] < self.margin - 1e-8) or np.any(poly[:, 1] > self.height - self.margin + 1e-8):
            return False
        return True

    def add_obstacle(self, poly):
        self.obstacles.append(poly)

    def collides(self, poly):
        # check if poly intersects any obstacle
        for obs in self.obstacles:
            if polygons_intersect(poly, obs):
                return True
        return False


def map_generator(MAP_W, MAP_H, MAP_M, OBS_N, OBS_R):
    m = ContinuousMap(width=MAP_W, height=MAP_H, margin=MAP_M)
    # added = make_non_overlapping_obstacles(m, num_obstacles=OBS_N, avg_radius=OBS_R, tries=5000)
    # added = make_overlapping_obstacles(m, num_obstacles=OBS_N, avg_radius=OBS_R, tries=5000)
    # added = make_non_overlapping_walls(m, num_walls=OBS_N, avg_radius=OBS_R, tries=5000)
    added = make_overlapping_walls(m, num_walls=OBS_N, avg_radius=OBS_R, tries=5000)

    # print(f"Generated {added} obstacles")
    return m, added

# m, added = map_generator(40, 40, 2, 8, 4)
