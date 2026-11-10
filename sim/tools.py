import numpy as np
import random
import math
from shapely.geometry import Polygon, Point


def project_polygon(poly, axis):
    """
    Project polygon onto axis (2D vector), return min,max scalars
    投影轴获得坐标，用于分离轴定理
    """
    dots = poly.dot(axis)
    return dots.min(), dots.max()


def polygons_intersect(poly1, poly2):
    """
    Separating Axis Theorem for convex polygons
    分离轴定理判断障碍物是否交叉
    """
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # axis is perpendicular to edge
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm == 0:
                continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            # if separated
            if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
                return False
    return True


def monotone_chain_convex_hull(points):
    """Compute convex hull of a set of 2D points using Andrew's monotone chain algorithm.
    使用 Andrew's Monotone Chain 算法 计算二维点集的凸包
    Input: points: (N,2) numpy array
    Output: hull as (M,2) numpy array in CCW order (no repeated start point).
    """
    pts = np.array(points)
    # remove duplicates
    pts = np.unique(pts.round(decimals=8), axis=0)
    if len(pts) <= 1:
        return pts.copy()
    # sort by x, then y
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def polygon_area(poly):
    """
    Signed area of polygon (Nx2)
    计算多边形的有符号面积
    """
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def is_counter_clockwise(poly):
    """判断多边形顶点是否为逆时针顺序"""
    return polygon_area(poly) > 0


def create_rectangle(position, theta, veh_obj, clearance=0.2):
    px, py = position
    c = math.cos(theta)
    s = math.sin(theta)
    cx = px + c * (veh_obj.length / 2.0 - veh_obj.rear_overhang)
    cy = py + s * (veh_obj.length / 2.0 - veh_obj.rear_overhang)
    l = veh_obj.length / 2.0 + clearance
    w = veh_obj.width / 2.0 + clearance
    corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w]])
    R = np.array([[c, -s], [s, c]])
    world = corners.dot(R.T) + np.array([cx, cy])
    return world


def is_pose_collision_free(map_obj, veh_obj, pose, clearance=0.2):
    x, y, theta = pose
    rect = create_rectangle((x, y), theta, veh_obj, clearance)
    if np.any(rect[:, 0] < 0) or np.any(rect[:, 0] > map_obj.width) or np.any(rect[:, 1] < 0) or np.any(
            rect[:, 1] > map_obj.height):
        return False
    return not map_obj.collides(rect)


def sample_free_pose(map_obj, veh_obj, clearance=0.2, max_tries=2000):
    for _ in range(max_tries):
        x = random.uniform(map_obj.margin + veh_obj.length / 2.0, map_obj.width - veh_obj.length / 2.0 - map_obj.margin)
        y = random.uniform(map_obj.margin + veh_obj.length / 2.0,
                           map_obj.height - veh_obj.length / 2.0 - map_obj.margin)
        theta = random.uniform(-math.pi, math.pi)
        if is_pose_collision_free(map_obj, veh_obj, (x, y, theta), clearance):
            return (x, y, theta)
    return None


# todo. 用于RRT_Connected
def steer_towards(p_from, p_to, step_size=1.0):
    fx, fy = p_from
    tx, ty = p_to
    dx = tx - fx
    dy = ty - fy
    dist = math.hypot(dx, dy)
    if dist <= step_size:
        return (tx, ty)
    else:
        ratio = step_size / dist
        return (fx + dx * ratio, fy + dy * ratio)


# todo. 用于RRT_Connected
def edge_collision_free(map_obj, p1, p2, veh_obj, clearance=0.2, step=0.5):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1e-8:
        theta = 0.0
        return is_pose_collision_free(map_obj, veh_obj, (x1, y1, theta), clearance)
    steps = max(2, int(dist / step))
    for i in range(steps + 1):
        t = i / float(steps)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        theta = math.atan2(y2 - y1, x2 - x1)
        if not is_pose_collision_free(map_obj, veh_obj, (x, y, theta), clearance):
            return False
    return True


# todo. 通过RRT_Connected快速判断是否连通，同时调整起点和终点的朝向
def rrt_connect(map_obj, start_xy, goal_xy, veh_obj,
                max_iters=2000, step_size=3.0, goal_sample_rate=0.05):
    nodes = [start_xy]
    parent = {0: None}
    for it in range(max_iters):
        sample = goal_xy if random.random() < goal_sample_rate else (
            random.uniform(0, map_obj.width), random.uniform(0, map_obj.height))
        dists = [((n[0] - sample[0]) ** 2 + (n[1] - sample[1]) ** 2) for n in nodes]
        nid = int(np.argmin(dists))
        nearest = nodes[nid]
        new = steer_towards(nearest, sample, step_size=step_size)
        if not edge_collision_free(map_obj, nearest, new, veh_obj):
            continue
        nodes.append(new)
        parent[len(nodes) - 1] = nid
        if math.hypot(new[0] - goal_xy[0], new[1] - goal_xy[1]) <= step_size:
            if edge_collision_free(map_obj, new, goal_xy, veh_obj):
                path = [goal_xy]
                cur = len(nodes) - 1
                while cur is not None:
                    path.append(nodes[cur])
                    cur = parent[cur]
                path.reverse()
                return path
    return None


# todo. 获得起点和终点
def sample_start_goal(map_obj, veh_obj, clearance=0.2, min_distance=35.0, max_attempts=400, max_distance = None):
    for attempt in range(max_attempts):
        start = sample_free_pose(map_obj, veh_obj, clearance)
        goal = sample_free_pose(map_obj, veh_obj, clearance)
        if start is None or goal is None:
            continue
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        if math.hypot(dx, dy) < min_distance:
            continue
        if max_distance is not None and math.hypot(dx, dy) > max_distance:
            continue
        test_path = rrt_connect(map_obj, (start[0], start[1]), (goal[0], goal[1]), veh_obj)
        if test_path is None:
            continue
        if len(test_path) >= 2:
            s_dir = math.atan2(test_path[2][1] - test_path[0][1], test_path[2][0] - test_path[0][0])
            g_dir = math.atan2(test_path[-1][1] - test_path[-3][1], test_path[-1][0] - test_path[-3][0])
        else:
            s_dir = start[2]
            g_dir = goal[2]
        start_pose = (start[0], start[1], s_dir)
        goal_pose = (goal[0], goal[1], g_dir)
        if not is_pose_collision_free(map_obj, veh_obj, start_pose, clearance): continue
        if not is_pose_collision_free(map_obj, veh_obj, goal_pose, clearance): continue
        return start_pose, goal_pose, test_path
    return None, None, None


# 其他的方法

def point_in_polygon(point, poly):
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-16) + xi)
        if intersect:
            inside = not inside
    return inside


def point_to_segment_distance(pt, a, b):
    # distance from pt to segment ab
    ax, ay = a
    bx, by = b
    px, py = pt
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = vx * vx + vy * vy
    if vv == 0:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)


def point_to_polygon_distance(pt, poly):
    # if inside polygon, distance 0; else min distance to edges
    if point_in_polygon(pt, poly):
        return 0.0
    dmin = float('inf')
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        d = point_to_segment_distance(pt, a, b)
        if d < dmin:
            dmin = d
    return dmin



from shapely import Point, Polygon
def point_to_polygon_distance_tongyi(pt, poly):
    p = Point(pt)
    poly_shapely = Polygon(poly)
    return p.distance(poly_shapely)