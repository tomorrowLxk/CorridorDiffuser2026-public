import numpy as np
import random
from sim.tools import *
import math

def generate_random_convex_polygon(center, avg_radius=5.0, radius_variation=0.2, num_points=12):
    """Create a random convex polygon around center by sampling points and taking convex hull"""
    cx, cy = center
    # sample random points in a disk-ish region
    angles = np.random.rand(num_points) * 2 * np.pi
    radii = np.random.rand(num_points) * avg_radius * radius_variation + avg_radius * (1 - radius_variation)
    pts = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
    hull = monotone_chain_convex_hull(pts)
    # ensure CCW orientation
    if not is_counter_clockwise(hull):
        hull = hull[::-1]
    return hull


def make_non_overlapping_obstacles(map_obj, num_obstacles=25, avg_radius=4.0, tries=5000):
    attempts = 0
    added = 0
    while added < num_obstacles and attempts < tries:
        sampled_radius = np.clip(random.gauss(avg_radius, avg_radius*0.1), avg_radius*0.6, avg_radius*1.4)
        attempts += 1
        # random center avoiding map borders
        cx = random.uniform(map_obj.margin + sampled_radius, map_obj.width - map_obj.margin - sampled_radius)
        cy = random.uniform(map_obj.margin + sampled_radius, map_obj.height - map_obj.margin - sampled_radius)
        poly = generate_random_convex_polygon((cx, cy), avg_radius=sampled_radius,
                                              radius_variation=0.1,
                                              num_points=random.randint(6, 12))
        if not map_obj.in_bounds(poly):
            continue
        # check overlap with existing obstacles
        ok = True
        for obs in map_obj.obstacles:
            if polygons_intersect(poly, obs):
                ok = False
                break
        if not ok:
            continue
        map_obj.add_obstacle(poly)
        added += 1
    return added


def make_overlapping_obstacles(map_obj, num_obstacles=25, avg_radius=4.0, tries=5000):
    attempts = 0
    added = 0
    while added < num_obstacles and attempts < tries:
        sampled_radius = np.clip(random.gauss(avg_radius, avg_radius*0.05), avg_radius*0.9, avg_radius*1.1)
        attempts += 1
        cx = random.uniform(map_obj.margin + sampled_radius, map_obj.width - map_obj.margin - sampled_radius)
        cy = random.uniform(map_obj.margin + sampled_radius, map_obj.height - map_obj.margin - sampled_radius)
        poly = generate_random_convex_polygon((cx, cy), avg_radius=sampled_radius,
                                              radius_variation=0.4,
                                              num_points=random.randint(3, 12))
        if not map_obj.in_bounds(poly):
            continue
        map_obj.add_obstacle(poly)
        added += 1
    return added


def _segment_to_rect(p1, p2, width):
    """Convert a line segment p1 -> p2 into a CCW rectangle polygon of given width."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-10:
        return None
    # unit perpendicular direction
    px = -dy / length * (width * 0.5)
    py =  dx / length * (width * 0.5)
    return np.array([
        [p1[0] + px, p1[1] + py],
        [p2[0] + px, p2[1] + py],
        [p2[0] - px, p2[1] - py],
        [p1[0] - px, p1[1] - py],
    ])


def generate_random_straight_wall(center, avg_radius=8.0, radius_variation=0.3,
                                  aspect_ratio_range=(8, 30)):
    """
    Generate a straight wall from a random chord within a circle.
    Returns a list containing one thin rectangle polygon.
    """
    cx, cy = center
    R = avg_radius * random.uniform(1.0 - radius_variation, 1.0 + radius_variation)
    R = max(R, 0.5)
    a1 = random.random() * 2.0 * math.pi
    a2 = random.random() * 2.0 * math.pi
    p1 = np.array([cx + R * math.cos(a1), cy + R * math.sin(a1)])
    p2 = np.array([cx + R * math.cos(a2), cy + R * math.sin(a2)])
    chord_len = np.linalg.norm(p2 - p1)
    aspect = random.uniform(*aspect_ratio_range)
    width = max(chord_len / aspect, 0.1)
    rect = _segment_to_rect(p1, p2, width)
    return [rect] if rect is not None else []


def generate_random_l_wall(center, avg_radius=8.0, radius_variation=0.3,
                            aspect_ratio_range=(8, 30)):
    """
    Generate an L-shaped wall from a diameter + a third point on the circle.
    By Thales' theorem the three points form a right angle.
    Returns two thin rectangles forming an L (right-angle corner).
    """
    cx, cy = center
    R = avg_radius * random.uniform(1.0 - radius_variation, 1.0 + radius_variation)
    R = max(R, 0.5)
    theta = random.random() * 2.0 * math.pi
    A = np.array([cx + R * math.cos(theta),             cy + R * math.sin(theta)])
    B = np.array([cx + R * math.cos(theta + math.pi),   cy + R * math.sin(theta + math.pi)])
    for _ in range(10):
        phi = random.random() * 2.0 * math.pi
        if min(abs(phi - theta) % (2.0 * math.pi), abs(phi - (theta + math.pi)) % (2.0 * math.pi)) > 0.3:
            break
    C = np.array([cx + R * math.cos(phi), cy + R * math.sin(phi)])
    len_ac = float(np.linalg.norm(C - A))
    len_cb = float(np.linalg.norm(B - C))
    avg_len = (len_ac + len_cb) * 0.5
    aspect = random.uniform(*aspect_ratio_range)
    width = max(avg_len / aspect, 0.1)
    rect1 = _segment_to_rect(A, C, width)
    rect2 = _segment_to_rect(C, B, width)
    result = []
    if rect1 is not None: result.append(rect1)
    if rect2 is not None: result.append(rect2)
    return result


def generate_random_u_wall(center, avg_radius=8.0, radius_variation=0.3,
                            aspect_ratio_range=(8, 30)):
    """
    Generate a U-shaped wall from 4 random points on the circle.
    Returns three thin rectangles forming a U.
    """
    cx, cy = center
    R = avg_radius * random.uniform(1.0 - radius_variation, 1.0 + radius_variation)
    R = max(R, 0.5)
    angles = sorted([random.random() * 2.0 * math.pi for _ in range(4)])
    pts = [np.array([cx + R * math.cos(a), cy + R * math.sin(a)]) for a in angles]
    segments = [(pts[i], pts[i + 1]) for i in range(3)]
    seg_lens = [float(np.linalg.norm(p2 - p1)) for p1, p2 in segments]
    avg_len = sum(seg_lens) / 3.0
    aspect = random.uniform(*aspect_ratio_range)
    width = max(avg_len / aspect, 0.1)
    result = []
    for p1, p2 in segments:
        rect = _segment_to_rect(p1, p2, width)
        if rect is not None:
            result.append(rect)
    return result


_WALL_GENERATORS = {
    'straight': generate_random_straight_wall,
    'L':        generate_random_l_wall,
    'U':        generate_random_u_wall,
}


def generate_random_wall(center, wall_type=None, avg_radius=8.0,
                         radius_variation=0.3, aspect_ratio_range=(8, 30)):
    """
    Generate a randomly parameterised wall of the given type.
    Returns (chosen_type, list_of_polygons).
    """
    if wall_type is None:
        wall_type = random.choice(list(_WALL_GENERATORS.keys()))
    gen = _WALL_GENERATORS[wall_type]
    return wall_type, gen(center, avg_radius, radius_variation, aspect_ratio_range)


def make_overlapping_walls(map_obj, num_walls=15, avg_radius=8.0, tries=5000):
    """Generate wall obstacles that may overlap each other (bounds only)."""
    attempts = 0
    added = 0
    while added < num_walls and attempts < tries:
        attempts += 1
        wt = random.choice(list(_WALL_GENERATORS.keys()))
        margin = avg_radius * 1.1 + map_obj.margin
        cx = random.uniform(margin, map_obj.width - margin)
        cy = random.uniform(margin, map_obj.height - margin)
        polys = _WALL_GENERATORS[wt]((cx, cy), avg_radius)
        if not all(map_obj.in_bounds(p) for p in polys):
            continue
        for p in polys:
            map_obj.add_obstacle(p)
        added += 1
    return added


def make_non_overlapping_walls(map_obj, num_walls=15, avg_radius=8.0, tries=5000):
    """Generate non-overlapping wall obstacles."""
    attempts = 0
    added = 0
    while added < num_walls and attempts < tries:
        attempts += 1
        wt = random.choice(list(_WALL_GENERATORS.keys()))
        margin = avg_radius * 1.1 + map_obj.margin
        cx = random.uniform(margin, map_obj.width - margin)
        cy = random.uniform(margin, map_obj.height - margin)
        polys = _WALL_GENERATORS[wt]((cx, cy), avg_radius)
        if not all(map_obj.in_bounds(p) for p in polys):
            continue
        ok = True
        for p in polys:
            if any(polygons_intersect(p, obs) for obs in map_obj.obstacles):
                ok = False
                break
        if not ok:
            continue
        for p in polys:
            map_obj.add_obstacle(p)
        added += 1
    return added
