# 本脚本用于自动化生成数据集。

import cv2
import numpy as np
from tqdm import tqdm

from sim.Vehicle import Vehicle
from sim.ContinuousMap import map_generator
from sim.tools import sample_start_goal

from dataset.tools import obstacle_inflate
from dataset.tools import generate_Voronoi, generate_GVD
from dataset.tools import gvd_to_topo, proj_point2topo, update_map
from dataset.tools import generate_topo_route, generate_gvd_route

from dataset.bspline import generate_Bspline_route
from dataset.tools import curvature_spline, generate_drivable_area
from dataset.path import generate_opt_path
from dataset.trajectory import generate_opt_trajectory
from dataset.speed_corridor import generate_speed_corridor

from dataset.tools import ConMap2RasMap, Point2Raster, Corridor2Raster

i = 10000

MAP_W = 51.2
MAP_H = 51.2
veh = Vehicle(width=2.0, length=4.2)



for _ in tqdm(range(100)):
    k = 0
    flag = False
    init_map, added = map_generator(MAP_W, MAP_H, 0.6, 13, 3.5)
    inflate_map = obstacle_inflate(init_map, veh, clearance=0.2)
    start_pose, goal_pose, init_route = sample_start_goal(inflate_map, veh, clearance=0.2)
    voronoi_map = generate_Voronoi(inflate_map)
    gvd_map = generate_GVD(voronoi_map, inflate_map)
    topo_map = gvd_to_topo(gvd_map)
    s_id, g_id, s_nids, g_nids = proj_point2topo(start_pose, goal_pose, topo_map, inflate_map, angle_range=np.pi / 3)
    if s_id is None or g_id is None:
        continue
    new_gvd_map = update_map(gvd_map, start_pose, goal_pose, s_nids, g_nids)
    new_topo_map = update_map(topo_map, start_pose, goal_pose, [s_id], [g_id])
    topo_routes = generate_topo_route(new_topo_map, s_nids, g_nids)
    _, min_cost = topo_routes[0]

    for j in range(min(len(topo_routes), 4)):
        topo_route, topo_cost = topo_routes[j]
        if topo_cost >= min_cost * 1.7:
            break
        gvd_route = generate_gvd_route(new_gvd_map, topo_route)
        bs_route, control_points = generate_Bspline_route(inflate_map, topo_route, new_gvd_map, gvd_route,
                                                          corridor_margin=0.1)
        flag, min_r = curvature_spline(bs_route, veh)
        if flag is False:
            continue
        drivable_area = generate_drivable_area(inflate_map, bs_route, max_radius=6.0, step=0.4)
        opt_path = generate_opt_path(drivable_area, veh, start_pose, goal_pose, bs_route)
        if opt_path is None:
            continue
        opt_traj = generate_opt_trajectory(opt_path, inflate_map, veh)
        if opt_traj['success'] is False:
            continue
        flag = True
        traj_name = f'data/trajectories/traj_{str(i).zfill(4)}_{k}.npy'
        np.save(traj_name, opt_traj['opt_traj'], allow_pickle=True)
        speed_corridor = generate_speed_corridor(inflate_map, opt_traj['opt_traj'], drivable_area)

        if k < 1:
            ras_map = ConMap2RasMap(init_map, resolution=0.2)
            ras_point = Point2Raster(init_map, veh, start_pose, goal_pose)
            map_name = f'data/maps/map_{str(i).zfill(4)}.png'
            ras_combined = np.zeros((ras_map.shape[0], ras_map.shape[1], 3), dtype=np.uint8)
            ras_combined[:, :, 0] = ras_map
            ras_combined[:, :, 1] = ras_point[:, :, 0]  # start
            ras_combined[:, :, 2] = ras_point[:, :, 1]  # goal
            cv2.imencode('.png', ras_combined)[1].tofile(map_name)

        target_name = f'data/targets/target_{str(i).zfill(4)}_{k}.png'
        ras_sc = Corridor2Raster(init_map, speed_corridor)
        cv2.imencode('.png', ras_sc)[1].tofile(target_name)
        k += 1
    if flag:
        i += 1

# map_name = f'./dataset/dataset/map/map_{str(0).zfill(4)}.png'
# img = cv2.imread(map_name, cv2.IMREAD_COLOR)  # (H,W,3), BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 转成 RGB
# goal_map = img[:,:,0]               # R = map
# start_map = img[:,:,1]               # G = start
# ras_map = img[:,:,2]               # B = goal
#
# target_name = f'./dataset/dataset/target/target_{str(i).zfill(4)}_{0}.png'
# ras_sc = cv2.imread(target_name, cv2.IMREAD_GRAYSCALE)  # (H,W)

