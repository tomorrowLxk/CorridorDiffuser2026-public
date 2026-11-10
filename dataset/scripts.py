from dataset.visualize import *


MAP_W = 51.2
MAP_H = 51.2

# todo ---------- 生成：车辆、初始地图、障碍物、起终点、初始路径 ----------
from sim.Vehicle import Vehicle
from sim.ContinuousMap import map_generator
from sim.tools import sample_start_goal
veh = Vehicle(width=2.0, length=4.2)
# init_map = np.load('m.npy', allow_pickle=True).item()
# start_pose, goal_pose = np.load('o.npy')
# init_route = np.load('p.npy', allow_pickle=True).item()
init_map, added = map_generator(MAP_W, MAP_H, 0.6, 15, 4.5)
print(f"Generated {added} obstacles")
# todo ---------- 生成：膨胀地图 ----------
from dataset.tools import obstacle_inflate
inflate_map = obstacle_inflate(init_map, veh, clearance=0.2)
# todo ---------- 可视化：膨胀地图 ----------
plot_inflate_map(inflate_map, veh)
# todo ---------- 生成：障碍物 ----------
start_pose, goal_pose, init_route = sample_start_goal(inflate_map, veh, clearance=0.2)
# todo ---------- 可视化：车辆、初始地图、障碍物、起终点、初始路径 ----------
plot_init_case(init_map, veh, start_pose, goal_pose, init_route)
np.save("init_map", init_map, allow_pickle=True)
np.save("start_pose", start_pose, allow_pickle=True)
np.save("goal_pose", goal_pose, allow_pickle=True)
np.save("init_route", init_route, allow_pickle=True)



# todo ---------- 生成：Voronoi地图/GVD地图 ----------
from dataset.tools import generate_Voronoi, generate_GVD
voronoi_map = generate_Voronoi(inflate_map)
gvd_map = generate_GVD(voronoi_map, inflate_map)
# todo ---------- 可视化：GVD地图 ----------
plot_gvd(inflate_map, gvd_map)


# todo ---------- 生成：拓扑图 ----------
from dataset.tools import gvd_to_topo
topo_map = gvd_to_topo(gvd_map)
# todo ---------- 可视化：拓扑图 ----------
plot_topo_map(inflate_map, topo_map, veh, start_pose, goal_pose)


# todo ---------- 生成：包含起终点的GVD图与拓扑图 ----------
from dataset.tools import proj_point2topo, update_map
s_id, g_id, s_nids, g_nids = proj_point2topo(start_pose, goal_pose, topo_map, inflate_map, angle_range=np.pi / 3)
new_gvd_map = update_map(gvd_map, start_pose, goal_pose, s_nids, g_nids)
new_topo_map = update_map(topo_map, start_pose, goal_pose, [s_id], [g_id])
# todo ---------- 可视化：包含起终点的GVD图与拓扑图 ----------
plot_gvd(inflate_map, new_gvd_map)
plot_topo_map(inflate_map, new_topo_map, veh, start_pose, goal_pose)


# todo ---------- 生成：同伦类 ----------
from dataset.tools import generate_topo_route
topo_routes = generate_topo_route(new_topo_map, s_nids, g_nids)
print(f"一共有{len(topo_routes)}条路线！")
# todo ---------- 可视化：同伦类路径 ----------
topo_route, topo_cost = topo_routes[5]
plot_topo_route(inflate_map, new_topo_map, topo_route)


# todo ---------- 生成：gvd路线 ----------
from dataset.tools import generate_gvd_route
gvd_route = generate_gvd_route(new_gvd_map, topo_route)
# todo ---------- 可视化：gvd路线 ----------
plot_gvd_route(inflate_map, topo_map, gvd_route, s_id, g_id)


# todo ---------- 生成：贝塞尔路线 ----------
from dataset.bspline import generate_Bspline_route
bs_route, control_points = generate_Bspline_route(inflate_map, topo_route, new_gvd_map, gvd_route, corridor_margin=0.1)
# todo ---------- 可视化：贝塞尔路线 ----------
plot_bspline_route(inflate_map, topo_map, bs_route, control_points, s_id, g_id)
# todo ---------- 检验：贝塞尔路线 ----------
from dataset.tools import curvature_spline
flag, min_r = curvature_spline(bs_route, veh)


# todo ---------- 生成：贝塞尔可行驶空间 ----------
from dataset.tools import generate_drivable_area
drivable_area = generate_drivable_area(inflate_map, bs_route, max_radius=6.0, step=0.4)
# todo ---------- 可视化：贝塞尔可行驶空间 ----------
plot_drivable_area(inflate_map, drivable_area, veh, start_pose, goal_pose)


# todo ---------- 生成：最优路径 ----------
from dataset.path import generate_opt_path
opt_path = generate_opt_path(drivable_area, veh, start_pose, goal_pose, bs_route)
# todo ---------- 可视化：最优路径 ----------
plot_opt_path(inflate_map, drivable_area, opt_path)


# todo ---------- 生成：最优轨迹 ----------
from dataset.trajectory import generate_opt_trajectory
opt_traj = generate_opt_trajectory(opt_path, inflate_map, veh)
# todo ---------- 可视化：最优轨迹 ----------
plot_opt_traj(inflate_map, opt_traj)


# todo ---------- 生成：速度走廊 ----------
from dataset.speed_corridor import generate_speed_corridor
speed_corridor = generate_speed_corridor(inflate_map, opt_traj['opt_traj'], drivable_area)
# todo ---------- 可视化：速度走廊 ----------
plot_speed_corridor(inflate_map, opt_traj['opt_traj'], speed_corridor)


# todo ---------- 生成：栅格化地图 ----------
from dataset.tools import ConMap2RasMap
ras_map = ConMap2RasMap(init_map, resolution=0.2)
# todo ---------- 可视化：栅格化地图 ----------
plt.imshow(ras_map, origin='lower');plt.show()


# todo ---------- 生成：栅格化起终 ----------
from dataset.tools import Point2Raster
ras_point = Point2Raster(init_map, veh, start_pose, goal_pose)
# todo ---------- 可视化：栅格化地图 ----------
plt.imshow(ras_point[:,:,0], origin='lower');plt.show()
plt.imshow(ras_point[:,:,1], origin='lower');plt.show()


# todo ---------- 生成：栅格化速度走廊 ----------
from dataset.tools import Corridor2Raster
ras_sc = Corridor2Raster(init_map, speed_corridor)
# todo ---------- 可视化：栅格化地图 ----------
plt.imshow(ras_sc, origin='lower');plt.colorbar();plt.show()



