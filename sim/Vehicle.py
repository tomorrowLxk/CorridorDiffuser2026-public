# sim/Vehicle.py
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, tan


class Vehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0,
                 width=2.0,  # 车辆宽度
                 length=4.2,  # 车辆长度
                 max_v=10.0,  # m/s (36 km/h)
                 min_v=-2.0,  # 支持倒车
                 max_delta=40 * np.pi / 180,  # 最大转向角 ±40°
                 max_acc=5.0, min_acc=-6.0,
                 dt=0.1,  # 仿真步长
                 rear_overhang=0.8,  # 后轮到车尾
                 front_overhang=1.0,  # 前轮到车头
                 ):
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.theta = theta  # 航向角（弧度）
        self.v = v  # 当前速度
        self.rear_overhang = rear_overhang
        self.front_overhang = front_overhang
        self.L = length - rear_overhang - front_overhang  # 轴距
        self.max_v = max_v
        self.min_v = min_v
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_delta = max_delta
        self.max_jerk = 6
        self.max_delta_dot = np.deg2rad(20.0)
        # self.max_delta_dot = np.deg2rad(30.0)
        self.dt = dt

    def step(self, a, delta):
        """
        一步动力学更新
        输入：
            a: 加速度 (m/s²)
            delta: 前轮转向角 (rad)
        输出：
            None (更新状态)
        """
        # 饱和控制输入
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        a = np.clip(a, self.min_acc, self.max_acc)
        v = self.v + a * self.dt
        self.v = np.clip(v, self.min_v, self.max_v)

        # 更新状态（一阶欧拉积分）
        self.x += self.v * cos(self.theta) * self.dt
        self.y += self.v * sin(self.theta) * self.dt
        self.theta += (self.v / self.L) * tan(delta) * self.dt

        # 标准化航向角到 [-π, π]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_position(self):
        return np.array([self.x, self.y])

    def get_state(self):
        return np.array([self.x, self.y, self.theta, self.v])

    def check_collision(self, map_obj, buffer_radius=1.5):
        """
        简化碰撞检测：用车辆中心 + 圆形缓冲区
        """
        pos = self.get_position()
        # 构造一个点作为多边形进行碰撞检测
        if map_obj.collides(np.array([pos])):
            return True
        # 可扩展：用圆形采样点检测
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for a in angles:
            offset = buffer_radius * np.array([cos(a), sin(a)])
            if map_obj.collides(np.array([pos + offset])):
                return True
        return False
