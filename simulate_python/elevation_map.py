import mujoco
import numpy as np
import threading


class MujocoElevationMap:
    def __init__(self, size_x=0.6, size_y=0.6, resolution=0.06, ray_start_height=0.5):
        """
        size_x:     x 方向网格长度（米）
        size_y:     y 方向网格长度（米）
        resolution: 网格间距（米/格），例如 0.06 表示相邻采样点间距 6cm
        ray_start_height: 射线起点相对于 pelvis 的高度偏移（米）
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.ray_start_height = ray_start_height

        # 采样点数 = round(size / resolution) + 1，保证两端点都包含在内
        nx = round(size_x / resolution) + 1
        ny = round(size_y / resolution) + 1
        _x = np.linspace(-size_x / 2, size_x / 2, nx)
        _y = np.linspace(-size_y / 2, size_y / 2, ny)
        self.X_rel, self.Y_rel = np.meshgrid(_x, _y)
        self.X_rel = self.X_rel.flatten()
        self.Y_rel = self.Y_rel.flatten()
        self.nx = nx
        self.ny = ny

        self._hit_points = np.zeros((nx * ny, 3))
        self._dist = np.zeros(nx * ny)
        self._lock = threading.Lock()

        self._ray_dir = np.array([0.0, 0.0, -1.0])
        self._geomid = np.zeros(1, dtype=np.int32)
        self._geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8) 

    def update(self, model, data, robot_body_id):
        """在物理循环中调用，更新地形高度；需在持有仿真锁时调用"""
        robot_pos = data.xpos[robot_body_id].copy()
        # print(f"Robot position: {robot_pos}")
        # 从旋转矩阵提取 yaw，将网格随机器人偏航角旋转
        # data.body_xmat 是 body->world 旋转矩阵（行主序 3x3）
        # 取机器人 x 轴（前向）投影到水平面得到 yaw-only 方向
        R = data.xmat[robot_body_id].reshape(3, 3)
        fwd_h = R[:, 0].copy()
        fwd_h[2] = 0.0
        fwd_h /= np.linalg.norm(fwd_h)
        left_h = np.array([-fwd_h[1], fwd_h[0], 0.0])

        X_abs = robot_pos[0] + self.X_rel * fwd_h[0] + self.Y_rel * left_h[0]
        Y_abs = robot_pos[1] + self.X_rel * fwd_h[1] + self.Y_rel * left_h[1]
        Z_start = robot_pos[2] + self.ray_start_height
        n = len(X_abs)
        new_pts = np.empty((n, 3))
        new_dist = np.empty(n)

        for i in range(n):
            ray_pnt = np.array([X_abs[i], Y_abs[i], Z_start])
            dist = mujoco.mj_ray(
                model, data,
                pnt=ray_pnt,
                vec=self._ray_dir,
                geomgroup=self._geomgroup,
                # geomgroup=None,
                flg_static=1,
                bodyexclude=-1,
                geomid=self._geomid,
            )
            if dist >= 0:
                new_pts[i] = ray_pnt + self._ray_dir * dist
                new_dist[i] = dist - self.ray_start_height - 0.5
                # print(f"dist:{dist}")
                # print(f" dist - self.ray_start_height: {dist - self.ray_start_height}")
            else:
                new_pts[i] = [X_abs[i], Y_abs[i], robot_pos[2] - 0.8]
                new_dist[i] = -1.0

        with self._lock:
            self._hit_points[:] = new_pts
            self._dist[:] = new_dist


    # def update(self, model, data, robot_site_id):
    #     """在物理循环中调用，更新地形高度；需在持有仿真锁时调用"""
    #     robot_pos = data.site_xpos[robot_site_id].copy()
    #     # 从旋转矩阵提取 yaw，将网格随机器人偏航角旋转
    #     # data.site_xmat 是 site->world 旋转矩阵（行主序 3x3）
    #     # 取机器人 x 轴（前向）投影到水平面得到 yaw-only 方向
    #     R = data.site_xmat[robot_site_id].reshape(3, 3)
    #     fwd_h = R[:, 0].copy()
    #     fwd_h[2] = 0.0
    #     fwd_h /= np.linalg.norm(fwd_h)
    #     left_h = np.array([-fwd_h[1], fwd_h[0], 0.0])

    #     X_abs = robot_pos[0] + self.X_rel * fwd_h[0] + self.Y_rel * left_h[0]
    #     Y_abs = robot_pos[1] + self.X_rel * fwd_h[1] + self.Y_rel * left_h[1] + 0.0000125
    #     Z_start = robot_pos[2] + self.ray_start_height
    #     n = len(X_abs)
    #     new_pts = np.empty((n, 3))
    #     new_dist = np.empty(n)

    #     for i in range(n):
    #         ray_pnt = np.array([X_abs[i], Y_abs[i], Z_start])
    #         # print(f"ray_pnt: \n {ray_pnt}")
    #         dist = mujoco.mj_ray(
    #             model, data,
    #             pnt=ray_pnt,
    #             vec=self._ray_dir,
    #             geomgroup=self._geomgroup,
    #             # geomgroup=None,
    #             flg_static=1,
    #             bodyexclude=-1,
    #             geomid=self._geomid,
    #         )
    #         # print(f"Ray {i}: start={ray_pnt}, dist={dist}, hit_geom={self._geomid[0]}")
    #         # geomid==-1: 未命中任何物体
    #         # geom_bodyid==0: 命中世界/地形（floor、hfield、障碍物均挂在 body 0）
    #         # geom_bodyid>0: 命中了机器人自身某个刚体，忽略
    #         # if dist >= 0 and model.geom_bodyid[self._geomid[0]] == 0:
    #         #     new_pts[i] = ray_pnt + self._ray_dir * dist
    #         # else:
    #         #     new_pts[i] = [X_abs[i], Y_abs[i], robot_pos[2] - 0.8]
    #         if dist >= 0:
    #             new_pts[i] = ray_pnt + self._ray_dir * dist
    #             new_dist[i] = dist - self.ray_start_height - 0.5
    #         else:
    #             new_pts[i] = [X_abs[i], Y_abs[i], robot_pos[2] - 0.8]
    #             new_dist[i] = -1.0

    #     with self._lock:
    #         self._hit_points[:] = new_pts
    #         self._dist[:] = new_dist
    #         # print(f"dist:\n{self._dist}")
    #         # print(f"dist_max:{np.max(self._dist)}, dist_min:{np.min(self._dist)}, dist_mean:{np.mean(self._dist)}")
    #         # print(f"hit points:\n{self._hit_points}")



    def get_hit_points(self):
        """线程安全地获取最新高程点（相对高度 = hit_points[:,2] - robot_z）"""
        with self._lock:
            return self._hit_points.copy()

    def get_dist(self):
        """线程安全地获取最新射线距离数组（未命中为 -1）"""
        with self._lock:
            np_array = self._dist.reshape(self.ny, self.nx)
            # print(f"dist_max:{np.max(self._dist)}, dist_min:{np.min(self._dist)}")
            # np.save("elevation_map_dist.npy", np_array)
            return self._dist.copy()
        

    def add_visual_markers(self, scene):
        """在渲染循环中调用，将高程点绘制为红色小球"""
        # 每帧必须先清零，否则 user_scn 会不断累积直到超出 maxgeom
        scene.ngeom = 0

        with self._lock:
            pts = self._hit_points.copy()

        for pt in pts:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco._enums.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.02, 0.02],
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
            scene.ngeom += 1
