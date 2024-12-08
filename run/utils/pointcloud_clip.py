"""
算法具有一定的Sim2Real的泛化能力, 如果有条件可用雷达+深度相机收集数据, 合并至仿真数据集中一同训练, 以进一步保证实飞的可靠性
# (1) 运行雷达里程计以记录无人机状态和地图真值. 注意保证地图和里程计处于同一坐标系，请在一次运行中同时记录图像与里程计的rosbag + 保存地图
# (可选) 运行本文件对地图进行降噪, 并可修改translation_no和R_no(yaw, pitch, roll)对地图进行变换，修正里程计漂移导致的地图倾斜，注意与data_collection_realworld一致
        (BUG: 打开保存的地图ply文件，手动把前面几行的double改成float)
# (3) 播包rosbag, 运行data_collection_realworld, 记录位置、姿态、图像，保存至save_dir
"""

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

R_no = Rotation.from_euler('ZYX', [0.0, 0.0, 0.0], degrees=True)  # yaw, pitch, roll
translation_no = np.array([0.0, 0.0, 0.0])

# 0. 加载点云数据
point_cloud = o3d.io.read_point_cloud("map_original.pcd")  # 替换为点云文件的路径

# 1. 统计离群点移除滤波
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=6, std_ratio=2.0)
point_cloud = point_cloud.select_by_index(ind)

# 2. 旋转地图以进行矫正
#  P_n = (R_no * P_o.T).T + t_no = P_o * R_on + t_no
R_on = R_no.inv().as_matrix()
point_cloud.points = o3d.utility.Vector3dVector(np.dot(np.asarray(point_cloud.points), R_on) + translation_no)

# o3d.visualization.draw_geometries([point_cloud])

# 3. 裁剪点云无关区域
min_bound = np.array([-50.0, -50.0, -1])
max_bound = np.array([50.0, 50.0, 6])

cropped_point_cloud = point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

o3d.io.write_point_cloud("map_processed.ply", cropped_point_cloud, write_ascii=True)

o3d.visualization.draw_geometries([cropped_point_cloud])