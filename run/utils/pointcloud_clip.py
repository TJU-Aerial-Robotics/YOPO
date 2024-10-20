# 实飞数据训练：将全局地图裁剪并保存
# 1、注意数据收集时，地面尽量平，且需要为z=0
# 2、收集数据不平时，修改yaw_angle_radians, pitch_angle_radians平移，并与data collection一致
# 3、bug：需要打开保存的文件，手动把前面几行的double改成float...

import open3d as o3d
import numpy as np

# 1. 加载点云数据
point_cloud = o3d.io.read_point_cloud("1.pcd")  # 替换为点云文件的路径


# # 统计离群点移除滤波
# cl, ind = cropped_point_cloud.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)  # 调整参数以控制移除离群点的程度
# filtered_cloud = cropped_point_cloud.select_by_index(ind)

# 2. 定义旋转角度（偏航角和俯仰角）
yaw_angle_degrees = -15  # 偏航角（以度为单位）
pitch_angle_degrees = -3  # 俯仰角（以度为单位）
# 3. 将角度转换为弧度
yaw_angle_radians = np.radians(yaw_angle_degrees)
pitch_angle_radians = np.radians(pitch_angle_degrees)

yaw_rotation = np.array([[np.cos(yaw_angle_radians), -np.sin(yaw_angle_radians), 0],
                         [np.sin(yaw_angle_radians), np.cos(yaw_angle_radians), 0],
                         [0, 0, 1]])

pitch_rotation = np.array([[np.cos(pitch_angle_radians), 0, np.sin(pitch_angle_radians)],
                           [0, 1, 0],
                           [-np.sin(pitch_angle_radians), 0, np.cos(pitch_angle_radians)]])
# 4. 平移2米到Z方向
translation_no = np.array([0, 0, 2])  # 平移2米到Z方向

# 5. 组合旋转矩阵 R old->new
R_on = np.dot(yaw_rotation, pitch_rotation)  # 内旋是右乘，先yaw后pitch
#  P_n = (R_no * P_o.T).T + t_no = P_o * R_on + t_no
point_cloud.points = o3d.utility.Vector3dVector(np.dot(np.asarray(point_cloud.points), R_on) + translation_no)

# o3d.visualization.draw_geometries([point_cloud])


# 2. 定义裁剪范围
# 例如，裁剪一个立方体范围，这里给出立方体的最小点和最大点坐标
min_bound = np.array([-5.0, -18.0, 0])  # 最小点坐标
max_bound = np.array([150.0, 25.0, 6])    # 最大点坐标

# 3. 使用crop函数裁剪点云
cropped_point_cloud = point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

o3d.io.write_point_cloud("realworld.ply", cropped_point_cloud, write_ascii=True)

o3d.visualization.draw_geometries([cropped_point_cloud])