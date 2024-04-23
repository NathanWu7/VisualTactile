import os
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
import pyrealsense2 as rs
import time
from utils.o3dviewer import PointcloudVisualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../Pointnet2_PyTorch/pointnet2_ops_lib"))
from pointnet2_ops import pointnet2_utils

# def rand_row(tensor, dim_needed):  
#     row_total = tensor.shape[0]
#     return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rand_row(points, n):
    m, _ = points.shape
    random_indices = np.random.choice(m, size=n, replace=False)
    return points[random_indices]

def sample_points(points, sample_num=1000, sample_mathed='furthest'):
    #print(points.shape)
    
    eff_points = points[points[:, 2]>0.04][points[:, 2]<2]
    if eff_points.shape[0] < sample_num :
        eff_points = points
    if sample_mathed == 'random':
        sampled_points = rand_row(eff_points, sample_num)
    elif sample_mathed == 'furthest':
        sampled_points = farthest_point_sample(eff_points, sample_num)
    return sampled_points

def depth_to_pointcloud(depth_image, intrinsic):
    # Create Open3D Image from depth map
    o3d_depth = o3d.geometry.Image(depth_image)

    # Get intrinsic parameters
    fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

    # Create Open3D PinholeCameraIntrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)

    # Create Open3D PointCloud object from depth image and intrinsic parameters
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

    return pcd

def save_pointcloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)

def get_pointcloud_frame():
    pass


if __name__ == "__main__":
    pointCloudVisualizer = PointcloudVisualizer()
    pointCloudVisualizerInitialized = False
    pcd = o3d.geometry.PointCloud()
    # 初始化 RealSense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    if not os.path.exists('data_0912'):
        os.makedirs('data_0912')

    subfolders = ['images', 'depths', 'point_clouds']
    for folder in subfolders:
        folder_path = os.path.join('data_0912', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    counter = 0

    try:
        while True:
        
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_depth_frame:
                continue
                
            depth_frame = frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            #cv2.resize((320,240),depth_image)
            color_image = np.asanyarray(color_frame.get_data())

            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrinsics  = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    

            pc = depth_to_pointcloud(depth_image, depth_intrinsics)
            
            #print(pc.points)
            # points = torch.tensor(pc.points)
            points = np.array(pc.points)
            #print(points.shape)
            #color = np.asarray(pc.color)

            pc_sampled = sample_points(points, sample_num=2048, sample_mathed='random')

            # print(pc_sampled.shape)

            # test = pcs[1, :, :3].cpu().numpy()
            # #print(test.shape)
            # color = pcs[1, :, 3].unsqueeze(1).cpu().numpy()
            # color = (color - min(color)) / (max(color)-min(color))
            colors = o3d.utility.Vector3dVector( [[1,0,0]])

            pcd.points = o3d.utility.Vector3dVector(list(pc_sampled))
            #pcd.colors = o3d.utility.Vector3dVector(list(colors))

            if pointCloudVisualizerInitialized == False :
                pointCloudVisualizer.add_geometry(pcd)
                pointCloudVisualizerInitialized = True
            else :
                pointCloudVisualizer.update(pcd)  

            cv2.imshow('RealSense', color_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
            #cv2.imshow('depth_color', depth_colormap)

            # 检查是否按下了 't' 键，如果按下了，就保存当前帧的 RGB、深度图和点云
            key = cv2.waitKey(1)
            if key == ord('t'):
                # 保存 RGB 图像
                rgb_file_path = os.path.join('data_0912', 'images', 'rgb_20230912_{:04d}.png'.format(counter))
                cv2.imwrite(rgb_file_path, color_image)
                print('color saved', rgb_file_path)

                # 保存深度图像
                depth_file_path = os.path.join('data_0912', 'depths', 'depth_20230912_{:04d}.png'.format(counter))
                cv2.imwrite(depth_file_path, depth_image)
                print('depth saved', depth_file_path)

                # 将点云保存为 pcd 文件
                pcd_file_path = os.path.join('data_0912', 'point_clouds', 'point_cloud_20230912_{:04d}.pcd'.format(counter))
                save_pointcloud(pc, pcd_file_path)
                print('pc saved', pcd_file_path)

                # 更新计数器
                counter += 1

            # 检查是否按下了 ESC 键，如果按下了，就退出循环
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

