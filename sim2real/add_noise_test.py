import cv2
import random
import numpy as np
import open3d as o3d
from gelsight import gsdevice
import time

# cv2.namedWindow("process image")
# cv2.namedWindow("raw image")
# cv2.namedWindow("raw image")
def add_noise(image, noise_type='gaussian', mean=0, stddev=10):
        if noise_type == 'gaussian':
            row, col, ch = image.shape
            #np.random.seed(42)  # 设置随机种子
            noisy_image = np.random.normal(mean, stddev, (row*col*ch))#.astype(np.uint8)
            noise_dim = int(row*col*ch*0.3)
            noisy_list = [1]*noise_dim+[0]*(row*col*ch-noise_dim)
            random.shuffle(noisy_list)
            #noise_value = (np.random.rand(row*col*ch) * 2 - 1 ) * 0.08
            noisy_image = noisy_image*noisy_list
            noisy_image = noisy_image.reshape(row,col,ch).astype(np.uint8)
            noisy_image = cv2.add(image, noisy_image)
            return noisy_image
        elif noise_type == 'gaussian_single':
            row, col = image.shape
            #np.random.seed(42)  # 设置随机种子
            noisy_image = np.random.normal(mean, stddev, (row*col))#.astype(np.uint8)
            noise_dim = int(row*col*0.3)
            noisy_list = [1]*noise_dim+[0]*(row*col-noise_dim)
            random.shuffle(noisy_list)
            #noise_value = (np.random.rand(row*col*ch) * 2 - 1 ) * 0.08
            noisy_image = noisy_image*noisy_list
            noisy_image = noisy_image.reshape(row,col).astype(np.uint8)
            noisy_image = cv2.add(image, noisy_image)
            return noisy_image
        else:
            raise ValueError('Invalid noise type')
def img_process(image):
        #img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        img = image
        # 将像素值映射到0-255范围
        min_val = np.min(img)
        max_val = np.max(img)
        print(max_val,min_val)
        if(max_val>20):
            scaled_image = ((img - min_val) / (max_val - min_val)) * 255

            # 将数据类型转换为无符号8位整数
            scaled_image = scaled_image.astype(np.uint8)
        else:
             scaled_image=img
            
        return scaled_image
def depth_from_images(image, reference_image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像差异
    diff = cv2.absdiff(gray, reference_gray)
    
    # 将差异图像转换为深度图像
    depth_image = cv2.bitwise_not(diff)  # 反转图像

    # 返回深度图像
    return depth_image

def depth_to_point_cloud(depth_image, focal_length, principal_point):
    # 获取图像尺寸
    height, width = depth_image.shape
    
    # 生成网格点坐标
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # 计算相机坐标系下的点云坐标
    x = (x_coords - principal_point[0]) * depth_image / focal_length
    y = (y_coords - principal_point[1]) * depth_image / focal_length
    z = depth_image
    # for i in range(len(z)):
    #      for j in range(len(z[i])):
    #         if (z[i][j]>200 and z[i][j]<250):
    #             pass
    #         else:
    #             z[i][j]=255
    
    # 将点云坐标堆叠成一个点云数组
    point_cloud = np.dstack((x, y, z)).reshape(-1, 3)
    #eff_points = point_cloud[point_cloud[:,2]>200 and point_cloud[:,2]<250]
    
    return point_cloud

def sample_point(point_cloud,num_samples=64):

    # 对点云进行随机采样
    sampled_points = np.random.choice(point_cloud.shape[0], size=num_samples, replace=False)
    sampled_point_cloud = point_cloud[sampled_points]

    return sampled_point_cloud     

def real_sensor_depth_to_point(tactile_img,ref_img,sample_num):
    #cv2.imshow('raw image',tactile_img)
    image = tactile_img
    # 设置相机参数
    focal_length = 500.0  # 焦距
    principal_point = (image.shape[1] / 2, image.shape[0] / 2)  # 主点
        # 获取深度图像
    depth_image = depth_from_images(image, ref_img)
    cv2.imshow('depth image',depth_image)
    depth_image = img_process(depth_image)
    cv2.imshow("process img",depth_image)
    cv2.waitKey(0)
    # 将深度图像转换为点云
    point_cloud = depth_to_point_cloud(depth_image, focal_length, principal_point)
    # 创建Open3D的点云数据结构
    point_cloud = sample_point(point_cloud,sample_num)
    #读取图片开启下面代码
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到可视化窗口
    vis.add_geometry(pcd)

    # 设置相机视角
    ctr = vis.get_view_control()

    ctr.set_lookat([0, 0, 0])  # 视点位置
    ctr.set_front([0, 0, -1])  # 相机方向
    ctr.set_up([0, -1, 0])  # 相机上方向

    # 渲染并显示点云
    vis.run()
    vis.destroy_window()
    return point_cloud

if __name__ == "__main__":
    #ref_left_path = '/home/zhou/graspnet-baseline-main/slip_detection_master/new_data/ref_right.jpg'
    ref_left_path = '/home/zhou/VisualTactile-main_new/sim2real/ref_img_left.jpg'
    raw_image_path = '/home/zhou/graspnet-baseline-main/slip_detection_master/new_data/label2/c/image0_left.jpg'
    # raw_image = cv2.imread(raw_image_path)
    # ref_left = cv2.imread(ref_left_path)
    # raw_image = cv2.subtract(raw_image,ref_left)
    # cv2.imshow('raw image',raw_image)
    # process_img = img_process(raw_image)
    # #process_img = add_noise(process_img, noise_type='gaussian_single', mean=0, stddev=0.25)
    # cv2.imshow('process image',process_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
     # 读取图像
    image = cv2.imread(raw_image_path)
    reference_image = cv2.imread(ref_left_path)
    real_sensor_depth_to_point(image,reference_image,sample_num=1000)

    #连接实际摄像头
    # right_device_type = "R0"
    # right_device_params = 3

    # # 打开右侧摄像头
    # cap_right = gsdevice.Camera(right_device_params,dev_type=right_device_type)
    # cap_right.connect()
    # pcd = o3d.geometry.PointCloud()
    # # 创建Open3D可视化窗口
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 设置相机视角
    # ctr = vis.get_view_control()

    # ctr.set_lookat([0, 0, 0])  # 视点位置
    # ctr.set_front([0, 0, -1])  # 相机方向
    # ctr.set_up([0, -1, 0])  # 相机上方向
    # while True:
    #     frame_right = cap_right.get_image()
    #     #cap_right.save_image('/home/zhou/VisualTactile-main/sim2real/ref_img.jpg')
    #     points = real_sensor_depth_to_point(frame_right,reference_image,sample_num=1000)
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     vis.add_geometry(pcd)

    #     # 更新可视化窗口
    #     # vis.update_geometry(pcd)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     #vis.run()