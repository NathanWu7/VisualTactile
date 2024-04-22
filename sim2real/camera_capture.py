from cali.Camera import Camera
import cv2
import numpy as np
import time
import os


mtx = np.array([[909.65582275 ,  0.   ,      652.90588379],[  0.     ,    907.89691162, 365.95373535],[  0,           0,           1        ]])
# dist = np.array(([[0.028784404196194664, 0.7481844381754564, 0.0028707604214314336, -0.0032153215725527914, -3.1796489988923713]]))
dist = np.array([0.0001, 0, 0, 0, 0])
ABS = 0

os.mkdir(f'/home/zhou/graspnet-baseline-main/data_photo/{int(time.time())}')
# color_image_name = f'/home/zhou/graspnet-baseline-main/data_photo/{int(time.time())}/color.png'
# depth_image_name = f'/home/zhou/graspnet-baseline-main/data_photo/{int(time.time())}/depth.png'
color_image_name = '/home/zhou/VisualTactile-main/sim2real/color.png'
depth_image_name = '/home/zhou/VisualTactile-main/sim2real/depth.png'

def get_img():
    camera = Camera()
    cam_intrinsics = camera.get_intrinsics()
    # print("cam_intrinsics:",cam_intrinsics)
    depth_scale = camera.get_depth_scale()
    # print("depth_scale:",depth_scale)
    camera_color_img, camera_depth_img = camera.get_array()
    camera.close_camera()
    # camera_color_img = image_cut(camera_color_img)
    # camera_depth_img = image_cut(camera_depth_img)
    #畸变矫正
    h1, w1 = camera_color_img.shape[:2]     #获取彩色图片的高、宽，并且赋值给h1和w1
    h2, w2 = camera_depth_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(camera_color_img, mtx, dist, None, newcameramtx)

    newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx, dist, (h2, w2), 0, (h2, w2))
    dst2 = cv2.undistort(camera_depth_img, mtx, dist, None, newcameramtx2)

    color_frame = dst1
    depth_frame = dst2
    camera_color_img = cv2.resize(camera_color_img,(256,256))
    cv2.imwrite(color_image_name,camera_color_img)
    cv2.imwrite(depth_image_name,camera_depth_img)

def image_cut(image):
    # 定义目标图像尺寸
    target_width, target_height = 1280, 720

    # 定义截取区域的原始坐标
    x1, y1 = 100, 200  # 左上角坐标
    x2, y2 = 1000, 600  # 右下角坐标

    # 截取图像区域
    cropped_image = image[y1:y2, x1:x2]
    cropped_image = cv2.resize(cropped_image, (target_width, target_height))
    return cropped_image
if __name__=='__main__':
    get_img()