import pyrealsense2 as rs
import numpy as np
import time
import cv2

class Camera():
    # 1.构造函数
    def __init__(self,ih= 720,iw=1280): #720 1280
        self.img_row = ih
        self.img_col = iw

        #创建一个管道
        self.pipeline = rs.pipeline()
        #Create a config并配置要流​​式传输的管道
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.img_col, self.img_row, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.img_col, self.img_row, rs.format.bgr8, 30)

        #开始流式传输
        self.profile = self.pipeline.start(config)

        #创建对齐对象
        #rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(rs.stream.color)

        time.sleep(1)       #等待1S缓冲，为了让帧稳定
        #获取颜色和深度的框架集
        frames = self.pipeline.wait_for_frames()
        #frames.get_depth_frame（）是640x480深度图像
        time.sleep(1)
        #将深度框与颜色框对齐
        aligned_frames = align.process(frames)
        
        #获取对齐的帧
        self.aligned_depth_frame = aligned_frames.get_depth_frame() #aligned_depth_frame是640x480深度图像
        self.color_frame = aligned_frames.get_color_frame()

    # 2.获取深度尺寸
    def get_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return depth_scale

    # 3.获取相机内参
    def get_intrinsics(self):
        #获取内参
        # 从帧句柄中检索出流简介
        cprofile = self.color_frame.get_profile()
        # 包含额外视频属性的流简介实例
        cvsprofile = rs.video_stream_profile(cprofile)         
        # 得到流简介内参属性
        intrin=cvsprofile.get_intrinsics()
        intrin_mat = np.array([[intrin.fx,0,intrin.ppx],[0,intrin.fy,intrin.ppy],[0,0,1]])              
        return intrin_mat      

    # 4.获取图像信息
    def get_array(self):
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        # depth_scale = self.get_depth_scale()
        # depth_image_mile = self.depth_image.astype(float) * depth_scale  
        '''
        depth_img_uint8 = depth_img*depth_scale*1000
        depth_img_uint8 = depth_img.astype(np.uint8)
        return self.color_image
        '''   
        return self.color_image,self.depth_image            

    # 5.保存图片
    def save_pic(self,color_image_name,depth_image_name):
        
        cv2.imwrite(color_image_name,self.color_image)
        cv2.imwrite(depth_image_name,self.depth_image)
    
    # 6.彩色图像和深度图像匹配图 
    def color_match_depth(self):
        #我们将删除对象的背景
        #clipping_distance_in_meters meters away(1m外的背景去除)
        depth_scale = self.get_depth_scale()
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        #remove background - 将clips_distance以外的像素设置为灰色
        grey_color = 153
        depth_image_3d = np.dstack((self.depth_image,self.depth_image,self.depth_image)) #depth image is 1 channel, color is 3 channels
        #depth_image_3d中距离大于1m或小于0的像素赋予灰度，否则就赋予彩色的值
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, self.color_image)
        #渲染图
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.imwrite("images.png",images)

    # 7.关闭管道
    def close_camera(self):
        self.pipeline.stop()
