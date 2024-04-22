import cv2
import random
import numpy as np

cv2.namedWindow("process image")
cv2.namedWindow("raw image")
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
        img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
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
ref_left_path = '/home/zhou/graspnet-baseline-main/slip_detection_master/new_data/ref_left.jpg'
for i in range(1):
    raw_image_path = '/home/zhou/graspnet-baseline-main/slip_detection_master/new_data/label2/c/image'+str(10)+'_left.jpg'
    raw_image = cv2.imread(raw_image_path)
    cv2.imshow('raw image',raw_image)
    ref_left = cv2.imread(ref_left_path)
    raw_image = cv2.subtract(raw_image,ref_left)
    cv2.imshow('process image',raw_image)
    process_img = img_process(raw_image)
    process_img = add_noise(process_img, noise_type='gaussian_single', mean=0, stddev=0.27)
    cv2.imshow('process2 image',process_img)
    cv2.waitKey(500)
cv2.waitKey(0)
cv2.destroyAllWindows()