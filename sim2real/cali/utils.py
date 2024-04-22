# from tf import transformations
import transforms3d as tfs
import math
import numpy as np
# import rospy
import time      
import cv2 
PI=3.1415926

def euler_to_matrix_rad(x, y, z):
    # T = transformations.euler_matrix(x, y, z, "sxyz")
    T = tfs.eular.euler2mat(x, y, z, "sxyz")
    return T

def matrix_to_euler_rad(matrix):
    # q = transformations.quaternion_from_matrix(matrix)
    q = tfs.quaternions.mat2quat(matrix)
    # eulers = transformations.euler_from_quaternion(q, axes='sxyz')
    eulers = tfs.quaternions.quat2mat(q, axes='sxyz')
    return eulers

def matrix_to_quaternion(matrix):
    # return transformations.quaternion_from_matrix(matrix)
    return tfs.quaternions.mat2quat(matrix)

#四元数是ijk3 也就是xyz的顺序
def quaternion_to_matrix(quat):
    # return transformations.quaternion_matrix(quat)
    return tfs.quaternions.quat2mat(quat)

def quaternion_to_euler_rad(quat):
    # return transformations.euler_from_quaternion(quat, axes='sxyz')
    return tfs.euler.quat2euler(quat,'sxyz')


def euler_to_quaternion_rad(x, y, z):
    # return transformations.quaternion_from_euler(x, y, z, axes='sxyz')
    return tfs.euler.euler2quat(x,y,z,'sxyz')

def rad_to_degree(rad):
    return rad / math.pi * 180

def degree_to_euler(degree):
    return degree / 180 * math.pi

def inverse_matrix(matrix):
    # return transformations.inverse_matrix(matrix)
    return np.linalg.inv(matrix)

#注意简单的a*b是按对应位置元素相乘
def dot_matrix(a, b):
    return np.dot(a, b)
 

def PoseToRTmatrix(x, y, z, Tx, Ty, Tz):
    R = euler_to_matrix_rad(x, y, z)
    R[0][3] = Tx
    R[1][3] = Ty
    R[2][3] = Tz
    # RT1=np.linalg.inv(RT1)
    return R
    
def RTmatrixToPose(T):
    x,y,z = matrix_to_euler_rad(T)
    Tx = T[0][3] 
    Ty = T[1][3]
    Tz = T[2][3]
    # RT1=np.linalg.inv(RT1)
    return x,y,z, Tx,Ty,Tz

def Q_PoseToRTmatrix(T,quat):
    # matrix = transformations.quaternion_matrix(quat)
    matrix = tfs.quaternions.quat2mat(quat)
    matrix[0][3] = T[0]
    matrix[1][3] = T[1]
    matrix[2][3] = T[2]
    # RT1=np.linalg.inv(RT1)
    return matrix
    
def Q_RTmatrixToPose(matrix):
    # quat = transformations.quaternion_from_matrix(matrix)
    quat = tfs.quaternions.mat2quat(matrix)
    T = [0,0,0]

    T[0] = matrix[0][3] 
    T[1] = matrix[1][3]
    T[2] = matrix[2][3]
    # RT1=np.linalg.inv(RT1)
    return T,quat

# 10.根据空间中3个点的坐标，获取平面的法向量
def plane_norm(point_a,point_b,point_c):
    vector_ab = point_b-point_a
    vector_ac = point_c-point_a
    plane_norm = np.cross(vector_ab,vector_ac)
    return plane_norm

# 1.转移矩阵的估计（求的是A相对于B坐标系的位姿（A在B下的表示））
def get_rigid_transform(A, B):                              #A.shape=B.shape=[n,3]
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)                         #压缩行，对各列求均值，返回 1* 3 矩阵
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))                    #每个点减去均值
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB)                        #[3,n]x[n,3]
    #对H进行奇异值分解，返回U, S, Vt;U大小为(M,M)，S大小为(M,N)，Vt大小为(N,N);H=UxSxVt
    #其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值
    U, S, Vt = np.linalg.svd(H)                             
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case（如果R的行列式<0）
       Vt[2,:] *= -1            #第三行乘以-1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t