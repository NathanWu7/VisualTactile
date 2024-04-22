import os
import numpy as np
import h5py
import urx
import cv2
import time
import threading
import pybullet as pb
import math

from DH_Gripper import Gripper
global real_action

# tactile.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# tactile.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# tactile.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
# tactile.set(cv2.CAP_PROP_FPS,30)
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fps = 30 
# out = cv2.VideoWriter('tactile1.avi', fourcc, fps, (640, 480))

#ac_dir = '/home/zhou/tactile_gym/tactile_gym/Privileged_learning/action/surface/surface_action.npy'
robot = urx.Robot("192.168.1.100")
robot.set_tcp((0,0,0.19,0,0,0))
robot.set_payload(2,(0,0,0.1))
reset_position=[-0.11158,-0.48746,0.24283,0.0184,3.1455,0] 


def robot_reset():
    robot.movel(reset_position,vel=0.3)

# while(1):
#     print(robot.getl())
#     time.sleep(1)

# f = h5py.File('/home/zhou/tactile_gym/tactile_gym/vae/real_train_data/simulation_all_pair.h5', 'w')


def action_limits(actions):
    max_xyz_vel,min_xyz_vel = 1.0,-1.0
    max_rpy_bel,min_rpy_vel = 5.0*3.14/180,-5.0*3.14/180
    max_close_vel,min_close_vel = 0.02,-0.02
    # 使用NumPy的clip函数将数组限制在范围内
    limited_action = np.zeros(7)
    limited_action[0:3] = np.clip(actions[0:3], min_xyz_vel, max_xyz_vel)
    limited_action[3:6] = np.clip(actions[3:6], min_rpy_vel, max_rpy_bel)
    if actions[6]>max_close_vel:
        limited_action[6] = max_close_vel
    elif actions[6]<min_close_vel:
        limited_action[6] = min_close_vel
    else:
        limited_action[6] = actions[6]

    return limited_action

def encode_TCP_frame_actions(actions,action_mode):
    encoded_actions = np.zeros(7)
    if action_mode == 'xyzrxryrz':
        encoded_actions = actions
    elif action_mode == 'xyzrz':
        encoded_actions[0:4] = actions[0:4]
        encoded_actions[6] = actions[4]
    else:
        print("action_mode error!")

    return encoded_actions

def robot_j_rad_to_angel():
    robot_j=np.zeros(6)
    robot_j[0]=robot.getj()[0]/3.14*180
    robot_j[1]=robot.getj()[1]/3.14*180
    robot_j[2]=robot.getj()[2]/3.14*180
    robot_j[3]=robot.getj()[3]/3.14*180
    robot_j[4]=robot.getj()[4]/3.14*180
    robot_j[5]=robot.getj()[5]/3.14*180

    return robot_j

def robot_j_real_to_sim():
    robot_j = robot.getj()
    tf_j=np.zeros(6)
    tf_j[0]=robot_j[0]
    tf_j[1]=robot_j[1]+1.57
    tf_j[2]=robot_j[2]
    tf_j[3]=robot_j[3]+1.57
    tf_j[4]=robot_j[4]
    tf_j[5]=robot_j[5]
    # print(tf_j)
    return tf_j

def goal_pose(delta_pose):
     # 获取当前机械臂的位置
    current_pose = robot.getl()
    current_pose=np.append(current_pose,0)
    zf = [-1,-1,1,1,1,1,1]
    delta_pose = [delta_pose[i] * zf[i] for i in range(7)]
    # 计算目标位置
    target_pose = [current_pose[i] + delta_pose[i] for i in range(7)]
    return target_pose

def delta_x_tf_speed(delta_pose):
    speed_control = np.zeros(6)
    zf = [-1, -1, 1, -1, -1, 1, 1]
    delta_pose = [delta_pose[i] * zf[i] for i in range(7)]
    t = 0.1
    speed_control = [delta_pose[i] / t for i in range(7)]
    return speed_control
     
def robot_tcp_real_to_sim(robot_tcp):
    tf_tcp = np.zeros(7)
    tf_tcp[0] = -robot_tcp[0]+0.0167
    tf_tcp[1] = -robot_tcp[1]-0.0064
    tf_tcp[2] =  robot_tcp[2]+0.8816
    quaternion = quaternion_from_rotation_vector(robot_tcp[3:6])
    tf_tcp[3] = quaternion[0]
    tf_tcp[4] = quaternion[1]
    tf_tcp[5] = quaternion[2]
    tf_tcp[6] = quaternion[3]
    return tf_tcp

#旋转矢量转四元数
def normalize_vector(vector):
    magnitude = math.sqrt(sum(component ** 2 for component in vector))
    return [component / magnitude for component in vector]


def quaternion_from_rotation_vector(rotation_vector):
    rotation_angle = math.sqrt(sum(component ** 2 for component in rotation_vector))
    rotation_axis = normalize_vector(rotation_vector)
    half_angle = rotation_angle / 2.0
    sin_half_angle = math.sin(half_angle)
    quaternion = [component * sin_half_angle for component in rotation_axis]
    quaternion.append(math.cos(half_angle))
    return quaternion

def finger_pose_real_to_sim(lf_pose,rf_pose):
    lf_pose[0] = -lf_pose[0]+0.016
    lf_pose[1] = -lf_pose[1]-0.005
    lf_pose[2] = lf_pose[2]+0.8512
    rf_pose[0] = -rf_pose[0]+0.0186
    rf_pose[1] = -rf_pose[1]-0.0064
    rf_pose[2] = rf_pose[2]+0.8512

    return lf_pose,rf_pose

def finger_sensor_pose(tcp_pose,grasp_num):
    lf_pose = np.zeros(3)
    rf_pose = np.zeros(3)
    theta = robot.getj()[5]
    r = 0.00004*grasp_num+0.0075
    lf_pose[2] = tcp_pose[2]
    rf_pose[2] = tcp_pose[2]
    lf_pose[0] = tcp_pose[0]+math.sin(theta)*r
    lf_pose[1] = tcp_pose[1]+math.cos(theta)*r
    rf_pose[0] = tcp_pose[0]-math.sin(theta)*r
    rf_pose[1] = tcp_pose[1]-math.cos(theta)*r

    #对应仿真
    #lf_pose,rf_pose = finger_pose_real_to_sim(lf_pose,rf_pose)

    return lf_pose,rf_pose


def control_thread():
    while True:
        if real_action is not None:
            #time3=time.time()
            #robot.speedl([-0.02,0,0,0,0,0],0.5,0.2)
            #robot.speedl([real_action[0],real_action[1],real_action[2],real_action[3],real_action[4],real_action[5]],0.5,0.2)
            #print(real_action)
            #time4 = time.time()
            pass  
            #time.sleep(0.008)

def robot_pose_check(real_action):
    x_pose_limit = [-0.3,0]
    y_pose_limit = [-0.8,-0.2]
    z_pose_limit = [0.08,0.35]
    check_flag = 0
    if((real_action[0]>x_pose_limit[0]) and (real_action[0]<x_pose_limit[1]) and
       (real_action[1]>y_pose_limit[0]) and (real_action[1]<y_pose_limit[1]) and
       (real_action[2]>z_pose_limit[0]) and (real_action[2]<z_pose_limit[1])):
        check_flag = 1
    else:
        check_flag = 0
        print("pose limit error!")
    return check_flag


swing = np.loadtxt('/home/zhou/VisualTactile/sim2real/action_record.txt')
gripper_2f140 = Gripper()
if __name__ == "__main__":
    robot_reset()
    # gripper_2f140.activate_init()
    gripper_2f140.open_gripper()
    grasp_num = 1000
    print("reset done!")
    real_action = None
    #使用速度控制可以开多线程
    # thread = threading.Thread(target=control_thread, daemon=True)
    # thread.start()

    for i in range(len(swing)):  #动作序列的长度
        time1=time.time()
        action = swing[i]
        # action=[0,0,0,0,0,0,0]
        # print(action)
        cmd_limit =[0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
        limited_action = action_limits(action)
        action = [limited_action[i]*cmd_limit[i] for i in range(7)]
        encoded_action = encode_TCP_frame_actions(action,action_mode='xyzrz')
        delta_action = encoded_action
        print(delta_action)
        real_action = goal_pose(delta_action)
        if robot_pose_check(real_action):
            # robot.movel([real_action[0],real_action[1],real_action[2],real_action[3],real_action[4],real_action[5]],0.5,0.01)
            pass
        #夹爪控制
        if(real_action[6]!=0):
            delta_grasp = real_action[6]/0.005/0.05*10*0.875
            grasp_num -= int(delta_grasp)
            if(grasp_num>50 and grasp_num<=1000):
                gripper_2f140.gripper_action(grasp_num)
            else:
                print("grasp error")
                break
        #real_action = delta_x_tf_speed(delta_action)
        #print(real_action)

        #输出现实观测量
        current_pose = robot.getl()
        lf_pose,rf_pose = finger_sensor_pose(robot.getl(),grasp_num)
        #现实观测量对应到仿真
        # current_pose = robot_tcp_real_to_sim(current_pose)
        # lf_pose,rf_pose = finger_pose_real_to_sim(lf_pose,rf_pose)
        # robot_j = robot_j_real_to_sim()

        # print(i,current_pose[:3])
        print(i,"lf_pose",lf_pose,"rf_pose:",rf_pose,"delta_pose:",rf_pose[1]-lf_pose[1])
        # print(robot_j)

        #控制周期设置
        time2 = time.time()
        frequence = 0.1
        time_wait = frequence - (time2-time1)

        if time_wait > 0:
            time.sleep(time_wait)
        else:
            pass

    # out.release()
    cv2.destroyAllWindows()
    robot.close()
    time.sleep(0.1)
    real_action=None


