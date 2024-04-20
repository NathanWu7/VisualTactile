import numpy as np
import pybullet as pb

#初始tcp相对于基坐标系的坐标（直接通过getl获得）
init_pos = [-0.18599,-0.45162,0.151]
init_rpy = [0.0121,-3.1593,-0.016]
init_orn = pb.getQuaternionFromEuler(init_rpy)
#print(init_orn)
#init_orn = [-0.707107, -0.707107, 0.000000, 0.000000]

#movement_mode = "TxRyRz"
#movement_mode = "TyRz"
movement_mode = "xyzRxRy"
#movement_mode = "xy"


min_action, max_action = -0.25,0.25
max_pos_vel = 0.01  # m/s
max_ang_vel = 5.0 * (np.pi / 180)  # rad/s


if(movement_mode == "xyzRxRy"):
    x_act_min, x_act_max = -max_pos_vel, max_pos_vel
    y_act_min, y_act_max = -max_pos_vel, max_pos_vel
    z_act_min, z_act_max = -max_pos_vel, max_pos_vel
    roll_act_min, roll_act_max = -max_ang_vel, max_ang_vel
    pitch_act_min, pitch_act_max =  -max_ang_vel, max_ang_vel
    yaw_act_min, yaw_act_max = -0,0
    init_pos = [-0.15123,-0.57240,0.06092]
    init_rpy = [3.131571178388143, 0.017776347872228968, -0.07070360740624171]
    init_orn = pb.getQuaternionFromEuler(init_rpy)
elif(movement_mode == "TxRyRz"):
    x_act_min, x_act_max = -max_pos_vel, max_pos_vel
    y_act_min, y_act_max = -max_pos_vel, max_pos_vel
    z_act_min, z_act_max = -0, 0
    roll_act_min, roll_act_max = -0, 0
    pitch_act_min, pitch_act_max =  -max_ang_vel, max_ang_vel
    yaw_act_min, yaw_act_max = -max_ang_vel, max_ang_vel
elif(movement_mode == "TyRz"):
    x_act_min, x_act_max = -max_pos_vel, max_pos_vel
    y_act_min, y_act_max = -max_pos_vel, max_pos_vel
    z_act_min, z_act_max = -0, 0
    roll_act_min, roll_act_max = -0, 0
    pitch_act_min, pitch_act_max =  -0,0
    yaw_act_min, yaw_act_max = -max_ang_vel, max_ang_vel 
elif(movement_mode == "xy"):
    x_act_min, x_act_max = -max_pos_vel, max_pos_vel
    y_act_min, y_act_max = -max_pos_vel, max_pos_vel
    z_act_min, z_act_max = -max_pos_vel, max_pos_vel
    roll_act_min, roll_act_max = -max_ang_vel, max_ang_vel
    pitch_act_min, pitch_act_max =  -max_ang_vel, max_ang_vel
    yaw_act_min, yaw_act_max = -0,0
    init_pos = [-0.17550,-0.46783,0.33973]
    init_rpy = [0.03766096816983848, -0.027912703921114015, 0.08591420542487516]
    init_orn = pb.getQuaternionFromEuler(init_rpy)
else:
    print("choice movement")
    x_act_min, x_act_max = -0, 0
    y_act_min, y_act_max = -0, 0
    z_act_min, z_act_max = -0, 0
    roll_act_min, roll_act_max = -0, 0
    pitch_act_min, pitch_act_max =  -0,0
    yaw_act_min, yaw_act_max = -0, 0

def worldvec_to_workvec(worldframe_vec):
        """
        Transforms a vector in world frame to a vector in work frame.
        """
        worldframe_vec = np.array(worldframe_vec)
        inv_workframe_pos, inv_workframe_orn = pb.invertTransform(init_pos, init_orn)
        rot_matrix = np.array(pb.getMatrixFromQuaternion(inv_workframe_orn)).reshape(3, 3)
        workframe_vec = rot_matrix.dot(worldframe_vec)

        return np.array(workframe_vec)

def encode_TCP_frame_actions(actions,cur_tcp_orn):
        
        #print("actions: ",actions)
        encoded_actions = np.zeros(6)

        # get rotation matrix from current tip orientation
        tip_rot_matrix = pb.getMatrixFromQuaternion(cur_tcp_orn)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        # define initial vectors
        par_vector = np.array([1, 0, 0])  # outwards from tip
        perp_vector = np.array([0, -1, 0])  # perp to tip

        # find the directions based on initial vectors
        par_tip_direction = tip_rot_matrix.dot(par_vector)
        perp_tip_direction = tip_rot_matrix.dot(perp_vector)

        # transform into workframe frame for sending to robot
        workframe_par_tip_direction = worldvec_to_workvec(par_tip_direction)
        workframe_perp_tip_direction = worldvec_to_workvec(perp_tip_direction)


        if movement_mode == "TyRz":

            # translate the direction
            perp_scale = actions[0]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            par_scale = 1.0 * 0.25
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]


        elif movement_mode == "TxTyRz":

            # translate the direction
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            par_scale = actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[2]
        elif movement_mode == "TxRyRz":

            # translate the direction
            #print(perp_action)
            # perp_scale = 1.0 * self.max_action
            # perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            par_scale = actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += par_action[0]
            encoded_actions[4] += actions[1]
            encoded_actions[5] += actions[2]      
        return encoded_actions

def scale_actions(actions):
        
        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, min_action, max_action)

        input_range = max_action - min_action

        new_x_range = x_act_max - x_act_min
        new_y_range = y_act_max - y_act_min
        new_z_range = z_act_max - z_act_min
        new_roll_range = roll_act_max - roll_act_min
        new_pitch_range = pitch_act_max - pitch_act_min
        new_yaw_range = yaw_act_max - yaw_act_min

        scaled_actions = [
            (((actions[0] - min_action) * new_x_range) / input_range) + x_act_min,
            (((actions[1] - min_action) * new_y_range) / input_range) + y_act_min,
            (((actions[2] - min_action) * new_z_range) / input_range) + z_act_min,
            (((actions[3] - min_action) * new_roll_range) / input_range) + roll_act_min,
            (((actions[4] - min_action) * new_pitch_range) / input_range) + pitch_act_min,
            (((actions[5] - min_action) * new_yaw_range) / input_range) + yaw_act_min,
        ]

        return np.array(scaled_actions)

if __name__ == "__main__":

        #输入：目前的动作和目前的tcp朝向

        action = [0.25,0.25,0.25]
        #cur_tcp_rpy=[]
        #cur_tcp_orn = pb.getQuaternionFromEuler(cur_tcp_rpy)
        #cur_tcp_orn = [0.737199, 0.675675, 0.000016, -0.000080]
        cur_tcp_orn=[0.12419803778869953, 0.7673050486516748, -0.618356322181074, 0.11598822587566217]
        #通过这两个函数得到输出的动作
        encoded_action = encode_TCP_frame_actions(action,cur_tcp_orn)
        scaled_action = scale_actions(encoded_action)
        
        print(scaled_action)


