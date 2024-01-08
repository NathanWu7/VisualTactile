# -*- coding: utf-8 -*-
# Author：Qiwei Wu
# Email: nathan.wuqw@gmail.com
# Date：2023-12

import urdfpy
import os 
import torch

def find_joints_with_dof(asset_root, asset_file, selected_joint_names):
    #In this function, only the selected joints will be processed
    #This function categorizes these joints into two groups and saves their attributes in a dictionary

    #urdf_paths
    urdf_path = os.path.join(asset_root, asset_file)

    #using urdfpy to load urdf files
    robot = urdfpy.URDF.load(urdf_path)
    
    joints_with_dof = []
    mimic_joints = []
    actuators = []
    Dofs = 0

    for joint_name in selected_joint_names:
        joint = robot.joint_map[joint_name]
        if joint.joint_type != "fixed":
            Dofs += 1
            joint_id = Dofs
            joint_info = {
                "id": joint_id, 
                "name": joint_name,
                "type": joint.joint_type,
                "origin": joint.origin,
                "parent_link": joint.parent,
                "child_link": joint.child,
                "axis": joint.axis,
                "calibration": getattr(joint, 'calibration', None),
                "dynamics": getattr(joint, 'dynamics', None),
                "limit": getattr(joint, 'limit', None),
                "mimic": getattr(joint, 'mimic', None),
                "safety_controller": getattr(joint, 'safety_controller', None)  
            }
            joints_with_dof.append(joint_info)
            if joint.mimic:
                mimic_info = {
                    "id": joint_id,
                    "name": joint_name,
                    "actuator": getattr(joint.mimic, 'joint'),
                    "multiplier": getattr(joint.mimic, 'multiplier', None),
                    "offset": getattr(joint.mimic, 'offset', None)
                }
                mimic_joints.append(mimic_info)
            else:
                actuator_info = {
                    "id": joint_id,
                    "name": joint_name,
                }
                actuators.append(actuator_info)

    return joints_with_dof, mimic_joints, actuators, Dofs


def actuate(actuators, mimic_joints, actuated_dof, u_delta, u_act):
    #This function automatically assigns values to individual actuated joints 
    #     and mimic joints in the order specified by selected_joint_names. 
    # u_act is a open-loop control signal
   
    #actuators: dict
    #mimic_joints: dict
    #actuated_dof: int, joints

    #u_delta: tensor or ndarray, matrix
    #         Required Format [actuated:unactuated], 
    #         shape(num_envs, dof of selected actuators + dof of selected mimic joints)

    #u_act:   Control Signal for actuators, 
    #         shape(num_envs, dof of selected actuators)

    unactuated_dof = len(actuators) - actuated_dof

    for i in range(unactuated_dof):

        #for actuators
        u_delta[:, actuators[actuated_dof+i]["id"]-1] = u_act[:, i]

        #for mimic_joints
        for item in mimic_joints:
            if item.get("actuator") == actuators[actuated_dof+i]["name"]:
                u_delta[:, item["id"]-1] = u_act[:, i] * item["multiplier"] + item["offset"]

    return u_delta

def position_check(actuators, mimic_joints, actuated_dof, dof_pos):
    # this function is used for close-loop control for adaptive gripper
    # u_offset is the error between current_gripper_dof_pos and target_gripper_dof_pos 
     
    origin_pos = dof_pos.clone()
    unactuated_dof = len(actuators) - actuated_dof
    target_pos = dof_pos.clone()

    for i in range(unactuated_dof):

        #for actuators
        
        target_pos[:, actuators[actuated_dof+i]["id"]-1] = origin_pos[:, actuated_dof+i]
        
        #for mimic_joints
        for item in mimic_joints:
            if item.get("actuator") == actuators[actuated_dof+i]["name"]:
                target_pos[:, item["id"]-1] = origin_pos[:, actuated_dof+i] * item["multiplier"] + item["offset"]

    
    u_offset = target_pos-origin_pos

    return u_offset

def mimic_clip(actuators, mimic_joints,actuated_dof, all_limits, action_limits):
    #this function is used for setting limitations
    
    unactuated_dof = len(actuators) - actuated_dof

    all_limits[:,:actuated_dof] = action_limits[:,:actuated_dof]
    for i in range(unactuated_dof):
        #for actuators
        all_limits[:,actuators[actuated_dof+i]["id"]-1] = action_limits[:,actuated_dof+i]

        
        #for mimic_joints
        for item in mimic_joints:
            if item.get("actuator") == actuators[actuated_dof+i]["name"]:
                all_limits[:,item["id"]-1] = action_limits[:,actuated_dof+i] * item["multiplier"]  
                if item["multiplier"] < 0:
                    all_limits[[0,1],item["id"]-1]=all_limits[[1,0],item["id"]-1]
    return all_limits               


if __name__ == "__main__":
    asset_root = "../assets"
    asset_file = "ur5_rq.urdf"

    selected_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint', 'left_inner_finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_inner_finger_joint']
    joints_with_dof, mimic_joints, actuators, dof = find_joints_with_dof(asset_root, asset_file, selected_joint_names)
    u_delta = torch.tensor([[-0.3495, -0.6570,  0.5364,  0.0082, -0.0630,  0.0755,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0476, -0.0871, -0.0787, -0.1867, -0.0776,  0.5742,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0358, -0.1763, -0.0809, -0.0278, -0.1416,  0.4039,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0363, -0.3671,  0.2545,  0.0325, -0.1595,  0.2221,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
    u_act = torch.tensor([[-0.5],[0.5],[0.1],[0.5]])
    actuated_dof = 6 #arm dof

    print("Dof of selected joints:", dof)
    print("\n")
    if joints_with_dof:
        print("Revolute Joints:")
        for joint_info in joints_with_dof:
            print(f"Joint ID: {joint_info['id']}")
            print(f"- Joint Name: {joint_info['name']}")
            print(f"- Joint Type: {joint_info['type']}")
            print(f"- Origin: {joint_info['origin']}")
            print(f"- Parent Link: {joint_info['parent_link']}")
            print(f"- Child Link: {joint_info['child_link']}")
            print(f"- Axis: {joint_info['axis']}")
            print(f"- Limit: {joint_info['limit']}")
            print(f"- Mimic: {joint_info['mimic']}")
            print("\n")
        for mimic_info in mimic_joints:
            print(f"Mimic Joint ID: {mimic_info['id']}")
            print(f"- Joint Name: {mimic_info['name']}")
            print(f"- actuator: {mimic_info['actuator']}")
            print(f"- multiplier: {mimic_info['multiplier']}")
            print(f"- offset: {mimic_info['offset']}")
            print("\n")
        for actuator_info in actuators:
            print(f"Actuator Joint ID: {actuator_info['id']}")
            print(f"- Joint Name: {actuator_info['name']}")
            print("\n")

    else:
        print("No revolute joints found.")
    u_delta = actuate(actuators, mimic_joints, actuated_dof, u_delta, u_act)
    print("u_delta_new: ", u_delta)
    
    action_limits = torch.tensor([[0,0,0,0,0,0,0],[1,1,1,1,1,1,0.1]])
    all_limits = torch.tensor([[0.,0,0,0,0,0,
                                0,0,0,0,0,0],
                               [0,0,0,0,0,0,
                                0,0,0,0,0,0]])
    print(all_limits.size())
    all_limits = mimic_clip(actuators, mimic_joints,actuated_dof,all_limits,action_limits)
    print("limits: ", all_limits)
    u_delta = torch.clamp(u_delta, min=all_limits[0],max=all_limits[1])
    print("clip_u_delta: ", u_delta)

