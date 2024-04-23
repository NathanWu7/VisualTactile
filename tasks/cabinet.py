import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from PIL import Image as Im
from tasks.base.base_task import BaseTask
from utils import o3dviewer
from utils.mimic_util import actuate, find_joints_with_dof, mimic_clip, position_check
from utils.torch_jit_utils import quat_mul, tensor_clamp, to_torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../Pointnet2_PyTorch/pointnet2_ops_lib"))
import torch
import time
#print(sys.path)
from torchvision import transforms
from pointnet2_ops import pointnet2_utils

#import numpy as np

gym_BLUE = gymapi.Vec3(0., 0., 1.)

def get_robotarm_asset(gym, sim, asset_root, robotarm_asset_file):
    """Create a robotarm asset with a linear slider."""
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    robotarm_asset = gym.load_asset(sim, asset_root, robotarm_asset_file, asset_options)
    return robotarm_asset

def get_cabinet_asset(gym, sim, asset_root, cabinet_asset_file):
    """Create a robotarm asset with a linear slider."""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = False
    cabinet_asset = gym.load_asset(sim, asset_root, cabinet_asset_file, asset_options)
    return cabinet_asset


class Cabinet(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.debug = cfg["env"]["debug"]
        self.debug_view_type = self.cfg["env"]["debug_camera_type"]
        self.dof_config = cfg["env"]["dof_config"]
        self.full_dof = cfg["env"]["full_dof"]
        self.obs_type = self.cfg["env"]["obs_type"]
        self.num_obs = cfg["env"]["numObservations"]


        print(self.obs_type)
        if "pointcloud" or "tactile" in self.obs_type:
            
        # task-specific parameters
            self.point_cloud_debug = self.cfg["env"]["Pointcloud_Visualize"]
            self.camera_view_debug = self.cfg["env"]["Camera_Visualize"]
        else: 
            print("no pointcloud")

        if self.debug == False:
            self.point_cloud_debug = False
            self.camera_view_debug = False
        else:
            plt.ion()

        self.num_obs = 25
        if self.dof_config == "XYZRxRYRz":
            self.num_act = 7  # force applied on the pole (-1 to 1)
        elif self.dof_config == "XYZRz":
            self.num_act = 5

        self.max_episode_length = cfg["env"]["episodeLength"] # maximum episode length



        # Tensor placeholders
        self.states = {} 
        
        self.cfg["env"]["numObservations"] = self.num_obs
        self.cfg["env"]["numActions"] = self.num_act
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.cfg["headless"] = headless

        self.num_envs = self.cfg["env"]["numEnvs"]
        self.arm_type = self.cfg["env"]["arm"]
        self.hand_type = self.cfg["env"]["hand"]
        self.sensor_type = self.cfg["env"]["sensor"]
        self.arm_dof = self.cfg["env"]["arm_dof"]
        self.hand_joint = self.cfg["env"]["hand_joint"]
        self.action_scale = self.cfg["env"]["actionScale"]


        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)

        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self._q = None  # Joint positions           (n_envs, n_dof)
        self.start_position_noise = 0.15
        self.start_rotation_noise = 0.785
        self._pos_control = None            # Position actions
        self.num_force_sensor = 2
        self.last_actions = None
        

        #self.device = self.cfg["env"]["sim_device"]
        super().__init__(cfg=self.cfg)
        #self.joint_limits = [[]]

        

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, spacing=2.5, num_per_row=int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        # define environment space (for visualisation)
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.transforms_depth = transforms.CenterCrop((self.sensor_cam_height,self.sensor_cam_width))
        
        self.position_limits = to_torch([self.dof_limits_low,
                                         self.dof_limits_high], device=self.device)
        self.osc_limits = to_torch([self.osc_limits_low,
                                    self.osc_limits_high], device=self.device)
        self.init_goal_pos = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        self.last_actions = torch.zeros((self.num_envs, self.full_dof), dtype=torch.float, device=self.device)

        # Set control limits
        self.cmd_limit = to_torch(self.control_limits, device=self.device).unsqueeze(0)


        asset_root = 'assets'
        robotarm_asset_file = self.arm_type+self.hand_type+self.sensor_type + '.urdf'
        cabinet_asset_file = 'sektion_cabinet_2.urdf'

        robotarm_assert = get_robotarm_asset(self.gym, self.sim, asset_root, robotarm_asset_file)
        cabinet_asset = get_cabinet_asset(self.gym, self.sim, asset_root, cabinet_asset_file)

        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        #default pos
        self.default_dof_pos = to_torch(
            self.arm_default_dof_pos +[0] * self.hand_joint #+ [0] * self.num_cabinet_dofs
            , device=self.device
        )

        self.num_dof = self.gym.get_asset_dof_count(robotarm_assert)
        self.env_dofs = self.num_cabinet_dofs + self.hand_joint + self.arm_dof
        robotarm_dof_names = self.gym.get_asset_dof_names(robotarm_assert)
        
        self.all_limits = torch.zeros((2,self.num_dof),device=self.device)

        
        # Create table asset
        self.table_stand_height = 0.83
        table_pos = [0.30, 0.365, self.table_stand_height/2]
        
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.8, 1.0, self.table_stand_height], table_opts)

        # Create table connector asset
        table_con_height = 0.02
        table_con_pos = [0.0, 0.0, self.table_stand_height + table_con_height / 2]
        table_con_opts = gymapi.AssetOptions()
        table_con_opts.fix_base_link = True
        table_con_asset = self.gym.create_box(self.sim, *[0.2, 0.15, table_con_height], table_opts)

        self.revolute_joints, self.mimic_joints, self.actuator_joints, dof = find_joints_with_dof(asset_root, robotarm_asset_file, robotarm_dof_names)

        self.all_limits = mimic_clip(self.actuator_joints, self.mimic_joints,self.arm_dof, self.all_limits, self.position_limits)
        self.num_state = 2 * self.num_dof #dof -> position speed

        # define robotarmreach pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, table_con_height + self.table_stand_height)  # generate the robotarmreach 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*table_pos)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_con_pose = gymapi.Transform()
        table_con_pose.p = gymapi.Vec3(*table_con_pos)
        table_con_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cabinet_pos = [1.10, 0.4, 0.83+0.4]
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*cabinet_pos)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        right_sensor_idx = self.gym.find_asset_rigid_body_index(robotarm_assert, "right_box")
        left_sensor_idx = self.gym.find_asset_rigid_body_index(robotarm_assert, "left_box")

        sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.02))
        sensor_pose2 = gymapi.Transform(gymapi.Vec3(-0.0, 0.0, 0.02))

        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        sensor_options.use_world_frame = False  # report forces in world frame (easier to get vertical components)

        sensor_idx1 = self.gym.create_asset_force_sensor(robotarm_assert, right_sensor_idx, sensor_pose1, sensor_options)
        sensor_idx2 = self.gym.create_asset_force_sensor(robotarm_assert, left_sensor_idx, sensor_pose2, sensor_options)
        self.net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # define robotarmreach dof properties
        dof_props = self.gym.get_asset_dof_properties(robotarm_assert)
        dof_props["driveMode"][:self.arm_dof].fill(gymapi.DOF_MODE_POS)
        dof_props["driveMode"][self.arm_dof:self.num_dof].fill(gymapi.DOF_MODE_EFFORT)

        dof_props["stiffness"][:self.arm_dof].fill(8000.0)
        dof_props["stiffness"][self.arm_dof:self.num_dof].fill(7000.0)

        dof_props["damping"][:self.arm_dof].fill(40.0)
        dof_props["damping"][self.arm_dof:self.num_dof].fill(1.0e2)

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        num_robot_bodies = self.gym.get_asset_rigid_body_count(robotarm_assert)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robotarm_assert)
        self.max_agg_bodies = num_robot_bodies + self.num_cabinet_bodies + 2     # 1 for table, table stand
        self.max_agg_shapes = num_robot_shapes + self.num_cabinet_shapes + 2     # 1 for table, table stand

        # generate environments
        self.envs = []
        self.targ_handles = []
        self.robotarm_handles = []
        self.targ_idxs = []
        print(f'Creating {self.num_envs} environments.')


        self.all_pointcloud = torch.zeros((self.num_envs, self.pointcloud_size, 3), device=self.device)
        if  "pointcloud" in self.obs_type:

            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True


            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            
            self.pointcloud_flatten = torch.zeros((self.num_envs, self.pointCloudDownsampleNum * 3), device=self.device)
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')


        else:
            self.pointcloud_flatten = None

        
        if "tactile" in self.obs_type:
            self.sensors = []     # 包含所有 sensors
            self.projs = []
            self.vinvs = []
            self.visualizers = []
            self.sensor_width = self.sensor_cam_width  # 1.65 320
            self.sensor_height = self.sensor_cam_height
            self.sensors_camera_props = gymapi.CameraProperties()
            self.sensors_camera_props.enable_tensors = True
            self.sensors_camera_props.horizontal_fov = self.sensor_cam_horizontal_fov
            self.sensors_camera_props.width = self.sensor_width
            self.sensors_camera_props.height = self.sensor_height
            
            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.sensor_pointcloud_flatten = torch.zeros((self.num_envs, self.sensor_downsample_num * 3), device=self.device)

            sensor_u = torch.arange(0, self.sensor_width, device=self.device)
            sensor_v = torch.arange(0, self.sensor_height, device=self.device)
            self.sensor_v2, self.sensor_u2 = torch.meshgrid(sensor_v, sensor_u, indexing='ij')
        else:
            self.sensor_pointcloud_flatten = None

        if self.point_cloud_debug:
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else:
            self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)
            # add robotarmreach here in each environment
            robotarm_handle = self.gym.create_actor(env_ptr, robotarm_assert, pose, "robotarmreach", i, 1, 0)
            self.robotarm_handles.append(robotarm_handle)

            self.gym.set_actor_dof_properties(env_ptr, robotarm_handle, dof_props)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            table_con_actor = self.gym.create_actor(env_ptr, table_con_asset, table_con_pose, "table_con", i, 1, 0)
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_start_pose, "cabinet", i, 2, 0)


            #camera
            if "pointcloud" in self.obs_type or "tactile" in self.obs_type:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z

            if  "pointcloud" in self.obs_type:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.2, 0.8, self.table_stand_height+0.7), gymapi.Vec3(0.7, 0.3, self.table_stand_height+0.2))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

                
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)


            if "tactile" in self.obs_type:
                # 创建 传感器 相机 handle
                # sensor_camera

                sensor_handle_1 = self.gym.create_camera_sensor(env_ptr, self.sensors_camera_props)
                right_sensor_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robotarm_handle, "right_box")
                camera_offset1 = gymapi.Vec3(0.0, -0.00, 0.00)
                camera_rotation1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), np.deg2rad(-90))
                actor_handle1 = self.gym.get_actor_handle(env_ptr, 0)

                body_handle1 = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle1, right_sensor_handle)  # right sensor
                # 将相机 handle 附着到sensor的 base_link 上
                self.gym.attach_camera_to_body(sensor_handle_1, env_ptr, body_handle1,
                                          gymapi.Transform(camera_offset1, camera_rotation1), gymapi.FOLLOW_TRANSFORM)
                self.sensors.append(sensor_handle_1)


                # 创建相机 handle sensor_camera
                sensor_handle_2 = self.gym.create_camera_sensor(env_ptr, self.sensors_camera_props)
                left_sensor_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robotarm_handle, "left_box")
                camera_offset2 = gymapi.Vec3(0.0, -0.00, 0.00)
                camera_rotation2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), np.deg2rad(-90))
                actor_handle2 = self.gym.get_actor_handle(env_ptr, 0)

                body_handle2 = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle2, left_sensor_handle)  # left sensor
                # 将相机 handle 附着到sensor的 base_link 上
                self.gym.attach_camera_to_body(sensor_handle_2, env_ptr, body_handle2,
                                          gymapi.Transform(camera_offset2, camera_rotation2), gymapi.FOLLOW_TRANSFORM)
                self.sensors.append(sensor_handle_2)



            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
        
        self.init_data()


    def init_data(self):
        env_ptr = self.envs[0] 
        robotarm_handle = 0
        #check your urdf
        num_robotarm_rigid_bodies = self.gym.get_actor_rigid_body_count(env_ptr, robotarm_handle)
        

        self.handles = {
            # robotarm
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, robotarm_handle, "ee_link"),
            "hand_left": self.gym.find_actor_rigid_body_handle(env_ptr, robotarm_handle, "left_box"),
            "hand_right": self.gym.find_actor_rigid_body_handle(env_ptr, robotarm_handle, "right_box"),

        }

        #for sensor index
        #num_rigid_bodies = num_robotarm_rigid_bodies + 3 #table tablestand 
        # self.index_rigid_bodies = {
        #     "hand": [self.handles["hand"] + i * num_rigid_bodies - 1 for i in range(self.num_envs)],
        #     "hand_left": [self.handles["hand_left"] + i * num_rigid_bodies -1 for i in range(self.num_envs)],
        #     "hand_right": [self.handles["hand_right"] + i * num_rigid_bodies -1 for i in range(self.num_envs)],
        # }

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  #including objs
        self.init_cabinet_pos = torch.tensor([1.10, 0.4, 0.83+0.4],device = self.device)

        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)      #only dof
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  #arm_hand
        
        #_net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.fs_tensor = gymtorch.wrap_tensor(force_sensor_tensor)
        self._contact_forces = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)


        #The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), 
        # linear velocity([7:10]), and angular velocity([10:13]).


        self.init_goal_pos[:,:3] = to_torch([0.74, 0.4, self.table_stand_height + 0.6], device=self.device)
        self.init_goal_pos[:,6] = to_torch([1], device=self.device)    
        self.goal_pos = self.init_goal_pos[:,:3]

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)  #pos speed
        self.robotarm_dof_state = self._dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof]
        self.robotarm_dof_pos = self.robotarm_dof_state[..., 0]
        self.robotarm_dof_vel = self.robotarm_dof_state[..., 1]
        self.cabinet_dof_state = self._dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["hand_left"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["hand_right"], :]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robotarmreach")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        self._j_eef = jacobian[:, self.arm_dof, :, :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.env_dofs), dtype=torch.float, device=self.device)
        self.touch_rate = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.full_dof), dtype=torch.float, device=self.device) #full dof

        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)  #3 num of actor
        

    def _update_states(self):
        self.states.update({
            "robotarm_dof_pos": self.robotarm_dof_pos[:, :7],
            "ee_pos": self._eef_state[:, :3],  #3
            "ee_quat": self._eef_state[:, 3:7],  #4
            "ee_lin_vel": self._eef_state[:, 7:10],  #3
            "ee_ang_vel": self._eef_state[:, 10:13],  #3
            "middle_gripper_state": (self._eef_lf_state[:,:3] + self._eef_rf_state[:,:3]) / 2. ,
            "ee_lf_pos": self._eef_lf_state[:, :3],   #3
            #"eef_lf_quat": self._eef_lf_state[:, 3:7],   #4
            "ee_rf_pos": self._eef_rf_state[:, :3], #3
            "goal_pos": self.goal_pos[:, :3],
            #"eef_rf_quat": self._eef_rf_state[:, 3:7],   #
            "last_actions": self.last_actions,  #7
            "all_pc": self.all_pointcloud,
            "touch_rate":self.touch_rate,
            "force": self._contact_forces,
            "cabinet_dof_pos": self.cabinet_dof_pos[:,2].unsqueeze(1)
        })    


    def _refresh(self):
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #self.gym.refresh_force_sensor_tensor(self.sim)
        if "pointcloud" in self.obs_type or "tactile" in self.obs_type:
            self.compute_point_cloud_observation()
        self.compute_contact_force()
        self._update_states()
        


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.success_buf[:] = compute_reach_reward(   self.reset_buf,
                                                                        self.progress_buf,
                                                                        self.states,
                                                                        self.max_episode_length)

    def compute_observations(self):
        self._refresh() #7 3      #4           #6                          #1            #3           #4
        obs =    ["robotarm_dof_pos", "ee_pos", "ee_quat",  "ee_lf_pos", "ee_rf_pos", "force", "goal_pos", "cabinet_dof_pos"]
        states = ["robotarm_dof_pos", "ee_pos", "ee_quat",  "ee_lf_pos", "ee_rf_pos", "force", "goal_pos", "cabinet_dof_pos"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        self.states_buf = torch.cat([self.states[state] for state in states], dim=-1)
        self.pointcloud_buf = self.states["all_pc"]
        #print(torch.mean(self.pointcloud_buf,axis=1))


    def reset(self, env_ids):

        multi_robotarm_ids_int32 = self._global_indices[env_ids, 0].flatten()
        multi_cabinet_ids_int32 = self._global_indices[env_ids, 3].flatten()
        multi_env_ids_int32 = torch.cat((multi_robotarm_ids_int32,multi_cabinet_ids_int32),dim=0)

        self.robotarm_dof_pos[env_ids, :] = self.default_dof_pos                          
        self.robotarm_dof_vel[env_ids, :] = torch.zeros_like(self.robotarm_dof_vel[env_ids])  
        self._pos_control[env_ids, :self.num_dof] = self.default_dof_pos             #dof_state

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])  #dof_state

        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._pos_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
 
        # # # #self.goal_pos = sampled_goal_state


        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(self._root_state),
        #     gymtorch.unwrap_tensor(multi_cabinet_ids_int32), len(multi_cabinet_ids_int32))

        #print(self.goal_pos)
        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.success_buf[env_ids] = 0

        self._refresh()
        # refresh new observation after reset
        self.compute_observations()
        # 
    
    def compute_contact_force(self):
        net_contact_force = gymtorch.wrap_tensor(self.net_contact_force_tensor).view(self.num_envs, self.max_agg_bodies, 3)
        # 计算总接触力
        left_contact_force = net_contact_force[:, self.handles["hand_left"], :]
        right_contact_force = net_contact_force[:, self.handles["hand_right"], :]
        self._contact_forces = torch.norm(left_contact_force-right_contact_force, dim=-1).unsqueeze(1)


    def compute_point_cloud_observation(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        #sensors_point_clouds = torch.zeros((self.num_envs, 2 * self.sensor_downsample_num, 3), device=self.device)
        sensors_point_clouds = self.env_origin.unsqueeze(1).repeat(1,2 * self.sensor_downsample_num,1).to(self.device)
        #all_point_clouds = torch.zeros((self.num_envs, self.pointcloud_size, 3), device=self.device)

        for i in range(self.num_envs):
            
            if  "pointcloud" in self.obs_type:
                
                # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
  
                points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], 
                                                        self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, 
                                                        self.camera_props.width, self.camera_props.height, 10, self.device).contiguous()

                #print(points.shape)
                if points.shape[0] > 0:
                    selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
                else:
                    selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
                
                point_clouds[i] = selected_points #centorids
            
            if  "tactile" in self.obs_type:
                
                # 获取接触力张量

                # self._contact_forces[:,0:3] = left_contact_force
                # self._contact_forces[:,3:6] = right_contact_force

                
                for index in range(2):
                    
                    real_index = 2*i + index

                    if torch.det(torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.sensors[real_index]))) == 0:

                        cam_vinv = torch.zeros(4,4).to(self.device)

                    else:
                        
                        cam_vinv = torch.inverse(
                            (torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.sensors[real_index])))).to(self.device)


                    #contact_force = np.dot(cam_vinv.cpu().numpy()[:3, :3], left_contact_force.reshape(3, 1))[2]
                    contact_force_tensor = torch.tensor(1).to(self.device)

                    cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], self.sensors[real_index]),
                                            device=self.device)

                    camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],
                                                                        self.sensors[real_index], gymapi.IMAGE_DEPTH)
                    torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                    torch_cam_tensor = self.transforms_depth(torch_cam_tensor) * contact_force_tensor

                    points = sensor_depth_image_to_point_cloud_GPU(torch_cam_tensor, cam_vinv,
                                                    cam_proj, self.sensor_u2, self.sensor_v2,
                                                    self.sensor_width, self.sensor_height, 0.1, self.device).contiguous()
                    if points.numel() != 0:
                        points = self.sample_points(points, sample_num=self.sensor_downsample_num, sample_mathed='random')
                        # 存储points pair
                        start_index = index * self.sensor_downsample_num
                        sensors_point_clouds[i, start_index:start_index + self.sensor_downsample_num, :] = points

                    
                # left_contact_force_cam_frame = np.dot(cam_vinvs[0], left_contact_force.reshape(3, 1))
                # right_contact_force_cam_frame = np.dot(cam_vinvs[1], right_contact_force.reshape(3, 1))
                # last_block_force = net_contact_force[:, 24, :].cpu().numpy()
                
                # print("left force", left_contact_force, left_contact_force_cam_frame.T)
                # print("right force", right_contact_force, right_contact_force_cam_frame.T)
                #print("left force", left_contact_force)
                #print("right force", right_contact_force)
                # print("last force", last_block_force)
            #print(point_clouds.size())

       #compute_tactile

        if self.camera_view_debug:
            if self.debug_view_type == "camera" and "pointcloud" in self.obs_type:
                self.camera_window = plt.figure("CAMERA_DEBUG")
                camera_rgba_image = self.camera_visulization(camera = self.cameras[0], is_depth_image=False)
                plt.imshow(camera_rgba_image)
                plt.pause(1e-9)
                plt.cla()

            elif self.debug_view_type == "sensor" and "tactile" in self.obs_type:
                self.camera_window = plt.figure("SENSOR_DEBUG")
                sensor_rgba_image = self.camera_visulization(camera = self.sensors[0], is_depth_image=True)
                plt.imshow(sensor_rgba_image)
                plt.pause(1e-9)      
                plt.cla()
            else:
                print("obs_type error!")

        if self.pointCloudVisualizer != None :
            import open3d as o3d
            #test_points = point_clouds[0, :, :3].cpu().numpy()
            if "pointcloud" in self.obs_type:
                points = np.concatenate(( point_clouds[0, :, :3].cpu().numpy() , sensors_point_clouds[0, :, :3].cpu().numpy()) , axis=0)
                #colors = plt.get_cmap(points)
                self.o3d_pc.points = o3d.utility.Vector3dVector(points)
                #self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])
            else:
                points = sensors_point_clouds[0, :, :3].cpu().numpy()
                #colors = plt.get_cmap(points)
                self.o3d_pc.points = o3d.utility.Vector3dVector(points)
                #self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])
            #self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])

            if self.pointCloudVisualizerInitialized == False :
                self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                self.pointCloudVisualizerInitialized = True
            else :
                self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)


        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)
        sensors_point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        is_zero = torch.all(sensors_point_clouds == 0, dim=-1)
        num_zero_points = torch.sum(is_zero, dim=-1)
        self.touch_rate[:,0] = (1 - num_zero_points / (self.sensor_downsample_num * 2))
        

        #self.sensor_pointcloud_flatten = sensors_point_clouds.view(self.num_envs, 2 * self.sensor_downsample_num * 3)
        self.all_pointcloud = torch.cat((point_clouds,sensors_point_clouds),dim=1)
        #print(torch.mean(self.all_pointcloud,axis=1))
        #print(self.all_pointcloud_flatten)
        #self.pc_data = point_clouds.view(self.num_envs , self.pointCloudDownsampleNum,  3)
        

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]
    
    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        #print(points.shape)
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, camera, is_depth_image=False):
        
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], camera, gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            
            
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -0.15, -0.)

            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                        to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], camera, gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)           
        
        return camera_image
       
    def safety_check(self, actions, eef_state, osc_limit):
        #force = self.fs_tensor.view(self.num_envs,self.num_force_sensor,6)
        #print(force[:,:,:3] * 1000)

        result = eef_state + actions
        exceed_limit = torch.any(result > osc_limit[1]) or torch.any(result < osc_limit[0])
        if exceed_limit:
            clamped_result = torch.clamp(result, osc_limit[0], osc_limit[1])
            adjusted_action = clamped_result - eef_state
            actions = adjusted_action

        return actions


    def pre_physics_step(self, action):

        self.actions = action
        self.last_actions = self.actions

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids) > 0:
            self.reset(env_ids)
        # apply action
        #actions = torch.zeros_like(actions,device=self.device)  #test

        actions = self.actions * self.cmd_limit / self.action_scale 
        #print(ee_end_state)
        #print(quat2euler(self.states["eef_quat"]))
        actions[:,:3] = self.safety_check(actions[:,:3], self.states["middle_gripper_state"], self.osc_limits)
         
        
        arm_dof_pos = self._dof_state[:,:self.num_dof,0]
        #check = dof_pos.clone()

        num_envs_tensor = torch.tensor(self.num_envs)
        num_dof_tensor = torch.tensor(self.num_dof)

        u_delta = control_ik(self._j_eef, actions[:,:6].unsqueeze(-1), num_envs_tensor, num_dofs=num_dof_tensor)

        # u_delta = control_ik(self._j_eef, actions[:,:6].unsqueeze(-1), self.num_envs, num_dofs=self.num_dof)
        u_delta = actuate(self.actuator_joints, self.mimic_joints, self.arm_dof, u_delta, actions[:,self.arm_dof:])
        
        check = (u_delta + arm_dof_pos).clone()
        u_offset = position_check(self.actuator_joints, self.mimic_joints, self.arm_dof, check)
        
        self._pos_control[:,:self.num_dof] = (u_delta + arm_dof_pos + u_offset)
        self._pos_control[:,:self.num_dof] = torch.clamp(self._pos_control[:,:self.num_dof], min=self.all_limits[0],max=self.all_limits[1])
        #self._pos_control[:,self.num_dof:] = 0

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))



    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward()

# define reward function using JIT
@torch.jit.script
def control_ik(j_eef, dpose, num_envs, num_dofs, damping:float=0.05):
    """Solve damped least squares, from `franka_cube_ik_osc.py` in Isaac Gym.

    Returns: Change in DOF positions, [num_envs,num_dofs], to add to current positions.
    """
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6).to(j_eef_T.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, num_dofs)
    return u

@torch.jit.script
def compute_reach_reward(reset_buf, progress_buf, states, max_episode_length):

    # type: (Tensor, Tensor, Dict[str, Tensor], float) -> Tuple[Tensor, Tensor, Tensor]
    
    #touch_rate = states["touch_rate"].squeeze(1)
    d_lf = torch.norm(states["goal_pos"] - states["ee_lf_pos"], dim=-1)
    d_rf = torch.norm(states["goal_pos"] - states["ee_rf_pos"], dim=-1)
    d_ff = torch.norm(states["ee_lf_pos"] - states["ee_rf_pos"], dim=-1)

    d_cabinet = states["cabinet_dof_pos"].squeeze(1)

    force = states["force"].squeeze(1)
    
    force[force > 200] = 200
    ungrasp = force == 0
    goal = d_cabinet > 0.1

    rew_buf = - 0.4 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) * ungrasp \
                + force * 0.0005 \
                + d_cabinet \
                + goal * 100

    #reset_buf = torch.where((progress_buf >= (max_episode_length - 1)) | (rewards > 0.8), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= (max_episode_length - 1)) | goal, torch.ones_like(reset_buf), reset_buf)
    success_buf = goal

    return rew_buf, reset_buf, success_buf

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

@torch.jit.script
def quat2euler(quat):
    """
    Converts quaternion to Euler angles.
    Args:
        quat (tensor): (..., 4) tensor where the last dimension is (x, y, z, w) quaternion

    Returns:
        tensor: (..., 3) tensor where the last dimension is (roll, pitch, yaw) Euler angles
    """
    # Extract quaternion components
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.all(torch.abs(sinp) < 1):
        pitch = torch.asin(sinp)
    else:
        pitch = torch.sign(sinp) * torch.pi / 2

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

#@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):

    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)
    
    E = torch.ones(len(X), device=device)
    position = torch.vstack((X, Y, Z, E))[:, valid]  #slow
    #next_position = torch.vstack((X, Y, Z, E))[:, valid] 
 
    position = position.permute(1, 0).to(device)

    position = position@vinv

    points = position[:, 0:3]
    #print(points.shape)
    #points = torch.zeros((65535,3)).to(device)

    return points

@torch.jit.script
def sensor_depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width: float,
                                   height: float, depth_bar: float, device: torch.device):
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)

    valid = ((-(0.005) > Z) & (Z > -(0.018)))   # 0.028
    # valid = (Z > -0.1)

    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points