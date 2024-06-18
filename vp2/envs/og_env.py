import json
import os
import yaml

import numpy as np
import gym.spaces as spaces


from hydra.utils import to_absolute_path
from scipy.spatial.transform import Rotation as R

from vp2.envs.base import BaseEnv
from vp2.mpc.utils import resize_np_image_aa

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet

class OGEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.env_hparams = kwargs
        self.og = og
        self.concat_imgs = []

        # Load the config
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Update it to create a custom environment and run some actions
        config["scene"]["scene_model"] = "Rs_int"
        config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
        
        # remove later
        config["scene"]['scene_file'] = '/home/arpit/test_projects/OmniGibson/small_grasp_dataset_test/episode_00008_start.json'
        
        config["objects"] = [
            {
                "type": "PrimitiveObject",
                "name": "box",
                "primitive_type": "Cube",
                "rgba": [1.0, 0, 0, 1.0],
                "scale": [0.1, 0.05, 0.1],
                # "size": 0.05,
                "position": [-0.5, -0.7, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "scale": [0.3, 0.3, 0.3],
                "position": [-0.7, 0.5, 0.2],
                "orientation": [0, 0, 0, 1]
            }
        ]

        self.keys_to_take = dict(
            rgb=f'{kwargs["camera_names"][0]}_image',
        )

        if "depth" in kwargs["planning_modalities"]:
            camera_depths = [True]
            self.keys_to_take["depth"] = f'{kwargs["camera_names"][0]}_depth'
        else:
            camera_depths = [False]

        if "normal" in kwargs["planning_modalities"]:
            camera_normals = [True]
            self.keys_to_take["normal"] = f'{kwargs["camera_names"][0]}_normal'
        else:
            camera_normals = [False]

        assert (
            len(kwargs["camera_names"]) == 1
        ), "Currently only one camera is supported!"

        self.og_env = og.Environment(configs=config)
        self.robot = self.og_env.robots[0]
        self.action_primitives = StarterSemanticActionPrimitives(self.og_env, enable_head_tracking=False)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=self.observation_shape
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dimension,))
        
        # remove later
        for i in range(10):
            og.sim.step()

    def reset(self):
        # Reset environment and robot
        self.og_env.reset()
        self.og_env.robots[0].reset()

        # Step simulator a few times so that the effects of "reset" take place
        for _ in range(10):
            og.sim.step()

    def reset_to(self, state):
        og.sim.restore(state)
        # step the simulator a few times 
        for i in range(20):
            print("i: ", i)
            og.sim.step()

    def reset_state(self, state):
        pass

    def get_image_obs(self, obs):
        img_obs = {}
        img_obs['rgb'] = obs['robot0']['robot0:eyes:Camera:0']['rgb'][:,:,:3]
        return img_obs
    
    def get_viewer_obs(self):
        viewer_obs = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3]
        return viewer_obs

    def step(self, action):
        pass

    def get_state(self):
        pass

    def compute_score(self, state, goal_state):
        pass

    @property
    def observation_shape(self):
        return self.env_hparams["camera_height"], self.env_hparams["camera_width"]
    
    @property
    def action_dimension(self):
        return self.robot.action_dim
        # return self.env_hparams["a_dim"]
    
    def arm_action_dimension(self):
        return 6

    def goal_generator(self):
        # goals_dataset_path = to_absolute_path(self.env_hparams["goals_dataset"])
        goals_dataset_path = self.env_hparams["goals_dataset"]
        print(f"==== Loading goals from {goals_dataset_path} ====")
        yield from self.goal_generator_from_og_hdf5(
            goals_dataset_path, self.env_hparams["camera_names"][0]
        )

    def execute_controller(self, ctrl_gen):
        obs = self.og_env.get_obs()[0]
        for action in ctrl_gen:
            if action == 'Done':
                continue
            obs, _, _, _ = self.og_env.step(action)
        return obs
    
    def move_primitive(self, action):
        current_pose = self.robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]

        delta_pos = action[:3]
        delta_orn = action[3:6]
        # print("current_orn, delta_orn: ", current_orn, delta_orn)
        # print('type 1: ', type(R.from_quat(current_orn)))
        # print('type 2: ', type(R.from_rotvec(delta_orn).as_quat()))

        target_pos = current_pos + delta_pos
        print("type(target_pos): ", type(target_pos))
        target_orn = R.from_quat(current_orn) * R.from_quat(R.from_rotvec(delta_orn).as_quat())
        print("target_orn: ", target_orn, target_orn.as_quat())
        target_orn = np.array(target_orn.as_quat())

        target_pose = (target_pos, target_orn)
        print("current_pose: ", current_pose)
        print("target_pose: ", target_pose)
        obs = self.execute_controller(self.action_primitives._move_hand_direct_ik(target_pose,
                                                                             stop_on_contact=False,
                                                                             ignore_failure=True,
                                                                             stop_if_stuck=False))
        return obs