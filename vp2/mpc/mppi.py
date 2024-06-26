import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import os

from vp2.mpc.optimizer import Optimizer
from vp2.mpc.utils import *

from omnigibson.utils.ui_utils import draw_line, clear_debug_drawing

temp_ideal_action = np.array([
    [0.0023, 0.0085, -0.0419, 0.0203, 0.0035, -0.0260, 1.0000],
    [0.0034, 0.0125, -0.0613, 0.0217, 0.0036, -0.0309, 1.0000],
    [0.0033, 0.0127, -0.0619, 0.0211, 0.0037, -0.0289, 1.0000],
    [0.0033, 0.0128, -0.0621, 0.0211, 0.0037, -0.0286, 1.0000],
    [0.0033, 0.0128, -0.0621, 0.0209, 0.0034, -0.0287, 1.0000],
    [0.0033, 0.0128, -0.0621, 0.0202, 0.0030, -0.0294, 1.0000],
    [0.0003, 0.0014, -0.0401, -0.0003, -0.0004, -0.0013, 1.0000],
    [0.0004, 0.0021, -0.0598, -0.0023, -0.0021, -0.0040, 1.0000],
    [0.0004, 0.0022, -0.0601, -0.0024, -0.0024, -0.0039, 1.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000],
    [0.0181, 0.0043, 0.0405, -0.0012, -0.0170, 0.0135, -1.0000],
    [0.0353, 0.0183, 0.0686, 0.0407, -0.0522, 0.0300, -1.0000],
    [0.0430, 0.0319, 0.0831, 0.0761, -0.0626, 0.0381, -1.0000],
    [0.0442, 0.0374, 0.0888, 0.0807, -0.0571, 0.0349, -1.0000],
    [0.0452, 0.0390, 0.0911, 0.0812, -0.0586, 0.0300, -1.0000],
    [0.0455, 0.0394, 0.0924, 0.0829, -0.0582, 0.0297, -1.0000],
    [0.0467, 0.0391, 0.0932, 0.0832, -0.0623, 0.0260, -1.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
])

class MPPIOptimizer(Optimizer):
    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        gamma,
        init_std=0.5,
        log_every=1,
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.gamma = gamma
        self.init_std = np.array(init_std)
        self.log_every = log_every
        self._model_prediction_times = list()

        # # remove later
        # import h5py
        # import cv2
        # f = h5py.File('/home/arpit/test_projects/OmniGibson/small_grasp_dataset_test/dataset.hdf5', "r") 
        # gt_preds = np.array(f['episode_00001']['observations']['rgb'])[:,:,:,:3]
        # self.gt_preds = []
        # for i in range(len(gt_preds)):
        #     img = gt_preds[i].copy()
        #     self.gt_preds.append(cv2.resize(img, (64, 64)))
        #     # import matplotlib.pyplot as plt
        #     # fig, ax = plt.subplots(1,2)
        #     # ax[0].imshow(img)
        #     # ax[1].imshow(self.gt_preds[-1])
        #     # plt.show()
        # # print("self.gt_preds: ", self.gt_preds.shape)
        # # input()

    def update_dist(self, samples, scores):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        print("inside update_dist, samples, scores: ", samples.shape, scores.shape)
        scaled_rews = self.gamma * (scores - np.max(scores))
        # exponentiated scores
        exp_rews = np.exp(scaled_rews)
        mu = np.sum(exp_rews * samples, axis=0) / (np.sum(exp_rews, axis=0) + 1e-10)
        print("update mu shape: ", mu.shape)
        return mu, self.init_std

    def draw_actions(self, actions, sorted_inds, rewards, env):
        base_pos, base_orn = env.robot.get_position_orientation()
        base_orn = np.array(R.from_quat(base_orn).as_matrix())
        T_w_r = np.array([
            [base_orn[0][0], base_orn[0][1], base_orn[0][2], base_pos[0]],
            [base_orn[1][0], base_orn[1][1], base_orn[1][2], base_pos[1]],
            [base_orn[2][0], base_orn[2][1], base_orn[2][2], base_pos[2]],
            [0, 0, 0, 1]
        ])
        current_pose = env.robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        point_1 = np.array([
            [1, 0, 0, current_pos[0]],
            [0, 1, 0, current_pos[1]],
            [0, 0, 1, current_pos[2]],
            [0, 0, 0, 1],
        ])
        point_1 = np.dot(T_w_r, point_1)
        point_1_pos = np.array([point_1[0, 3], point_1[1, 3], point_1[2, 3]])

        def hsl_to_rgb(h, s, l):
            from colorsys import hls_to_rgb
            return tuple(round(i * 255) for i in hls_to_rgb(h / 360.0, l, s))
        
        for i, ind in enumerate(sorted_inds):
            target_pos = current_pos + actions[ind, 1, :3]
            point_2 = np.array([
                [1, 0, 0, target_pos[0]],
                [0, 1, 0, target_pos[1]],
                [0, 0, 1, target_pos[2]],
                [0, 0, 0, 1],
            ])
            point_2 = np.dot(T_w_r, point_2)
            point_2_pos = np.array([point_2[0, 3], point_2[1, 3], point_2[2, 3]])
            # point_2_pos = [-0.5, -0.5, 0.5]

            hue = i * (360 / 200)
            lightness = 1.0 - (i / 200)
            # We use full saturation (1.0) and lightness (0.5)
            rgb = hsl_to_rgb(hue, 1.0, 0.5)
            color = (rgb[0], rgb[1], rgb[2], 1)
            # print("i, reward[ind]: ", i, rewards[ind])
            # Convert RGB to hex
            # color_hex = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
            # colors.append(color_hex)

            # if i % 10 == 0:
            draw_line(point_1_pos, point_2_pos, color=color)
            # env.og.sim.step()
                # input()
    
    def plan(
        self,
        t,
        log_dir,
        obs_history,
        state_history,
        action_history,
        goal,
        init_mean=None,
        env=None,
        folder_name=None
    ):
        assert init_mean is not None or t == 1, "init_mean must be provided for MPPI"
        n_ctxt = self.model.num_context
        print("action_history: ", np.array(action_history).shape)
        print("obs_history: ", obs_history.data_dict['rgb'].shape)
        print("grasped_history: ", obs_history.data_dict['grasped'].shape)
        print("state_history: ", np.array(state_history).shape)

        action_history = action_history[-(n_ctxt - 1) :]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]

        context_actions = np.tile(
            np.array(action_history)[None], (self.num_samples, 1, 1)
        )
        # print("context_actions: ", context_actions)
        print("context_actions: ", context_actions.shape)
        # print("self.horizon: ", self.horizon)
        print("init_mean: ", init_mean)

        if init_mean is not None:
            print("len(init_mean): ", len(init_mean))
            mu = np.zeros((self.horizon, self.a_dim))
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        else:
            mu = np.zeros((self.horizon, self.a_dim))
            # check if this is correct
            mu[:, -1] = 1.0
        std = self.init_std[None].repeat(self.horizon, axis=0)

        new_action_samples = self.sampler.sample_actions(self.num_samples, mu, std)
        new_action_samples = np.clip(new_action_samples, -1, 1)

        # # remove later -----------
        # ideal_action = temp_ideal_action[t-1 : t-1+self.horizon]
        # print("temp_ideal_action: ", ideal_action.shape, ideal_action[0])
        # # print("11new_action_sample[199]: ", new_action_samples[199, :2])
        # new_action_samples[199] = ideal_action
        # # print("22new_action_sample[199]: ", new_action_samples[199, :2])
        # # ------------------------

        print("new_action_samples: ", new_action_samples.shape)
        action_samples = np.concatenate((context_actions, new_action_samples), axis=1)
        
        print("action_samples: ", action_samples.shape)
        # for i in range(10):
        #     for j in range(self.horizon):
        #         print("norms: ", np.linalg.norm(action_samples[i, j, :3]))
        # input()

        print("oobs video: ", np.array(obs_history[self.model.base_prediction_modality]).shape)
        batch = {
            "video": np.tile(
                np.array(obs_history[self.model.base_prediction_modality])[None],
                (self.num_samples, 1, 1, 1, 1),
            ),
            "grasped": np.tile(
                np.array(obs_history['grasped'])[None],
                (self.num_samples, 1, 1)
            ),
            "actions": action_samples,
            "state_obs": state_history,
        }
        print("batch[grasped]: ", batch['grasped'].shape)
        print("batch[video]: ", batch['video'].shape)
        import matplotlib.pyplot as plt
        # plt.imshow(batch['video'][0, 1])
        # plt.show()
        print("batch[actions]: ", batch['actions'].shape)

        pred_start_time = time.time()
        predictions = self.model(batch)
        # print("predictions: ", predictions['rgb'][0,0,:2, :2, :])
        print("predictions: ", predictions['rgb'].shape, type(predictions['rgb']), type(predictions['rgb'][0,0,0,0,0]))
        # print("grasped predictions: ", predictions['grasped'].shape, predictions['grasped'][:3])
        predictions['grasped'] = np.round(predictions['grasped'])
        counter = 0
        for j in range(len(predictions['grasped'])):
            # print("--j", np.squeeze(predictions['grasped'][j]))
            if any(np.squeeze(predictions['grasped'][j])):
                counter += 1
        print("Number of trajectories with grasped predictions: ", counter)
        input()
        prediction_time = time.time() - pred_start_time
        self._model_prediction_times.append(prediction_time)
        print(f"Prediction time {prediction_time}")
        print(
            f"Out of {len(self._model_prediction_times)}, Median prediction time {np.median(self._model_prediction_times)}"
        )

        # --- remove later -----
        # new_pred = np.array(self.gt_preds[t-1 : t+self.horizon])
        # .repeat(self.horizon, axis=0)
        # predictions['rgb'][199] = np.array(self.gt_preds[t-1 : t+self.horizon]) / 255
        # ----------------------

        rewards = self.obj_fn(predictions, goal)
        sorted_prediction_inds = np.argsort(-rewards.flatten())
        best_prediction_inds = sorted_prediction_inds[:10]
        worst_prediction_inds = sorted_prediction_inds[-10:]
        print("best_prediction_inds:", best_prediction_inds)
        print("worst_reward_inds: ", worst_prediction_inds)
        best_rewards = [rewards[i] for i in best_prediction_inds]
        worst_rewards = [rewards[i] for i in worst_prediction_inds]
        vis_preds = list()
        for i in best_prediction_inds:
            if len(predictions["rgb"].shape) == 6:
                vis_preds.append(
                    ObservationList({k: v[0, i] for k, v in predictions.items()})
                )
            else:
                vis_preds.append(
                    ObservationList({k: v[i] for k, v in predictions.items()})
                )
        # best_actions = [action_samples[i] for i in best_predictions[:3]]
        print("best rewards:", best_rewards)
        print("worst rewards: ", worst_rewards)
        
        # plot actions on og interface and save it to disk
        self.draw_actions(action_samples, np.flip(sorted_prediction_inds), rewards, env)
        for _ in range(50):
            env.og.sim.step()
        obs_w_lines = env.og_env.get_obs()[0]
        obs_w_lines = env.get_image_obs(obs_w_lines)['rgb'] / 255
        viewer_obs_w_lines = env.get_viewer_obs() / 255
        goal_img = goal['rgb'][0]
        # print("1----", viewer_obs_w_lines[0,0,:5])
        # print("2----", obs_w_lines[0,0,:5])
        # print("3----", goal_img[0,0,:5])
        concat_img = hori_concatenate_image([viewer_obs_w_lines, obs_w_lines, goal_img])
        # plt.imshow(concat_img)
        # plt.show()
        os.makedirs(f"{folder_name}/traj", exist_ok=True)
        save_np_img(concat_img, f"{folder_name}/traj/{t:02d}")
        env.concat_imgs.append(concat_img)
        clear_debug_drawing()

        # print("rewards for pred[199]: ", rewards[199])
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(4,11)
        # for i in range(11):
        #     ax[0][i].imshow(goal['rgb'][i])
        #     ax[1][i].imshow(predictions['rgb'][-1][i])
        #     ax[2][i].imshow(predictions['rgb'][best_prediction_inds[0]][i])
        #     ax[3][i].imshow(predictions['rgb'][worst_prediction_inds[-1]][i])
        # plt.show()

        # print('best actions:', best_actions)
        # uncomment later
        # if t % self.log_every == 0:
        #     self.log_best_plans(
        #         f"{log_dir}/step_{t}_best_plan", vis_preds, goal, best_rewards
        #     )

        mu, std = self.update_dist(action_samples[:, n_ctxt - 1 :], rewards)
        print(f"mu shape = {mu.shape}: {mu}")

        return mu
        # return action_samples[np.argmax(rewards), n_ctxt:]
