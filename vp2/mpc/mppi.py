import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt

from vp2.mpc.optimizer import Optimizer
from vp2.mpc.utils import *

from omnigibson.utils.ui_utils import draw_line, clear_debug_drawing

# temp_prior = np.array([
#     [-0.014, -0.035, -0.029, -0.019, -0.002, -0.012,  1.   ],
#     [-0.02 , -0.049, -0.042, -0.021, -0.002, -0.013,  1.   ],
#     [-0.021, -0.051, -0.043, -0.021, -0.003, -0.013,  1.   ],
#     [-0.021, -0.051, -0.043, -0.021, -0.003, -0.013,  1.   ],
#     [-0.021, -0.051, -0.043, -0.021, -0.003, -0.013,  1.   ],
#     [-0.004, -0.008, -0.057, -0.001, -0.001, -0.001,  1.   ],
#     [-0.005, -0.011, -0.077, -0.001,  0.001, -0.001,  1.   ],
#     [-0.001, -0.016,  0.043, -0.001,  0.,    -0.004,  1.   ],
#     [-0.001, -0.023,  0.061, -0.003,  0.,    -0.006,  1.   ],
#     [-0.001, -0.023,  0.062, -0.003, -0.,    -0.005,  1.   ],
#     [-0.001, -0.024,  0.063, -0.002, -0.002, -0.006,  1.   ],
#     [-0.001, -0.024,  0.063, -0.001,  0.001, -0.004,  1.   ],
#     [-0.001, -0.024,  0.064, -0.001,  0.001, -0.004,  1.   ],
#     [ 0.   , -0.031,  0.081,  0.003, -0.004, -0.009,  1.   ],
#     [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.   ]
# ])

temp_prior = np.array([
    [ 0.013,  0.02,  -0.035, -0.034, -0.01,  -0.01,   1.   ],
    [ 0.004,  0.03,  -0.053, -0.038, -0.013, -0.011,  1.   ],
    [ 0.004,  0.03,  -0.052, -0.035, -0.012, -0.01,   1.   ],
    [ 0.   ,  0.006, -0.047,  0.005, -0.011, -0.005,  1.   ],
    [-0.001,  0.014, -0.091,  0.007, -0.052,  0.014,  1.   ],
    [ 0.013,  0.012, -0.085,  0.003, -0.042, -0.059,  1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,    -1.   ],
    [-0.026,  0.007,  0.029, -0.004, -0.003, -0.053, -1.   ],
    [-0.042,  0.018,  0.058,  0.025,  0.031, -0.065, -1.   ],
    [-0.047,  0.023,  0.068,  0.022,  0.044, -0.092, -1.   ],
    [-0.049,  0.024,  0.07,   0.02,   0.049, -0.106, -1.   ],
    [-0.049,  0.024,  0.07,   0.019,  0.049, -0.113, -1.   ],
    [-0.049,  0.024,  0.069,  0.018,  0.05,  -0.119, -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.   ]
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
        # print("inside update_dist, samples, scores: ", samples.shape, scores.shape)
        scaled_rews = self.gamma * (scores - np.max(scores))
        # print("scaled_rews: ", scaled_rews)
        # exponentiated scores
        exp_rews = np.exp(scaled_rews)
        # print("exp_rews: ", np.squeeze(exp_rews))
        mu = np.sum(exp_rews * samples, axis=0) / (np.sum(exp_rews, axis=0) + 1e-10)
        # print("update mu shape: ", mu.shape)

        print("action before rounding off grasp action: ", mu[0])
        # trying for grasp action. we don't want to just average it.
        for i in range(self.horizon):
            if mu[i, -1] < 0:
                mu[i, -1] = -1.0
            else:
                mu[i, -1] = 1.0

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
    
    def visualize_action_pred_rew(self, predictions, goal, action_samples, rewards, sorted_prediction_inds):
        goal = np.argmax(goal['gripper_obj_seg'], axis=-1)
        fig, ax = plt.subplots(10,8)
        rewards = np.squeeze(rewards)
        temp_ind = 0
        for i in range(10):
            ind = sorted_prediction_inds[temp_ind]
            print(rewards[ind])
            for j in range(8):
                if j == 7:
                    ax[i][j].imshow(goal[0])
                else:
                    ax[i][j].imshow(predictions['gripper_obj_seg'][ind][j])
            temp_ind += 19
        plt.show()

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
        # print("action_history: ", np.array(action_history).shape)
        # print("obs_history: ", obs_history.data_dict['rgb'].shape)
        # plt.imshow(obs_history.data_dict['rgb'][-1])
        # plt.show()
        # print("grasped_history: ", obs_history.data_dict['grasped'])
        # print("state_history: ", np.array(state_history).shape)
        # print("n_ctxt: ", n_ctxt)

        action_history = action_history[-(n_ctxt - 1) :]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]

        context_actions = np.tile(
            np.array(action_history)[None], (self.num_samples, 1, 1)
        )
        # print("context_actions: ", context_actions)
        # print("context_actions: ", context_actions.shape)
        # print("self.horizon: ", self.horizon)
        # print("init_mean: ", init_mean)

        
        # # Not using a prior
        # if init_mean is not None:
        #     mu = np.zeros((self.horizon, self.a_dim))
        #     mu[: len(init_mean)] = init_mean
        #     mu[len(init_mean) :] = init_mean[-1]
        #     mu[:, -1] = 1.0
        # else:
        #     mu = np.zeros((self.horizon, self.a_dim))
        #     # check if this is correct
        #     mu[:, -1] = 1.0

        # Using a prior
        mu = temp_prior[t-1 : t-1+self.horizon]
        
        std = self.init_std[None].repeat(self.horizon, axis=0)
        new_action_samples = self.sampler.sample_actions(self.num_samples, mu, std)
        new_action_samples = np.clip(new_action_samples, -1, 1)

        # # remove later -----------
        # ideal_action = temp_prior[t-1 : t-1+self.horizon]
        # print("temp_ideal_action: ", ideal_action.shape, ideal_action[0])
        # # print("11new_action_sample[199]: ", new_action_samples[199, :2])
        # # new_action_samples[199] = ideal_action
        # new_action_samples[-1] = ideal_action
        # # print("22new_action_sample[199]: ", new_action_samples[199, :2])
        # # ------------------------

        # print("new_action_samples: ", new_action_samples.shape)
        # added by Arpit
        action_samples = new_action_samples
        if n_ctxt > 1:
            action_samples = np.concatenate((context_actions, new_action_samples), axis=1)

        # # remove later
        # action_samples[:, :, 3:6] = [0.0, 0.0, 0.0]
        
        # print("action_samples: ", action_samples[:5], action_samples[199])
        print("action_samples: ", action_samples.shape)

        counter = 0
        counter2 = 0
        for j in range(len(action_samples)):
            # print("--j", np.squeeze(predictions['grasped'][j]))
                if action_samples[j][-1][-1] == -1.0:
                    counter += 1
                elif action_samples[j][-1][-1] == 1.0:
                    counter2 += 1
        print("Number of trajectories with a grasped action and no grasped actions respectively: ", counter, counter2)
        # input()


        # for i in range(10):
        #     for j in range(self.horizon):
        #         print("norms: ", np.linalg.norm(action_samples[i, j, :3]))
        # input()

        # print("obs video: ", np.array(obs_history[self.model.base_prediction_modality]).shape)
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
        print("batch[grasped], batch[video], batch[actions]: ", batch['grasped'].shape, batch['video'].shape, batch['actions'].shape)
        # obs_input_to_model = np.argmax(batch['video'], axis=-1)
        # plt.imshow(obs_input_to_model[0, 0])
        # plt.show()

        pred_start_time = time.time()
        with torch.no_grad():
            predictions = self.model(batch)
        # print("predictions: ", predictions['rgb'][0,0,:2, :2, :])
        # print("predictions: ", predictions['rgb'].shape, type(predictions['rgb']), type(predictions['rgb'][0,0,0,0,0]))
        # print("grasped predictions: ", predictions['grasped'], predictions['grasped'].shape)
        
        # post-process predictions to convert it into right format. 
        # i.e. round the grasped values to 0 or 1 and the one-hot segmentation classes to 0 and 1s as well
        predictions['grasped'] = np.round(predictions['grasped'])
        logSoftmax = torch.nn.LogSoftmax(dim=-1)
        pred_torch = torch.from_numpy(predictions['gripper_obj_seg'])
        out = logSoftmax(pred_torch).numpy()
        predictions['gripper_obj_seg'] = np.argmax(out, axis=-1)
        
        # Testing how many episodes out of 200 have a grasp state = 1 predicted
        counter = 0
        for j in range(len(predictions['grasped'])):
            # print("--j", np.squeeze(predictions['grasped'][j]))
            if any(np.squeeze(predictions['grasped'][j])):
                counter += 1
        print("Number of trajectories with grasped predictions: ", counter)
        # input()
        
        prediction_time = time.time() - pred_start_time
        self._model_prediction_times.append(prediction_time)
        # print(f"Prediction time {prediction_time}")
        # print(
        #     f"Out of {len(self._model_prediction_times)}, Median prediction time {np.median(self._model_prediction_times)}"
        # )

        # # print("action[199], prediction_grasped[199]: ", action_samples[199], predictions['grasped'][199])
        # # print("action[185]: ", action_samples[185])
        # # print("action[187]: ", action_samples[187])
        # # input()

        # # --- remove later -----
        # # new_pred = np.array(self.gt_preds[t-1 : t+self.horizon])
        # # .repeat(self.horizon, axis=0)
        # # predictions['rgb'][199] = np.array(self.gt_preds[t-1 : t+self.horizon]) / 255
        # # ----------------------

        rewards = self.obj_fn(predictions, goal)
        sorted_prediction_inds = np.argsort(-rewards.flatten())
        best_prediction_inds = sorted_prediction_inds[:10]
        worst_prediction_inds = sorted_prediction_inds[-10:]
        print("best_prediction_inds:", best_prediction_inds)
        print("worst_reward_inds: ", worst_prediction_inds)
        best_rewards = [rewards[i] for i in best_prediction_inds]
        worst_rewards = [rewards[i] for i in worst_prediction_inds]
        vis_preds = list()
        for i in best_prediction_inds[:5]:
            # vis_preds.append(
            #     ObservationList({k: v[i] for k, v in predictions.items()})
            # )
            for k, v in predictions.items():
                if k == 'rgb':
                    curr_obs_frame = np.expand_dims(obs_history.data_dict['rgb'][-1], axis=0)
                    val = np.concatenate((curr_obs_frame, v[i]), axis=0)
                    vis_preds.append(ObservationList({k: val}))
                    # vis_preds.append(ObservationList({k: v[i]}))
            # )
        for i in worst_prediction_inds[-5:]:
            for k, v in predictions.items():
                if k == 'rgb':
                    curr_obs_frame = np.expand_dims(obs_history.data_dict['rgb'][-1], axis=0)
                    val = np.concatenate((curr_obs_frame, v[i]), axis=0)
                    vis_preds.append(ObservationList({k: val}))
                    # vis_preds.append(ObservationList({k: v[i]}))

        
        best_actions = [action_samples[i] for i in sorted_prediction_inds[:3]]
        print("best rewards:", np.squeeze(np.array(best_rewards)))
        print("worst rewards: ", np.squeeze(np.array(worst_rewards)))
        # breakpoint()

        # visualize action, prediction and reward
        self.visualize_action_pred_rew(predictions, goal, action_samples, rewards, sorted_prediction_inds)
        
        # remove later
        for e, ind in enumerate(sorted_prediction_inds):
            if e > 4:
                break
            print(np.array(rewards)[ind], np.array(action_samples)[ind], np.squeeze(predictions['grasped'][ind]))
            print("---------------")

        print("Checking rewards of first step grasps.")
        actions_with_first_step_grasp = 0
        for i in range(len(action_samples)):
            if action_samples[i][0][-1] == -1.0:
                actions_with_first_step_grasp += 1
                # print(np.squeeze(np.array(rewards[i])), action_samples[i], np.squeeze(predictions['grasped'][i]))
        print("actions_with_first_step_grasp: ", actions_with_first_step_grasp)
        
        # plot actions on og interface and save it to disk
        self.draw_actions(action_samples, np.flip(sorted_prediction_inds), rewards, env)
        for _ in range(50):
            env.og.sim.step()
        obs_w_lines = env.og_env.get_obs()[0]
        obs_w_lines = env.get_rgb_obs(obs_w_lines)['rgb'] / 255
        viewer_obs_w_lines = env.get_viewer_obs() / 255
        goal_img = goal['gripper_obj_seg'][0]
        print("shapes: ", obs_w_lines.shape, viewer_obs_w_lines.shape, goal_img.shape)
        # print("1----", viewer_obs_w_lines[0,0,:5])
        # print("2----", obs_w_lines[0,0,:5])
        # print("3----", goal_img[0,0,:5])
        concat_img = hori_concatenate_image([viewer_obs_w_lines, obs_w_lines])
        # plt.imshow(concat_img)
        # plt.show()
        os.makedirs(f"{folder_name}/traj", exist_ok=True)
        save_np_img(concat_img, f"{folder_name}/traj/{t:02d}")
        env.concat_imgs.append(concat_img)
        clear_debug_drawing()

        for _ in range(50):
            env.og.sim.step()

        # temp_obs = env.og_env.get_obs()[0]
        # temp_obs = env.get_image_obs(temp_obs)['rgb']
        # print("temp_obs: ", type(temp_obs))
        # plt.imshow(temp_obs)
        # plt.show()


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
        print(f"mu: {mu}")

        return mu, action_samples[sorted_prediction_inds[0], n_ctxt - 1:]
        # return action_samples[np.argmax(rewards), n_ctxt:]
