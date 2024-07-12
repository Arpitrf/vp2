# Run control using Visual Foresight.
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import h5py
import datetime
import csv
import os
import copy
import hydra
import matplotlib.pyplot as plt

from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from PIL import Image

from vp2.models.simulator_model import SimulatorModel
from vp2.mpc.utils import *
from vp2.mpc.agent import PlanningAgent

temp_prior = np.array([
    [ 0.012, -0.021, -0.033,  0.024,  0.017, -0.024,  1.   ],
    [ 0.017, -0.031, -0.049,  0.027,  0.017, -0.03,   1.   ],
    [ 0.017, -0.031, -0.049,  0.025,  0.016, -0.028,  1.   ],
    [ 0.017, -0.031, -0.049,  0.024,  0.016, -0.027,  1.   ],
    [ 0.017, -0.031, -0.049,  0.025,  0.019, -0.024,  1.   ],
    [ 0.017, -0.031, -0.049,  0.025,  0.019, -0.025,  1.   ],
    [ 0.017, -0.031, -0.049,  0.025,  0.019, -0.025,  1.   ],
    [ 0.017, -0.031, -0.049,  0.025,  0.019, -0.025,  1.   ],
    [ 0.004, -0.005, -0.045, -0.011,  0.003, -0.028,  1.   ],
    [ 0.008, -0.006, -0.089, -0.037,  0.02,  -0.112,  1.   ],
    [ 0.009, -0.007, -0.125, -0.059,  0.048, -0.153,  1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     -1.   ],
    [-0.017,  0.013,  0.035, -0.001, -0.005,  0.002,  -1.   ],
    [-0.031,  0.031,  0.066,  0.059,  0.023,  0.033,  -1.   ],
    [-0.032,  0.043,  0.081,  0.079,  0.023,  0.008,  -1.   ],
    [-0.032,  0.047,  0.085,  0.079,  0.031, -0.009,  -1.   ],
    [-0.032,  0.049,  0.086,  0.077,  0.032, -0.013,  -1.   ],
    [-0.031,  0.049,  0.086,  0.075,  0.027, -0.011,  -1.   ],
    [-0.03 ,  0.049,  0.086,  0.073,  0.021, -0.005,  -1.   ],
    [-0.029,  0.049,  0.085,  0.072,  0.015,  0.002,  -1.   ],
    [-0.028,  0.049,  0.085,  0.07,   0.011,  0.009,  -1.   ],
    [-0.026,  0.048,  0.084,  0.067,  0.002,  0.02,   -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.   ],
])


def create_env(cfg):
    env = instantiate(cfg)
    ObservationList.IMAGE_SHAPE = env.observation_shape
    return env


def run_trajectory(cfg, folder_name, agent, env, initial_state, goal_state, goal_image):

    # _ = env.reset_to(initial_state)

    # if isinstance(agent, PlanningAgent) and isinstance(
    #     agent.optimizer.model, SimulatorModel
    # ):
    #     agent.optimizer.model.reset_to(initial_state)

    obs, _, _, _ = env.og_env.step(
        np.zeros(env.action_dimension)
    )  # not taking this step delays iGibson observations, TODO debug this!!
    obs = env.get_image_obs(obs)
    grasped_state = env.robot.custom_is_grasping()
    obs['grasped'] = grasped_state
    # plt.imshow(obs['rgb'])
    # plt.show()

    num_steps = 0
    observations = ObservationList.from_obs(obs, cfg)
    # print("observations: ", observations.data_dict['rgb'].shape)
    observations.save_image(f"{folder_name}/obs_after_reset", index=0)
    observations.append(ObservationList.from_obs(obs, cfg))
    # print("observations: ", observations.data_dict['rgb'].shape)
    
    #TODO: implement state_observations
    state_observations = []
    # state_observations = [env.get_state()]
    # print("state_observation: ", state_observations[0].shape)

    observations.save_image(f"{folder_name}/obs_after_step", index=-1)

    agent.reset()
    agent.set_log_dir(folder_name)

    rews = []

    concat_imgs = []
    while num_steps < cfg.max_traj_length:
        print(f"=================== Step {num_steps} ===================")
        # TODO: goal_length should eventually just be cfg.planning_horizon, but the length of predictions from model
        # classes is currently cfg.planning_horizon + cfg.n_context - 1
        goal_length = cfg.n_context + cfg.planning_horizon - 1
        # print("goal_length: ", goal_length)
        # print("11goal_image: ", goal_image.data_dict['rgb'].shape)
        if num_steps < len(goal_image):
            # print("11")
            goal_image = goal_image[num_steps : num_steps + goal_length]
        else:
            # print("22")
            goal_image = goal_image[-1]
        # print("22goal_image: ", goal_image.data_dict['rgb'].shape)
        if len(goal_image) < goal_length:
            goal_image = goal_image.append(
                goal_image[-1].repeat(goal_length - len(goal_image))
            )
        # print("Goal image array shape: ", goal_image.data_dict['rgb'].shape)

        agent.set_goal(goal_image)

        # action = agent.act(num_steps, observations, state_observations, env, folder_name)
        
        # Execute the prior directly
        action = temp_prior[num_steps]
        if num_steps == 0:
            goal_img_temp = goal_image.data_dict['rgb'][0]
        obs = env.og_env.get_obs()[0]
        obs = env.get_image_obs(obs)['rgb'] / 255
        viewer_obs = env.get_viewer_obs() / 255
        concat_img = hori_concatenate_image([viewer_obs, obs, goal_img_temp])
        os.makedirs(f"{folder_name}/traj", exist_ok=True)
        save_np_img(concat_img, f"{folder_name}/traj/{num_steps:02d}")
        env.concat_imgs.append(concat_img)

        # print("action: ", action)
        obs = env.move_primitive(action)
        for _ in range(60):
            env.og.sim.step()
        obs = env.get_image_obs(obs)
        grasped_state = env.robot.custom_is_grasping()
        print("grasped_state after action: ", grasped_state)
        obs['grasped'] = grasped_state

        # obs, _, _, _ = env.step(action)

        observations.append(ObservationList.from_obs(obs, cfg))

        if (
            observations[-1][cfg.planning_modalities[0]].sum() == 0
            or observations[-1][cfg.planning_modalities[0]].min() >= 255
        ):
            # If rendering breaks (black screen), rerun the trajectory
            return None, None, False

        state_observations.append(env.get_state())
        # rews.append(env.get_reward())
        print(f"Step {num_steps}: Action = {action}")
        # input()
        num_steps += 1

    # uncomment later
    # for state_observation in state_observations:
    #     rews.append(env.compute_score(state_observation, goal_state))

    write_moviepy_video(env.concat_imgs, f"{folder_name}/traj.mp4")

    # Currently success is always returned True, even if the task is not solved, so each task is run once
    return observations, rews, True


def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


def get_already_completed_runs(folder_name):
    rews_path = os.path.join(folder_name, "all_rews.txt")
    num_trajectories_completed = 0
    if os.path.exists(rews_path):
        with open(rews_path, "r") as f:
            rews = list(f.read().splitlines())
            num_trajectories_completed = len(rews)
    return num_trajectories_completed


@hydra.main(config_path="configs", config_name="config")
def run_control(cfg):
    set_all_seeds(cfg.seed)
    # with open_dict(cfg):
    #     cfg.slurm_job_id = os.getenv("SLURM_JOB_ID", 0)

    # with open("config.yaml", "w") as f:
    #     OmegaConf.save(config=cfg, f=f.name)

    # The model needs to be instantiated separately from the agent because it could have a complex __init__ function,
    # for example if it performs multiprocessing.
    model = instantiate(
        cfg.model, _recursive_=(not "SimulatorModel" in cfg.model._target_)
    )  # prevent recursive instantiation for simulator model
    agent = instantiate(cfg.agent, optimizer={"model": model})

    env = create_env(cfg.env)
    goal_itr = env.goal_generator()
    
    # remove later
    # for i, retval in enumerate(goal_itr):
    #     if i == 10:
    #         init_state, goal_state, goal_image = retval
    #         break
    # init_state, goal_state, goal_image = next(goal_itr)

    # remove later
    # _ = env.reset_to(init_state)
    # for i in range(1000):
    #     env.og.sim.step()


    folder_name = os.getcwd()
    print(f"Log directory : {os.getcwd()}")

    all_traj_rews = list()

    # # Resume previous run, if applicable
    # if cfg.resume:
    #     num_trajectories_completed = get_already_completed_runs(folder_name)
    # else:
    #     num_trajectories_completed = 0

    # print("num_trajectories_completed: ", num_trajectories_completed)

    for t in range(cfg.num_trajectories):
        print(f"Running trajectory {t}...")
        try:
            init_state, goal_state, goal_image = next(goal_itr)
        except StopIteration:
            print("Ran out of goals, stopping.")
            break

        # if t != 16:
        #     continue

        # print("init_state: ", init_state.keys())
        # print("init_state[states]: ", init_state['states'].shape)
        # print("init_state[model]: ", init_state['model'])
        # print("goal_state: ", goal_state.shape)
        print("goal_image: ", goal_image.data_dict['rgb'].shape)
        # plt.imshow(goal_image.data_dict['rgb'][0])
        # plt.show()

        traj_folder = f"{folder_name}/traj_{t}/"
        print("traj_folder: ", traj_folder)
        # input()

    #     if t < num_trajectories_completed:
    #         # The trajectory has already been completed in a previous run. Skip it, but make sure goal and initial state
    #         # are iterated.
    #         assert os.path.exists(
    #             traj_folder
    #         ), "Trajectory folder does not exist, but control is resuming from a further point"

    #         # Load the rewards from the previous run for bookkeeping
    #         all_traj_rews.append(np.load(f"{traj_folder}/traj_{t}_rews.npy"))
    #         # Skip this trajectory
    #         continue

        os.makedirs(traj_folder, exist_ok=True)
        goal_image.log_gif(f"{traj_folder}/goal_img")
        success = False
        while not success:
            # Try to run control on a starting state and goal repeatedly
            # Not that here success does *not* mean that the task was solved,
            # but that the trajectory was run e.g. without any environment errors.
            obs_out, rews, success = run_trajectory(
                cfg, traj_folder, agent, env, init_state, goal_state, goal_image
            )

    #     obs_out.append(goal_image[-1].repeat(len(obs_out)), axis=1).log_gif(
    #         f"{traj_folder}/traj_{t}_vis"
    #     )
    #     traj_rews = np.array(rews)
    #     all_traj_rews.append(traj_rews)
    #     np.save(f"{traj_folder}/traj_{t}_rews", traj_rews)
    #     with open(f"{folder_name}/all_rews.txt", "a") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([t, traj_rews.min()])

    # # Clean up anything that the model created (like multiprocess spawns)
    # model.close()


if __name__ == "__main__":
    run_control()
