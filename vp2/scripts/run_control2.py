# Run control using Visual Foresight.
import numpy as np
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


def create_env(cfg):
    env = instantiate(cfg)
    ObservationList.IMAGE_SHAPE = env.observation_shape
    return env


def run_trajectory(cfg, folder_name, agent, env, initial_state, goal_state, goal_image):

    # _ = env.reset_to(initial_state)

    if isinstance(agent, PlanningAgent) and isinstance(
        agent.optimizer.model, SimulatorModel
    ):
        agent.optimizer.model.reset_to(initial_state)

    obs, _, _, _ = env.og_env.step(
        np.zeros(env.action_dimension)
    )  # not taking this step delays iGibson observations, TODO debug this!!
    obs = env.get_image_obs(obs)

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
        print("33goal_image: ", goal_image.data_dict['rgb'].shape)

        agent.set_goal(goal_image)

        action = agent.act(num_steps, observations, state_observations, env, folder_name)

        # # remove later
        # temp_ideal_action = np.array([
        #     [0.0023, 0.0085, -0.0419, 0.0203, 0.0035, -0.0260, 1.0000],
        #     [0.0034, 0.0125, -0.0613, 0.0217, 0.0036, -0.0309, 1.0000],
        #     [0.0033, 0.0127, -0.0619, 0.0211, 0.0037, -0.0289, 1.0000],
        #     [0.0033, 0.0128, -0.0621, 0.0211, 0.0037, -0.0286, 1.0000],
        #     [0.0033, 0.0128, -0.0621, 0.0209, 0.0034, -0.0287, 1.0000],
        #     [0.0033, 0.0128, -0.0621, 0.0202, 0.0030, -0.0294, 1.0000],
        #     [0.0003, 0.0014, -0.0401, -0.0003, -0.0004, -0.0013, 1.0000],
        #     [0.0004, 0.0021, -0.0598, -0.0023, -0.0021, -0.0040, 1.0000],
        #     [0.0004, 0.0022, -0.0601, -0.0024, -0.0024, -0.0039, 1.0000],
        #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000],
        #     [0.0181, 0.0043, 0.0405, -0.0012, -0.0170, 0.0135, -1.0000],
        #     [0.0353, 0.0183, 0.0686, 0.0407, -0.0522, 0.0300, -1.0000],
        #     [0.0430, 0.0319, 0.0831, 0.0761, -0.0626, 0.0381, -1.0000],
        #     [0.0442, 0.0374, 0.0888, 0.0807, -0.0571, 0.0349, -1.0000],
        #     [0.0452, 0.0390, 0.0911, 0.0812, -0.0586, 0.0300, -1.0000],
        #     [0.0455, 0.0394, 0.0924, 0.0829, -0.0582, 0.0297, -1.0000],
        #     [0.0467, 0.0391, 0.0932, 0.0832, -0.0623, 0.0260, -1.0000],
        #     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        # ])
        # action = temp_ideal_action[num_steps]

        # print("action: ", action)
        obs = env.move_primitive(action)
        obs = env.get_image_obs(obs)

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
    # torch.backends.cudnn.deterministic = True


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
    for i, retval in enumerate(goal_itr):
        if i == 8:
            init_state, goal_state, goal_image = retval
            break
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

        # print("init_state: ", init_state.keys())
        # print("init_state[states]: ", init_state['states'].shape)
        # print("init_state[model]: ", init_state['model'])
        # print("goal_state: ", goal_state.shape)
        # print("goal_image: ", goal_image.data_dict['rgb'].shape)
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
