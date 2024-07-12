import numpy as np


class Sampler:
    def __init__(self):
        pass


class GaussianSampler(Sampler):
    def __init__(self, horizon, a_dim):
        self.horizon = horizon
        self.a_dim = a_dim

    def sample_actions(self, num_samples, mu, std):
        return (
            np.expand_dims(mu, 0)
            + np.random.normal(size=(self.num_samples, self.a_dim)) * std[0]
        )


class CorrelatedNoiseSampler(GaussianSampler):
    def __init__(self, a_dim, beta, horizon):
        # Beta is the correlation coefficient between each timestep.
        super().__init__(horizon, a_dim)
        self.beta = beta

    def sample_actions(self, num_samples, mu, std):
        print("mu: ", mu)
        noise_samples = [np.random.normal(size=(num_samples, self.a_dim)) * std[0]]
        for i in range(1, self.horizon):
            noise_samp = (
                self.beta * noise_samples[-1]
                + (1 - self.beta)
                * np.random.normal(size=(num_samples, self.a_dim))
                * std[i]
            )
            noise_samples.append(noise_samp)
        noise_samples = np.stack(noise_samples, axis=1)
        ret_actions = np.expand_dims(mu, 0) + noise_samples

        # remove later
        counter = 0
        counter2 = 0
        for j in range(len(ret_actions)):
            # print("--j", np.squeeze(predictions['grasped'][j]))
                if ret_actions[j][-1][-1] == -1.0:
                    counter += 1
                elif ret_actions[j][-1][-1] == 1.0:
                    counter2 += 1
        print("In sampler: Number of trajectories with a grasped action and no grasped actions respectively: ", counter, counter2)
        
        # change certain actions to grasp
        num_changes = np.random.randint(30, 50)
        bs_indices = np.random.choice(num_samples, num_changes, replace=False) #num_samples=200
        traj_indices = np.random.randint(0, self.horizon, num_changes) 
        for i in range(len(bs_indices)):
            ret_actions[bs_indices[i], traj_indices[i]] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            ret_actions[bs_indices[i], traj_indices[i]:, -1] = -1.0

        # always have an immediate grasp + stay action
        for i in range(self.horizon):
            # ret_actions[199, i] = np.array([0.01*np.random.normal(), 0.01*np.random.normal(), 0.01*np.random.normal(), 0.0, 0.0, 0.0, -1.0])
            ret_actions[199, i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        return ret_actions
