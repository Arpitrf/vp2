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
        print("mu std: ", mu.shape, std.shape)
        noise_samples = [np.random.normal(size=(num_samples, self.a_dim)) * std[0]]
        print("11noise_samples: ", np.array(noise_samples).shape)
        for i in range(1, self.horizon):
            noise_samp = (
                self.beta * noise_samples[-1]
                + (1 - self.beta)
                * np.random.normal(size=(num_samples, self.a_dim))
                * std[i]
            )
            noise_samples.append(noise_samp)
        print("22noise_samples: ", np.array(noise_samples).shape)
        print("---", np.array(noise_samples)[0,0], np.array(noise_samples)[1,0], np.array(noise_samples)[2,0])
        noise_samples = np.stack(noise_samples, axis=1)
        print("33noise_samples: ", np.array(noise_samples).shape)
        ret_actions = np.expand_dims(mu, 0) + noise_samples
        # change certain actions to grasp
        num_changes = np.random.randint(5, 30)
        bs_indices = np.random.randint(0, 200, num_changes)
        traj_indices = np.random.randint(0, 10, num_changes)
        for i in range(len(bs_indices)):
            ret_actions[bs_indices[i], traj_indices[i]] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            ret_actions[bs_indices[i], traj_indices[i]:, -1] = -1.0
            # print(bs_indices[i], traj_indices[i], ret_actions[bs_indices[i]])
            # print("**", ret_actions[bs_indices[i], traj_indices[i]:, -1])
            # print("------------------")
        # print("ret_actions.shape: ", ret_actions.shape)
        return ret_actions
