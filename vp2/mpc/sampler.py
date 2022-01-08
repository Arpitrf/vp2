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
        return np.expand_dims(mu, 0) + noise_samples
