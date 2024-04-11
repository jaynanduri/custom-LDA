from typing import List

import numpy as np
import scipy
import matplotlib.pyplot as plt


class GibbsSampling:
    """
    This class represents a gibbs sampling, used to sample from a multidim gaussian generative joint.
    """
    @staticmethod
    def sample(mu: List[float], sigma: List[float], sample_size: int) -> np.ndarray:
        """
        Main method to sample from conditionals. x1 is sampled given prob(X|Y=y0) and y1 sampled given the prob(Y|X=x1).
        :param mu: array of mean values for different dimensions.
        :param sigma: array of standard deviation values for different dimensions
        :param sample_size: number of samples
        :return: array of points sampled from a multi-dim normal distribution
        """
        mu1, mu2 = mu
        cov = np.array([[np.square(sigma[0]), 0], [0, np.square(sigma[1])]])
        sigma11, sigma12 = cov[0]
        sigma21, sigma22 = cov[1]
        x1, x2 = 0, 0  # Initial values
        samples = np.zeros((sample_size, 2))

        for i in range(sample_size):
            # Sample x1 from p(x1|x2)
            mean_x1_given_x2 = mu1 + sigma12 * (x2 - mu2) / sigma22
            var_x1_given_x2 = sigma11 - sigma12 * sigma21 / sigma22
            x1 = scipy.stats.multivariate_normal(mean_x1_given_x2, np.sqrt(var_x1_given_x2)).rvs()

            # Sample x2 from p(x2|x1)
            mean_x2_given_x1 = mu2 + sigma21 * (x1 - mu1) / sigma11
            var_x2_given_x1 = sigma22 - sigma21 * sigma12 / sigma11
            x2 = scipy.stats.multivariate_normal(mean_x2_given_x1, np.sqrt(var_x2_given_x1)).rvs()

            samples[i, :] = [x1, x2]

        return samples


if __name__ == "__main__":

    mu = [10, 10]
    sigma = [1, 1]
    sampler = GibbsSampling()
    x = sampler.sample(mu, sigma, 10000)
    x = np.array(x)
    xs = x[:, 0]
    ys = x[:, 1]
    pdfs = scipy.stats.multivariate_normal(mu, np.array([[sigma[0], 0], [0, sigma[1]]])).pdf(x)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    surf = ax.scatter(xs, ys, pdfs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('PDF')
    plt.show()


