from typing import List, Union, Any

import numpy as np
import scipy
import bisect
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleSampling:
    """
    This class represents a simple Sampling class. This contains 3 methods - sample_uniform, sample normal
    distribution, sample_gaussian_2d and steven_method.
    """

    @staticmethod
    def sample_uniform(minVal: float, maxVal: float, sample_size: int) -> List[float]:
        """
        Simple sampling from uniform distribution. This is straightforward, [min, (max - min) * rand()] do this for
        sample_size times.
        :param minVal: min value of the Uniform dist
        :param maxVal: max value of the Uniform dist
        :param sample_size: number of samples needed from Uniform dist
        :return: sample from a uniform distribution
        """
        return [minVal + (maxVal - minVal) * np.random.rand() for _ in range(sample_size)]

    @staticmethod
    def sample_gaussian(mu: float, sigma: float, sample_size) -> List[float]:
        """
        Sampling from normal distribution using rejection sampling. X bounds are given by (mu - 3*sigma, mu + 3*sigma),
        y is bounded the max of pdf of the dist which occurs at mu.
        :param mu: mean of the distribution
        :param sigma: standard deviation of the distribution
        :param sample_size: number of samples needed from Normal dist
        :return: samples from Normal Distribution
        """
        x_min = mu - (3 * sigma)
        x_max = mu + (3 * sigma)

        scale = x_max - x_min
        y_max = scipy.stats.norm(mu, sigma).pdf(mu)
        samples = []
        while len(samples) < sample_size:
            x = x_min + scale * np.random.rand()
            y = y_max * np.random.rand()
            if y <= scipy.stats.norm(mu, sigma).pdf(x):
                samples.append(x)
        return samples

    @staticmethod
    def sample_gaussian_2d(mu: List[float], sigma: List[float], sample_size: int) -> List[List[Union[float, Any]]]:
        """
        Sampling from a 2d gaussian distribution using rejection sampling. X bounds are given by (mu[0] - 3*sigma[0],
        mu[0] + 3*sigma[0]), y bounds are (mu[1] - 3*sigma[1], mu[1] + 3*sigma[1]) and z is bounded the max of pdf of
        the dist which occurs at mu.
        :param mu: two values mu1 and mu2, means in two directions
        :param sigma: two values sigma1 and sigma2, std in two directions
        :param sample_size: number of samples needed from Normal dist
        :return: samples from 2D Normal Distribution
        """
        # computing covariance
        cov = np.array([[np.square(sigma[0]), 0], [0, np.square(sigma[1])]])
        samples = []
        x_min = mu[0] - (3 * sigma[0])
        x_max = mu[0] + (3 * sigma[0])
        xscale = x_max - x_min

        y_min = mu[1] - (3 * sigma[1])
        y_max = mu[1] + (3 * sigma[1])
        yscale = y_max - y_min

        z_max = scipy.stats.multivariate_normal(mu, cov).pdf(mu)

        while len(samples) < sample_size:
            x = x_min + xscale * np.random.rand()
            y = y_min + yscale * np.random.rand()
            z = z_max * np.random.rand()
            if z <= scipy.stats.multivariate_normal(mu, cov).pdf(np.array([x, y])):
                samples.append([x, y])
        return samples

    @staticmethod
    def stevens_method(sample_size: int, prob_dist: np.ndarray) -> List[float]:
        """
        Sampling from a discrete non-uniform distribution. Elements are selected with probabilities proportional to
        their given probabilities.
        :param sample_size: number of samples
        :param prob_dist: list of probabilities associated with each element in the distribution
        :return: samples from a discrete non-uniform distribution
        """

        # Sort the probabilites in descending order
        idx_sorted = np.argsort(-prob_dist)
        pdist_sorted = prob_dist[idx_sorted]

        # Create buckets of size sample_size
        buckets = []
        for i in range(0, len(prob_dist), sample_size):
            bucket = pdist_sorted[i:min(i + sample_size, len(prob_dist))]
            buckets.append(bucket)

        # Normalize Probability for each bucket
        bucket_pdist = []
        for i in range(len(buckets)):
            bucket_pdist.append(np.sum(buckets[i]))

        # Extract sample_size from bucket_pdist without replacement
        bucket_cdf = np.cumsum(bucket_pdist)
        sampled_buckets = []
        for i in range(sample_size):
            k = np.random.rand()
            sampled_bucket = bisect.bisect_right(bucket_cdf, k)
            sampled_buckets.append(sampled_bucket)

        # Create a dictionary of bucket counts
        bucket_cnts = {}
        for i in sorted(sampled_buckets):
            bucket_cnts[i] = 1 + bucket_cnts.get(i, 0)

        # Pick k random samples from each bucket, with k = count of that bucket
        samples = []
        for counter, i in enumerate(bucket_cnts.keys()):
            idx = []
            while len(idx) < bucket_cnts[i]:
                k = int(round(sample_size * np.random.rand(), 0)) + counter * sample_size
                idx.append(k)
            samples.extend(idx)

        return samples


if __name__ == "__main__":
    sampler = SimpleSampling()
    # Uniform distribution
    x = sampler.sample_uniform(10, 15, 1000000)
    sns.histplot(x, kde=True)
    plt.show()
    # gaussian distribution
    x = sampler.sample_gaussian(10, 1, 10000)
    sns.histplot(x, kde=True)
    plt.show()
    # 2d gaussian distribution
    mu = [10, 10]
    sigma = [1, 1]

    x = sampler.sample_gaussian_2d(mu, sigma, 10000)
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
    # # steven's method
    xs = np.linspace(1, 5, 1000)
    ys = np.exp(-xs)
    ys = ys / np.sum(ys)
    plt.bar(xs, ys)
    plt.show()
    sampled_idx = sampler.stevens_method(50, ys)
    x_samples = xs[np.sort(np.array(sampled_idx))]
    y_samples = ys[np.sort(np.array(sampled_idx))]
    plt.bar(x_samples, y_samples)
    plt.show()



