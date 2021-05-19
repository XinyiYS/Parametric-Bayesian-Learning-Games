import theano
import parameters as pr


'''
"""Limit the cpu cores usage"""
# Set the max number of cpu cores used by numpy BLAS
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
mkl_set_num_threads(pr.max_num_cores)

# Set the max number of cpu cores used by openmp
theano.config.openmp = False
import os
os.environ["OMP_NUM_THREADS"] = str(pr.max_num_cores)
'''

import numpy as np
import matplotlib.pyplot as plt

# Copy info from globals
sample_size_range = pr.sample_size_range
num_samples = pr.num_samples
fisher_sample_size = pr.fisher_sample_size
posterior_sample_size = pr.posterior_sample_size

num_params = pr.num_params
true_param = pr.true_param

import player_1 as player_1
import player_2 as player_2
import player_12 as player_12

prior_mean = pr.prior_mean
prior_cov = pr.prior_cov

# Compute the limiting KL divergences and Shapley value
# The constant term (same for all players and coalitions) for the limiting divergence
diff = np.subtract(prior_mean, true_param)
diff.shape = (num_params, 1)
const_term = np.matmul(np.transpose(diff), np.linalg.inv(prior_cov))
const_term = np.matmul(const_term, diff)
const_term -= num_params
const_term = np.repeat(const_term, len(sample_size_range))
log_term = np.multiply(num_params, np.log(sample_size_range))
const_term = (const_term + log_term) / 2

# Get the limiting divergence for player 1
FI_1 = player_1.estimate_fisher_information(fisher_sample_size)
limiting_kl_1 = np.log(np.linalg.det(np.matmul(prior_cov, FI_1))) / 2
limiting_kl_1 = const_term + np.repeat(limiting_kl_1, len(sample_size_range))

# Get the limiting divergence for player 2
FI_2 = player_2.estimate_fisher_information(fisher_sample_size)
limiting_kl_2 = np.log(np.linalg.det(np.matmul(prior_cov, FI_2))) / 2
limiting_kl_2 = const_term + np.repeat(limiting_kl_2, len(sample_size_range))

# Get the limiting divergence for player 1+2
FI_12 = player_12.estimate_fisher_information(fisher_sample_size)
limiting_kl_12 = np.log(np.linalg.det(np.matmul(prior_cov, FI_12))) / 2
limiting_kl_12 = const_term + np.repeat(limiting_kl_12, len(sample_size_range))

# Sample KL divergences from player 1
sample_kl_1, data_x1, data_y1 = player_1.sample_kl_divergences(sample_size_range, num_samples, 
                                    posterior_sample_size, prior_mean, prior_cov)
# Sample KL divergences from player 2
sample_kl_2, data_x2 = player_2.sample_kl_divergences(sample_size_range, num_samples, 
                                    posterior_sample_size, prior_mean, prior_cov)
# Sample KL divergences from players 1 and 2
sample_kl_12, post_mean = player_12.sample_kl_divergences(sample_size_range, num_samples, 
                                               posterior_sample_size, 
                                               prior_mean, prior_cov, 
                                               data_x1, data_y1, 
                                               data_x2)

# Write the results to file
np.savetxt("p1_kl_limit.txt", limiting_kl_1)
np.savetxt("p2_kl_limit.txt", limiting_kl_2)
np.savetxt("p12_kl_limit.txt", limiting_kl_12)
np.savetxt("p1_kl_sample.txt", sample_kl_1)
np.savetxt("p2_kl_sample.txt", sample_kl_2)
np.savetxt("p12_kl_sample.txt", sample_kl_12)

"""
# Plot the KL samples against the limiting KL
plt.plot(sample_size_range, limiting_kl_1, linestyle='--', color='red')
plt.plot(sample_size_range, limiting_kl_2, linestyle='--', color='blue')
plt.plot(sample_size_range, limiting_kl_12, linestyle='--', color='black')
plt.plot(sample_size_range, sample_kl_1, linestyle='none',
     marker='o', color='red', fillstyle='none')
plt.plot(sample_size_range, sample_kl_2, linestyle='none',
     marker='o', color='blue', fillstyle='none')
plt.plot(sample_size_range, sample_kl_12, linestyle='none',
     marker='o', color='black', fillstyle='none')
plt.savefig('output_kl.pdf')
plt.show()
plt.clf()

# Compute the limiting Shapley value
shapley_1 = (0.5 * limiting_kl_1) + (0.5 * (limiting_kl_12 - limiting_kl_2))
shapley_2 = (0.5 * limiting_kl_2) + (0.5 * (limiting_kl_12 - limiting_kl_1))
 
# Compute the sampled Shapley values
sample_shapley_1 = np.multiply(0.5, sample_kl_1) + np.multiply(0.5, (np.subtract(sample_kl_12, sample_kl_2)))
sample_shapley_2 = np.multiply(0.5, sample_kl_2) + np.multiply(0.5,(np.subtract(sample_kl_12, sample_kl_1)))

# Plot the sample shapley values against the limiting values
plt.plot(sample_size_range, shapley_1, linestyle='--', color='red')
plt.plot(sample_size_range, shapley_2, linestyle='--', color='blue')
plt.plot(sample_size_range, sample_shapley_1, marker='o', linestyle='none', color='red', fillstyle='none')
plt.plot(sample_size_range, sample_shapley_2, marker='o', linestyle='none', color='blue', fillstyle='none')
plt.savefig('output_shapley.pdf')
plt.show()
plt.clf()

# Compute the difference in the sampled Shapley value
sample_shapley_diff = np.subtract(sample_shapley_1, sample_shapley_2)

# Plot the difference
plt.plot(sample_size_range, shapley_1 - shapley_2, linestyle='--', color='black')
plt.plot(sample_size_range, sample_shapley_diff, marker='o', fillstyle='none', linestyle='none', color='black')
plt.savefig('output_shapley_diff.pdf')
plt.show()
plt.clf()

"""