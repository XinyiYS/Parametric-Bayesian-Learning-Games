import numpy as np

# Number of cores to be used
max_num_cores = 2

# Sample size range & Number of samples to be generated
sample_size_range = range(10, 25, 10)
num_samples = 3

# Samplers parameters
#fisher_sample_size = 20000
#posterior_sample_size = 5000
#tuning_step = 3000
fisher_sample_size = 1000
posterior_sample_size = 1000
tuning_step = 1000

# The true parameters
num_params = 10 #
true_params = [-0.00130591, -0.01399582, -0.00134454,  0.02420948, -0.01538096,  0.00052861, 0.01242821, -0.01452801,  0.02239355,  0.02074722]
best_lambda = 1.0

# The prior distribution
prior_mean = np.zeros(num_params)
prior_cov = np.diag(np.repeat(1, num_params))

# For fair data sharing rate
base_sample_size = 20
base_sample_increment = 10
max_sample_increment = 60
max_iteration = 50
