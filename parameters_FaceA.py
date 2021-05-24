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
# num_params = 4 
# true_param = [-1, 0, 2, 3]

num_params = 9 #

true_param = [-0.05524394,  -0.71245971, 0.01543807,  -0.04481575,  0.8788603, -0.03137792 , 0.00271527 , -0.02351783, -0.01747129  ]
best_lambda = 1.0 

# The prior distribution
prior_mean = np.zeros(num_params)
prior_cov = np.diag(np.repeat(1, num_params))

# For fair data sharing rate
base_sample_size = 20
base_sample_increment = 10
max_sample_increment = 60
max_iteration = 50


