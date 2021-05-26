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
tuning_step = 500


num_classes = 10
latent_dim = 2
num_params = num_classes * latent_dim #
true_param = None


# The prior distribution
prior_mean = np.zeros(latent_dim)
prior_cov = np.diag(np.repeat(1, latent_dim))

# For fair data sharing rate
base_sample_size = 5
base_sample_increment = 1
max_sample_increment = 60
max_iteration = 50


