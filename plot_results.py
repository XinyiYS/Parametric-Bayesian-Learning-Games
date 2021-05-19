import matplotlib.pyplot as plt
import numpy as np

import parameters as pr
sample_size_range = pr.sample_size_range

# Load results
limiting_kl_1 = np.loadtxt("p1_kl_limit.txt")
limiting_kl_2 = np.loadtxt("p2_kl_limit.txt")
limiting_kl_12 = np.loadtxt("p12_kl_limit.txt")
sample_kl_1 = np.loadtxt("p1_kl_sample.txt")
sample_kl_2 = np.loadtxt("p2_kl_sample.txt")
sample_kl_12 = np.loadtxt("p12_kl_sample.txt")

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
