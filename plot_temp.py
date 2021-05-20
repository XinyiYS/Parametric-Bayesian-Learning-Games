import os 
from os.path import join as oj
import matplotlib.pyplot as plt

import numpy as np

exp_dir = 'CaliH/lvg_iid'



p1_sample_size_list = np.loadtxt(oj(exp_dir, "cumulative_1.txt"))
p2_sample_size_list = np.loadtxt(oj(exp_dir, "cumulative_2.txt"))

p1_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_1.txt"))
p2_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_2.txt"))

# Plot sample sizes
plt.plot(p1_sample_size_list, linestyle='--', color='red', label='player 1')
plt.plot(p2_sample_size_list, linestyle='--', color='blue', label='player 2')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Cumulative number of points shared', fontsize=16)

plt.legend()
plt.savefig(oj(exp_dir, 'output_sharing_rate.pdf'))
plt.show()
plt.clf()    

# Plot the shapley value
plt.plot(p1_shapley_list, linestyle='--', color='red', label='player 1')
plt.plot(p2_shapley_list, linestyle='--', color='blue', label='player 2')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Shapley value', fontsize=16)
plt.legend()
plt.savefig(oj(exp_dir, 'output_shapley_fair.pdf'))
plt.show()
plt.clf()    

