import os 
from os.path import join as oj
import matplotlib.pyplot as plt

import numpy as np

print(plt.style.available)
plt.style.use('seaborn')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

print("Available colors:",  colors)


linestyles = ['solid', 'dashed', 'dotted', 'dashdot',]
markers = ['+','', '', '+', '', 'v', '','^', '<', '>' , 'x']



# name ='CaliH/'
# exps = ['P2-01_04', 'P2-001_04']

# name = 'KingH'
# exps = ['P2-0.1_0.4', 'P2-0.01_0.4']

# name ='FaceA'
# exps = ['P2-0.5_0.1','P2-0.1_0.1']

name ='MNIST_VAE'
exps = ['P1-size-1000_P2-size-1000_P1-ratio-0.1', 'P1-size-1000_P2-size-1000_P1-ratio-0.5']


figures_dir = oj(name, 'figures')
os.makedirs(figures_dir, exist_ok=True)


with open( oj(figures_dir, 'settings.txt'), 'w') as f:
	f.write('Plotting from the below experiments: \n' )
	for exp in exps:
		f.write( oj(name, exp) + '\n')


plt.figure(figsize=(12, 8))

for i, exp in enumerate(exps):
	exp_dir = oj(name, exp)

	p1_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_1.txt"))
	p2_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_2.txt"))

	label = exps[i].replace('-size', '')

	plt.plot(p1_shapley_list, linestyle=linestyles[i], marker=markers[i], color=colors[2], label=label, linewidth = 4.5)
	plt.plot(p2_shapley_list, linestyle=linestyles[i], marker=markers[i], color=colors[3], label=label, linewidth = 4.5)


plt.ylim(-10, 10)

plt.xlabel('Iterations', fontsize=32)
plt.ylabel('Shapley value', fontsize=32)
plt.legend(fontsize=20, loc='lower right')

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.title("SV vs. Iterations", fontsize=32)
plt.tight_layout()
# plt.show()
plt.savefig(oj(figures_dir, 'SV.png'))
plt.clf()
plt.close()


plt.figure(figsize=(12, 8))

for i, exp in enumerate(exps):
	exp_dir = oj(name, exp)


	p1_sample_size_list = np.loadtxt(oj(exp_dir, "cumulative_1.txt"))
	p2_sample_size_list = np.loadtxt(oj(exp_dir, "cumulative_2.txt"))

	label = exps[i].replace('-size', '')

	plt.plot(p1_sample_size_list, linestyle=linestyles[i], marker=markers[i], color=colors[2], label=label, linewidth = 4.5)
	plt.plot(p2_sample_size_list, linestyle=linestyles[i], marker=markers[i], color=colors[3], label=label, linewidth = 4.5)

plt.xlabel('Iterations', fontsize=32)
plt.ylabel('Cumulative count', fontsize=32)
plt.legend(fontsize=20, loc='lower right')
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.title("Cumulative count vs. Iterations", fontsize=36)
plt.tight_layout()
# plt.show()
plt.savefig(oj(figures_dir, 'count.png'))

plt.clf()    


