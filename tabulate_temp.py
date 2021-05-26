import os 
from os.path import join as oj
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import ast
# print(plt.style.available)
plt.style.use('seaborn')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# print("Available colors:",  colors)


# name ='CaliH/maxiter-30'
name ='CaliH/'

# P1_DATA_SIZE =  1000
# P1_LOCAL_SAMPLE_SIZE =  100
# P1_LOCAL_SAMPLE =  iid

# P2_DATA_RATIO =  0.01
# P2_NAN_RATIO =  0.05

name = 'KingH'

name ='FaceA'


tables_dir = oj(name, 'table')
os.makedirs(tables_dir, exist_ok=True)


epsilon = 0.1
window_size = 5
burnin = 5

columns = ['P1_data','P1_local','P1_sampling', 'Lowest', 'Average', 'StDev', 'Iter']

data_rows = []
for i, exp in enumerate(os.listdir(name)):

	if exp.startswith('P1-'):

		exp_dir = oj(name, exp)

		with open(oj(exp_dir, 'settings.txt'), 'r') as f:
			lines = [line.strip() for line in f.readlines()]

		for line in lines:
			if 'P1_DATA_SIZE =  ' in line:
				P1_data = ast.literal_eval(line.replace('P1_DATA_SIZE =  ', ''))
			
			if 'P1_LOCAL_SAMPLE_SIZE =  ' in line:
				P1_local = ast.literal_eval(line.replace('P1_LOCAL_SAMPLE_SIZE =  ', ''))

			if 'P1_LOCAL_SAMPLE =  ' in line:
				P1_sampling = line.replace('P1_LOCAL_SAMPLE =  ', '')


		p1_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_1.txt"))
		p2_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_2.txt"))

		relative_p1s = p1_shapley_list / (p1_shapley_list + p2_shapley_list)
		relative_p2s = p2_shapley_list / (p1_shapley_list + p2_shapley_list)
		relative_diffs = np.abs(  relative_p1s - relative_p2s )

		lowest = min(relative_diffs[burnin:]).round(3)
		average = np.mean(relative_diffs[burnin:]).round(3)
		std = np.std(relative_diffs[burnin:]).round(3)

		time_to_epsilon = len(relative_diffs) + 1

		for t in range(burnin, len(relative_diffs) - window_size):
			if (relative_diffs[t: t + window_size] < epsilon).all():
				time_to_epsilon = t
				break

		if time_to_epsilon > len(relative_diffs):
			time_to_epsilon = str(len(relative_diffs)) + '*'

		row = [P1_data, P1_local, P1_sampling, lowest, average, std, time_to_epsilon]

		data_rows.append(row)

df = pd.DataFrame(data=data_rows, columns=columns).sort_values(['P1_data', 'P1_local', 'P1_sampling'])


df.to_csv( oj(tables_dir,'P1-results.csv'), index=False)
df.to_latex( oj(tables_dir,'P1-results.latex'), index=False)


columns = ['P2_data','P2_nan', 'Lowest', 'Average', 'StDev', 'Iter']

data_rows = []
for i, exp in enumerate(os.listdir(name)):
	if exp.startswith('P2-'):

		exp_dir = oj(name, exp)

		with open(oj(exp_dir, 'settings.txt'), 'r') as f:
			lines = [line.strip() for line in f.readlines()]

		for line in lines:
			if 'P2_DATA_RATIO =  ' in line:
				P2_data = ast.literal_eval(line.replace('P2_DATA_RATIO =  ', ''))
			
			if 'P2_NAN_RATIO =  ' in line:
				P2_nan = ast.literal_eval(line.replace('P2_NAN_RATIO =  ', ''))


		p1_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_1.txt"))
		p2_shapley_list = np.loadtxt(oj(exp_dir, "shapley_fair_2.txt"))

		relative_p1s = p1_shapley_list / (p1_shapley_list + p2_shapley_list)
		relative_p2s = p2_shapley_list / (p1_shapley_list + p2_shapley_list)
		relative_diffs = np.abs(  relative_p1s - relative_p2s )

		lowest = min(relative_diffs[burnin:]).round(3)
		average = np.mean(relative_diffs[burnin:]).round(3)
		std = np.std(relative_diffs[burnin:]).round(3)

		time_to_epsilon = len(relative_diffs) + 1

		for t in range(burnin, len(relative_diffs) - window_size):
			if (relative_diffs[t: t + window_size] < epsilon).all():
				time_to_epsilon = t
				break

		if time_to_epsilon > len(relative_diffs):
			time_to_epsilon = str(len(relative_diffs)) + '*'

		row = [P2_data, P2_nan, lowest, average, std, time_to_epsilon]

		data_rows.append(row)

df = pd.DataFrame(data=data_rows, columns=columns).sort_values(['P2_data', 'P2_nan'])

print(df)
df.to_csv( oj(tables_dir,'P2-results.csv'), index=False)
df.to_latex( oj(tables_dir,'P2-results.latex'), index=False)
exit()





