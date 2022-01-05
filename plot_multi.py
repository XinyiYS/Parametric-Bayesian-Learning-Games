import numpy as np
import matplotlib.pyplot as plt

# print(plt.style.available)
plt.style.use('seaborn')

LABEL_FONTSIZE = 24
MARKER_SIZE = 10
AXIS_FONTSIZE = 26
TITLE_FONTSIZE= 26
LINEWIDTH = 5

# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title

import os
from os.path import join as oj

result_dir = oj('multiplayer', 'CaliH', 'P1-1000_10')
os.chdir(result_dir)

N = 4
player_sample_size_lists = [np.loadtxt('cumulative_{}.txt'.format(i)) for i in range(1 , 1+N) ]
player_shapley_lists = [np.loadtxt('shapley_fair_{}.txt'.format(i)) for i in range(1 , 1+N) ]
player_FI_lists = [np.loadtxt('FI_det_{}.txt'.format(i)) for i in range(1 , 1+N) ]

plt.figure(figsize=(6, 4))

# Plot sample sizes

for player_index in range(N):
    plt.plot(player_sample_size_lists[player_index], label='P'+str(player_index+1))
    
# plt.plot(p1_sample_size_list, linestyle='-', color='C0',  label='P1')
# plt.plot(p2_sample_size_list, linestyle='--', color='C1', label='P2')
# plt.plot(p3_sample_size_list, linestyle='-.', color='C2', label='P3')
plt.ylabel('Cumulative count')
plt.xlabel('Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('output_sharing_rate.pdf',  bbox_inches='tight')
# plt.show()
plt.clf()    
plt.close()

plt.figure(figsize=(6, 4))

# Plot the shapley value
for player_index in range(N):
    plt.plot(player_shapley_lists[player_index], label='P'+str(player_index+1))

# plt.plot(p1_shapley_list, linestyle='-', color='C0', label='P1')
# plt.plot(p2_shapley_list, linestyle='--', color='C1', label='P2')
# plt.plot(p3_shapley_list, linestyle='-.', color='C2', label='P3')
plt.ylabel('Shapley value')
plt.xlabel('Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('output_shapley_fair.pdf',  bbox_inches='tight')
# plt.show()
plt.clf()    
plt.close()


plt.figure(figsize=(6, 4))

# Plot the FI dets

for player_index in range(N):
    plt.plot(player_FI_lists[player_index], label='P'+str(player_index+1))

# plt.plot(p1_FI_list, linestyle='-', color='C0', label='P1')
# plt.plot(p2_FI_list, linestyle='--', color='C1', label='P2')
# plt.plot(p3_FI_list, linestyle='-.', color='C2', label='P3')
plt.ylabel('FI Determinant')
plt.xlabel('Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('fi_determinant.pdf',  bbox_inches='tight')
# plt.show()
plt.clf()    
plt.close()
