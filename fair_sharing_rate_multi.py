
# Repurposed to test out 3 players

import numpy as np
from scipy.stats import tvar
import matplotlib.pyplot as plt

import parameters as pr

import player_1_multi as player_1
import player_2_multi as player_2
import player_3_multi as player_3
import player_12_multi as player_12
import player_13_multi as player_13
import player_23_multi as player_23
import player_123_multi as player_123


import os
from os.path import join as oj

import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')
result_dir = oj('multiplayer', 'synthetic', st)

os.makedirs(result_dir, exist_ok=True)

os.chdir(result_dir)

with open(oj('settings.txt'), 'w') as f:

    f.write("Experiment Parameters: \n")

    f.write("P1_known_noise_std =  " + str(player_1.noise_std_dev) + '\n')
    f.write("P2_known_noise_cov =  " + str(player_2.data_cov)+ '\n')
    f.write("P3_unknown_noise_std =  " + str(player_3.noise_std_dev)+ '\n')
    f.write("P3_noise_std_prior =  " + str(player_3.noise_std_prior)+ '\n')

    f.write("\n")

    f.write("Algorithm Parameters: \n")

    f.write("fisher_sample_size =  " + str(pr.fisher_sample_size)+ '\n')
    f.write("posterior_sample_size =  " + str(pr.posterior_sample_size)+ '\n')
    f.write("tuning_step =  " + str(pr.tuning_step)+ '\n')
    f.write("num_params =  " + str(pr.num_params)+ '\n')
    f.write("true_param =  " + str(pr.true_param)+ '\n')
    # f.write("best_lambda =  " + str(pr.best_lambda)+ '\n')

    f.write("base_sample_size =  " + str(pr.base_sample_size)+ '\n')
    f.write("base_sample_increment =  " + str(pr.base_sample_increment)+ '\n')
    f.write("max_sample_increment =  " + str(pr.max_sample_increment)+ '\n')
    f.write("max_iteration =  " + str(pr.max_iteration)+ '\n')
    
    f.write("sample_size_range =  " + str(pr.sample_size_range)+ '\n')
    f.write("num_samples =  " + str(pr.num_samples)+ '\n')


posterior_sample_size = pr.posterior_sample_size
prior_mean = pr.prior_mean
prior_cov = pr.prior_cov
num_params = pr.num_params

base_sample_size = pr.base_sample_size
base_sample_increment = pr.base_sample_increment
max_sample_increment = pr.max_sample_increment
max_iteration = pr.max_iteration

# p1_sample_size = base_sample_size
# p2_sample_size = base_sample_size
# p3_sample_size = base_sample_size

player_sample_sizes = [base_sample_size, base_sample_size, base_sample_size]

p1_sample_size_list = []
p2_sample_size_list = []
p3_sample_size_list = []

p1_FI_list = []
p2_FI_list = []
p3_FI_list = []

p1_shapley_list = []
p2_shapley_list = []
p3_shapley_list = []

p1_FI_det_list = []
p2_FI_det_list = []
p3_FI_det_list = []

# prior for unknown noises
player_3_noise_std_estimate = player_3.noise_std_prior

print("Testing multiplayer scenario for 3 players with {} iterations.".format(max_iteration))

for i in range(max_iteration):
    # Progress
    print("Iteration: {}/{} ".format(i + 1, max_iteration))
    print("Sample size: {}".format(player_sample_sizes))


    # Record current sample sizes
    p1_sample_size_list.append(player_sample_sizes[0])
    p2_sample_size_list.append(player_sample_sizes[1])
    p3_sample_size_list.append(player_sample_sizes[2])
    


    # Generate the sample kl divergences
    sample_kl_1, data_x1, data_y1 = player_1.sample_kl_divergences(
        [player_sample_sizes[0]], 1, posterior_sample_size, prior_mean, prior_cov)
    
    sample_kl_2, data_x2 = player_2.sample_kl_divergences(
        [player_sample_sizes[1]], 1, posterior_sample_size, prior_mean, prior_cov)
    
    sample_kl_3, data_x3, data_y3 = player_3.sample_kl_divergences(
        [player_sample_sizes[2]], 1, posterior_sample_size, prior_mean, prior_cov, noise_std_estimate=player_3_noise_std_estimate)
    
    print("Iteration: {}, data_y3 shape: {}.".format(i+1, data_y3[0][0].shape))

    sample_kl_12, post_mean = player_12.sample_kl_divergences(
        [player_sample_sizes[0] + player_sample_sizes[1]], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_x1, data_y1, data_x2)
    
    sample_kl_13, post_mean = player_13.sample_kl_divergences(
        [player_sample_sizes[0] + player_sample_sizes[2]], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_x1, data_y1, 
        data_x3, data_y3, noise_std_estimate=player_3_noise_std_estimate)

    sample_kl_23, post_mean = player_23.sample_kl_divergences(
        [player_sample_sizes[1] + player_sample_sizes[2]], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_x2, 
        data_x3, data_y3, noise_std_estimate=player_3_noise_std_estimate)


    sample_kl_123, post_mean = player_123.sample_kl_divergences(
        [player_sample_sizes[0] + player_sample_sizes[1] + player_sample_sizes[2]], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_x1, data_y1, 
        data_x2,
        data_x3, data_y3, noise_std_estimate=player_3_noise_std_estimate)


    # Current Shapley value
    # ordering 123, 132, 213, 312, 321, 213
    sample_shapley_1 = sample_kl_1 + sample_kl_1\
    + np.subtract(sample_kl_12, sample_kl_2) + np.subtract(sample_kl_13, sample_kl_3)\
    + np.subtract(sample_kl_123, sample_kl_23) + np.subtract(sample_kl_123, sample_kl_23)

    sample_shapley_1 /= 6


    sample_shapley_2 = sample_kl_2 + sample_kl_2\
    + np.subtract(sample_kl_12, sample_kl_1) + np.subtract(sample_kl_23, sample_kl_3)\
    + np.subtract(sample_kl_123, sample_kl_13) + np.subtract(sample_kl_123, sample_kl_13)

    sample_shapley_2 /= 6


    sample_shapley_3 = sample_kl_3 + sample_kl_3\
    + np.subtract(sample_kl_13, sample_kl_1) + np.subtract(sample_kl_23, sample_kl_2)\
    + np.subtract(sample_kl_123, sample_kl_12) + np.subtract(sample_kl_123, sample_kl_13)

    sample_shapley_3 /= 6


    p1_shapley_list.append(sample_shapley_1.flatten())
    p2_shapley_list.append(sample_shapley_2.flatten())
    p3_shapley_list.append(sample_shapley_3.flatten())
        
    # Get the current parameter estimate
    estimated_param = post_mean
    
    # Estimate the Fisher informations at the estimated parameter
    # player 1
    emp_Fisher_1 = np.zeros((num_params, num_params))
    for j in range(len(data_x1[0][0])):
        sample_dlogL = player_1.dlogL(data_x1[0][0][j], data_y1[0][0][j], estimated_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_1 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_1 = emp_Fisher_1 / len(data_x1[0][0])
    
    # player 2
    emp_Fisher_2 = np.zeros((num_params, num_params))
    for data_point_x in data_x2[0][0]:
        sample_dlogL = player_2.dlogL(data_point_x, estimated_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_2 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_2 = emp_Fisher_2 / len(data_x2[0][0])
    

    # player 3
    # estimate the sample variance in the y of player 3 and use the sample var as posterior
    # player_3_sample_variance_y = tvar(data_y3[0][0])
    # player_3_noise_std_estimate =  np.sqrt(player_3_sample_variance_y)
    # print("latest p3 noise std estimate: ", player_3_noise_std_estimate)
    
    emp_Fisher_3 = np.zeros((num_params, num_params))
    for j in range(len(data_x3[0][0])):
        sample_dlogL = player_3.dlogL(data_x3[0][0][j], data_y3[0][0][j], estimated_param, player_3_noise_std_estimate)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_3 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_3 = emp_Fisher_3 / len(data_x3[0][0])


    # Store the determinant of estimated Fisher information
    p1_FI_list.append(np.linalg.det(emp_Fisher_1))
    p2_FI_list.append(np.linalg.det(emp_Fisher_2))
    p3_FI_list.append(np.linalg.det(emp_Fisher_3))
    
    # Estimate the fair rate & next sample size
    det_F1 = np.linalg.det(emp_Fisher_1)
    det_F2 = np.linalg.det(emp_Fisher_2)
    det_F3 = np.linalg.det(emp_Fisher_3)

    p1_FI_det_list.append(det_F1)
    p2_FI_det_list.append(det_F2)
    p3_FI_det_list.append(det_F3)


    player_FI_dets = [det_F1, det_F2, det_F3]
    max_FI = max(player_FI_dets)

    max_FI_sample_count = player_sample_sizes[player_FI_dets.index(max_FI)]

    for i, (FI, sample_size) in enumerate(zip(player_FI_dets, player_sample_sizes)):
        
        if FI == max_FI:
            player_sample_sizes[i] += base_sample_increment

        else:
            rate = np.power(max_FI / FI, 1.0 / num_params)
            target = round(max_FI_sample_count * rate)
            if sample_size < target:
                sample_size += min(target - sample_size, max_sample_increment)

            player_sample_sizes[i] = int(sample_size)

        player_sample_sizes[i] = int(player_sample_sizes[i])

    '''

    p1_sample_size, p2_sample_size, p3_sample_size = \
    player_sample_sizes[0], player_sample_sizes[1], player_sample_sizes[2]

    if det_F1 > det_F2:
        print("F1 large" + '*' * 60)

        p1_sample_size += base_sample_increment

        # for player 2
        rate = np.power(det_F1 / det_F2, 1.0/num_params)
        target = round(p1_sample_size * rate)
        if p2_sample_size < target:
            p2_sample_size += min(target - p2_sample_size, max_sample_increment)

        # for player 3
        rate = np.power(det_F1 / det_F3, 1.0/num_params)
        target = round(p1_sample_size * rate)
        if p3_sample_size < target:
            p3_sample_size += min(target - p2_sample_size, max_sample_increment)
            
    else: 
        if det_F2 > det_F1:
            print("F2 large" + '-' * 60)
          
            p2_sample_size += base_sample_increment

            # for player 1
            rate = np.power(det_F2 / det_F1, 1.0/num_params)
            target = round(p2_sample_size * rate)
            if p1_sample_size < target:
                p1_sample_size += min(target - p1_sample_size, max_sample_increment)

            # for player 3
            rate = np.power(det_F2 / det_F3, 1.0/num_params)
            target = round(p2_sample_size * rate)
            if p3_sample_size < target:
                p3_sample_size += min(target - p1_sample_size, max_sample_increment)

        else:
            p1_sample_size += base_sample_increment
            p2_sample_size += base_sample_increment   
            p3_sample_size += base_sample_increment

    player_sample_sizes[0], player_sample_sizes[1], player_sample_sizes[2]=\
    int(p1_sample_size), int(p2_sample_size), int(p3_sample_size)

    '''


np.savetxt("cumulative_1.txt", p1_sample_size_list)
np.savetxt("cumulative_2.txt", p2_sample_size_list)
np.savetxt("cumulative_3.txt", p3_sample_size_list)

np.savetxt("shapley_fair_1.txt", p1_shapley_list)
np.savetxt("shapley_fair_2.txt", p2_shapley_list)
np.savetxt("shapley_fair_3.txt", p3_shapley_list)

np.savetxt("FI_det_1.txt", p1_FI_det_list)
np.savetxt("FI_det_2.txt", p2_FI_det_list)
np.savetxt("FI_det_3.txt", p3_FI_det_list)


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



plt.figure(figsize=(6, 4))

# Plot sample sizes
plt.plot(p1_sample_size_list, linestyle='-', color='C0',  label='P1')
plt.plot(p2_sample_size_list, linestyle='--', color='C1', label='P2')
plt.plot(p3_sample_size_list, linestyle='-.', color='C2', label='P3')
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
plt.plot(p1_shapley_list, linestyle='-', color='C0', label='P1')
plt.plot(p2_shapley_list, linestyle='--', color='C1', label='P2')
plt.plot(p3_shapley_list, linestyle='-.', color='C2', label='P3')
plt.ylabel('Shapley value')
plt.xlabel('Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('output_shapley_fair.pdf',  bbox_inches='tight')
# plt.show()
plt.clf()    
plt.close()


plt.figure(figsize=(6, 4))

# Plot the shapley value
plt.plot(p1_FI_det_list, linestyle='-', color='C0', label='P1')
plt.plot(p2_FI_det_list, linestyle='--', color='C1', label='P2')
plt.plot(p3_FI_det_list, linestyle='-.', color='C2', label='P3')
plt.ylabel('FI Determinant')
plt.xlabel('Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('fi_determinant.pdf',  bbox_inches='tight')
# plt.show()
plt.clf()    
plt.close()