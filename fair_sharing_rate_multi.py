# Repurposed to test out 3 players

import numpy as np
from collections import defaultdict
from math import factorial as fac

from scipy.stats import tvar
import matplotlib.pyplot as plt

import parameters as pr

from utils import powerset

import player_1_multi as player_1
import player_2_multi as player_2
import player_3_multi as player_3

from player_manager import sample_kl_divergences

import os
from os.path import join as oj


from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')




for P1_noise_std_dev in [1, 1.2, 1.4]:
    for P3_noise_std_dev in [1.1, 1.3, 1.5]:
        P2_cov = player_2.cov

        result_dir = oj('multiplayer', 'synthetic', "P1-{}_P2-{}_P3-{}".format(str(P1_noise_std_dev), str(P2_cov), str(P3_noise_std_dev)) )

        player_1.noise_std_dev = P1_noise_std_dev
        player_3.noise_std_dev = P3_noise_std_dev

        os.makedirs(result_dir, exist_ok=True)

        # os.chdir(result_dir)


        with cwd(result_dir):


            with open(oj('settings.txt'), 'w') as f:

                f.write("Experiment Parameters: \n")

                f.write("P1_known_noise_std =  " + str(player_1.noise_std_dev) + '\n')
                f.write("P2_known_noise =  " + str(P2_cov)+ '\n')
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

            N = 3
            P_set = powerset(list(range(N)))

            players = [player_1, player_2, player_3] # create a list of the player py files for calling player-specific custom functions

            player_sample_sizes = [base_sample_size for _ in range(N)] # a list of N numbers

            player_sample_size_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 

            player_FI_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 

            player_shapley_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 


            # prior for unknown noises
            player_3_noise_std_estimate = player_3.noise_std_prior

            print("Testing multiplayer scenario for {} players with {} iterations.".format(N, max_iteration))

            for i in range(max_iteration):
                # Progress
                print("Iteration: {}/{} ".format(i + 1, max_iteration))
                print("Sample size: {}".format(player_sample_sizes))

                # Record current sample sizes
                for player_sample_size_list, player_sample_size in zip(player_sample_size_lists, player_sample_sizes):
                    player_sample_size_list.append(player_sample_size)
                

                # Generate the sample kl divergences

                sample_kl_1, data_x1, data_y1 = player_1.sample_kl_divergences(
                    [player_sample_sizes[0]], 1, posterior_sample_size, prior_mean, prior_cov)
                
                sample_kl_2, data_x2 = player_2.sample_kl_divergences(
                    [player_sample_sizes[1]], 1, posterior_sample_size, prior_mean, prior_cov)
                
                sample_kl_3, data_x3, data_y3 = player_3.sample_kl_divergences(
                    [player_sample_sizes[2]], 1, posterior_sample_size, prior_mean, prior_cov, noise_std_estimate=player_3_noise_std_estimate)
                
                player_3_noise_std_estimate = np.sqrt(tvar(data_y3[0][0]))
                
                
                player_sample_kl_list = [sample_kl_1, sample_kl_2, sample_kl_3] # a list of N numbers for each iteration
                
                
                player_index_data_dict = {0:[data_x1, data_y1], 1:[data_x2],\
                    2:[data_x3, data_y3, player_3_noise_std_estimate]}
                
                sample_kls = defaultdict(float) # a dict for later calculation of SV
                
                # update the dict for individual player sample_kl
                for player_index, player_sample_kl in enumerate(player_sample_kl_list):
                     sample_kls[tuple([player_index])] = np.squeeze(np.asarray(player_sample_kl)) 
                
                for subset in P_set:
                    print(subset)
                    if len(subset) <= 1:continue
                    
                    else:
                        print("Executing the sample kl divergences for :", subset)
                        estimated_kl_values, post_mean = sample_kl_divergences(
                            [sum(player_sample_sizes[index] for index in subset)] , 1, 
                            posterior_sample_size, prior_mean, prior_cov,             
                            {index: player_index_data_dict[index] for index in subset}
                            )
                        print("Estimated kl values are: ", estimated_kl_values)
                        sample_kls[tuple(subset)] = np.squeeze(np.asarray(estimated_kl_values))
                        if len(subset) == N:
                            # Get the current parameter estimate from all the players, used later for calculating FI
                            estimated_param = post_mean

                # Compute Shapley values
                sample_shapleys = [0 for _ in range(N)]

                for index in range(N):
                    sample_shapleys[index] = 0
                    for subset in P_set:

                        if index not in subset:
                            subset_included = tuple(sorted(subset + [index]))

                            C = len(subset) 
                            if C == 0:
                                sample_shapleys[index] +=  (fac(C) * fac(N-C-1) / fac(N)) * \
                                sample_kls[subset_included]
                            else:                    
                                sample_shapleys[index] +=  (fac(C) * fac(N-C-1) / fac(N)) * \
                                (sample_kls[subset_included] - sample_kls[tuple(subset)])

                for player_index, player_shapley_list in enumerate(player_shapley_lists):
                    player_shapley_list.append(sample_shapleys[player_index])
                                

                # Compute Fisher information matrix (determinants) 
                
                player_FI_dets = []
                
                for player_index, player in enumerate(players):
                    
                    player_data = player_index_data_dict[player_index]
                    
                    # calling the custom estimate_FI for each player
                    emp_Fisher = player.estimate_FI(player_data, estimated_param, num_params)
                    
                    player_FI_det = np.linalg.det(emp_Fisher)
                    player_FI_dets.append(player_FI_det) # for comparison among players in this iteration
                    
                    player_FI_lists[player_index].append(player_FI_det) # for record keeping over iterations
                
                # Compute fair sharing rates

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


            for player_index in range(N):

                np.savetxt('cumulative_{}.txt'.format(str(player_index+1)), player_sample_size_lists[player_index])
                
                np.savetxt('shapley_fair_{}.txt'.format(str(player_index+1)), player_shapley_lists[player_index])
                
                np.savetxt('FI_det_{}.txt'.format(str(player_index+1)), player_FI_lists[player_index])




exit()
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