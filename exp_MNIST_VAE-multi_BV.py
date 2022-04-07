import os 
import sys
from os.path import join as oj

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

mus = pd.read_csv('.data/MNIST_vae2d/vae2d_MNIST_mus-train.csv').values
logvars = pd.read_csv('.data/MNIST_vae2d/vae2d_MNIST_logvars-train.csv').values
y = pd.read_csv('.data/MNIST_vae2d/vae2d_MNIST_labels-train.csv').values.squeeze()


indices_by_digit = []
for i in range(10):
    indices_by_digit.append( [i for i, include in enumerate(y==i) if include]  )
    
import random
import numpy as np
from numpy.random import choice

random.seed(7913)
np.random.seed(7913)

def sample1(data_size):

    # digit_counts = np.ceil(data_size * np.asarray(P1_PROBS)).astype(int)

    # fix digit counts due to lack of PyMC3 support
    
    digit_counts = np.random.multinomial(data_size, P1_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P1_DATA_SIZE]  , count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    '''
    for c, count in enumerate(digit_counts):
        if count == 0:
            sampled_vae_mus[c] = np.zeros( (1, latent_dim) )
            sampled_vae_logvars[c] = np.zeros( (1, latent_dim) ) 

    '''
    return sampled_vae_mus, sampled_vae_logvars


def sample2(data_size):
    # digit_counts = np.ceil(data_size * np.asarray(P2_PROBS)).astype(int)

    digit_counts = np.random.multinomial(data_size, P2_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P2_DATA_SIZE], count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    return sampled_vae_mus, sampled_vae_logvars


def sample3(data_size):

    digit_counts = np.random.multinomial(data_size, P3_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P3_DATA_SIZE], count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    return sampled_vae_mus, sampled_vae_logvars


def sample4(data_size):
    # digit_counts = np.ceil(data_size * np.asarray(P2_PROBS)).astype(int)

    digit_counts = np.random.multinomial(data_size, P4_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P4_DATA_SIZE], count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    return sampled_vae_mus, sampled_vae_logvars


import parameters_MNIST as pr
import player_1_gmm as player_1
import player_2_gmm as player_2
import player_3_gmm as player_3
import player_4_gmm as player_4
# import player_12_gmm as player_12

import player_manager_gmm_multi
from player_manager_gmm_multi import sample_kl_divergences
from utils import powerset


from collections import defaultdict



from collections import defaultdict
from math import factorial as fac
import numpy as np
from scipy.stats import tvar

import matplotlib.pyplot as plt

import theano
import theano.tensor as T



P2_BALANCE = 0.1
P2_DATA_SIZE = 1000

P3_DATA_SIZE = 2000
P4_DATA_SIZE = 5000
P3_BALANCE = P4_BALANCE = 0.5

for P1_DATA_SIZE, P2_DATA_SIZE in [(1000, 5000), (5000, 5000)]:   
# for P1_DATA_SIZE in [1000, 5000]:
    # for P2_DATA_SIZE in [1000, 5000]:
    for P1_BALANCE in np.linspace(0.1, 0.9, 9):

        P1_PROBS = [1 - P1_BALANCE, P1_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]
        P2_PROBS = [1 - P2_BALANCE, P2_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]

        P3_PROBS = [1 - P3_BALANCE, P3_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]

        P4_PROBS = [1 - P4_BALANCE, P4_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]

        exp_dir = oj('multiplayer-BV', 'MNIST_VAE', "P1-size-{}_P2-size-{}_P1-ratio-{}".format(str(P1_DATA_SIZE), str(P2_DATA_SIZE), str(P1_BALANCE)) )

        os.makedirs(exp_dir, exist_ok=True)

        log_file = open(oj(exp_dir, 'log')  ,"w")
        sys.stdout = log_file

        with open(oj(exp_dir, 'settings.txt'), 'w') as f:

            f.write("Experiment Parameters: \n")

            f.write("P1_DATA_SIZE =  " + str(P1_DATA_SIZE) + '\n')
            f.write("P2_DATA_SIZE =  " + str(P2_DATA_SIZE)+ '\n')
            f.write("P3_DATA_SIZE =  " + str(P3_DATA_SIZE)+ '\n')
            f.write("P4_DATA_SIZE =  " + str(P4_DATA_SIZE)+ '\n')

            f.write("P1_BALANCE =  " + str(P1_BALANCE)+ '\n')
            f.write("P2_BALANCE =  " + str(P2_BALANCE)+ '\n')
            f.write("P3_BALANCE =  " + str(P3_BALANCE)+ '\n')
            f.write("P4_BALANCE =  " + str(P4_BALANCE)+ '\n')

            f.write("Algorithm Parameters: \n")

            f.write("fisher_sample_size =  " + str(pr.fisher_sample_size)+ '\n')
            f.write("posterior_sample_size =  " + str(pr.posterior_sample_size)+ '\n')
            f.write("tuning_step =  " + str(pr.tuning_step)+ '\n')
            f.write("num_params =  " + str(pr.num_params)+ '\n')

            f.write("base_sample_size =  " + str(pr.base_sample_size)+ '\n')
            f.write("base_sample_increment =  " + str(pr.base_sample_increment)+ '\n')
            f.write("max_sample_increment =  " + str(pr.max_sample_increment)+ '\n')
            f.write("max_iteration =  " + str(pr.max_iteration)+ '\n')
            
            f.write("sample_size_range =  " + str(pr.sample_size_range)+ '\n')
            f.write("num_samples =  " + str(pr.num_samples)+ '\n')


        latent_dim = 2

        posterior_sample_size = pr.posterior_sample_size
        prior_mean = pr.prior_mean
        prior_cov = pr.prior_cov
        num_params = pr.num_params

        base_sample_size = pr.base_sample_size
        base_sample_increment = pr.base_sample_increment
        max_sample_increment = pr.max_sample_increment
        max_iteration = pr.max_iteration


        num_classes = ((np.asarray(P1_PROBS) + np.asarray(P1_PROBS)) > 0).sum()
        print("Total number of non-zero probs classes:", num_classes)
        pr.num_classes = num_classes


        sampled_vae_mus_1, sampled_vae_logvars_1 = sample1(5)
        sampled_vae_mus_2, sampled_vae_logvars_2 = sample2(5)
        sampled_vae_mus_3, sampled_vae_logvars_3 = sample3(5)
        sampled_vae_mus_4, sampled_vae_logvars_4 = sample4(5)


        pr.latent_dim = latent_dim
        true_means = np.zeros((num_classes, latent_dim)) # 10 dimensional mean for each class

        for player in [player_1, player_2, player_3, player_4, player_manager_gmm_multi]:

            player.num_classes = num_classes
            player.num_params = num_classes * latent_dim
            player.latent_dim = latent_dim
        


        true_logvars = np.zeros((num_classes, latent_dim)) 
        for i in range(num_classes):
            indices = y == i
            true_means[i] = mus[indices].mean(axis=0)
            true_logvars[i] = logvars[indices].mean(axis=0)                


        N = 4
        P_set = powerset(list(range(N)))

        players = [player_1, player_2, player_3, player_4] # create a list of the player py files for calling player-specific custom functions


        player_sample_sizes = [base_sample_size for _ in range(N)] # a list of N numbers

        player_sample_size_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 

        player_FI_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 

        player_shapley_lists = [[] for _ in range(N)] # a list of N lists, each of length=max_iterations 


        for i in range(max_iteration):
            # Progress
            print("----------------- Iteration: {}/{} -----------------".format(i + 1, max_iteration))
            print("Sample size: {}".format(player_sample_sizes))
            
            # Record current sample sizes
            for player_sample_size_list, player_sample_size in zip(player_sample_size_lists, player_sample_sizes):
                player_sample_size_list.append(player_sample_size)
            
            # Generate the sample kl divergences
            sample_kl_1, sampled_vae_mus_1, sampled_vae_logvars_1 = player_1.sample_kl_divergences(
                [player_sample_sizes[0]], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample1)

            sample_kl_2, sampled_vae_mus_2, sampled_vae_logvars_2 = player_2.sample_kl_divergences(
                [player_sample_sizes[1]], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample2)


            sample_kl_3, sampled_vae_mus_3, sampled_vae_logvars_3 = player_3.sample_kl_divergences(
                [player_sample_sizes[2]], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample3)

            sample_kl_4, sampled_vae_mus_4, sampled_vae_logvars_4 = player_4.sample_kl_divergences(
                [player_sample_sizes[3]], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample4)


            player_sample_kl_list = [sample_kl_1, sample_kl_2, sample_kl_3, sample_kl_4] # a list of N numbers for each iteration

            player_index_data_dict = {0:[sampled_vae_mus_1, sampled_vae_logvars_1], 1:[sampled_vae_mus_2, sampled_vae_logvars_2],\
                2:[sampled_vae_mus_3, sampled_vae_logvars_3], 3:[sampled_vae_mus_4, sampled_vae_logvars_4]}



            sample_kls = defaultdict(float) # a dict for later calculation of SV

            # update the dict for individual player sample_kl
            for player_index, player_sample_kl in enumerate(player_sample_kl_list):
                 sample_kls[tuple([player_index])] = np.squeeze(np.asarray(player_sample_kl)) 
            

            for subset in P_set:
                print(subset)
                if len(subset) <= 1:continue
                
                else:
                    print("Executing the sample kl divergences for :", subset)
                    estimated_kl_values, post_mean, post_cov, _ = sample_kl_divergences(
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


            BV_coalition_weight = 1./ 2**(N-1)
            for index in range(N):
                sample_shapleys[index] = 0
                for subset in P_set:

                    if index not in subset:
                        subset_included = tuple(sorted(subset + [index]))

                        C = len(subset) 
                        if C == 0:
                            sample_shapleys[index] +=  BV_coalition_weight * \
                            sample_kls[subset_included]
                        else:                    
                            sample_shapleys[index] +=  BV_coalition_weight * \
                            (sample_kls[subset_included] - sample_kls[tuple(subset)])

            for player_index, player_shapley_list in enumerate(player_shapley_lists):
                player_shapley_list.append(sample_shapleys[player_index])
                         
            
            theano.config.compute_test_value = 'ignore'
            
            # Get the current parameter estimate
            # shape: c * latent_dim
            estimated_means = post_mean[:num_classes*latent_dim].reshape(num_classes, latent_dim)

            # p12_estimated_means.append(estimated_means)

            # shape: c * (latent_dim * latent_dim)
            M = latent_dim
            estimated_covs  = [post_cov[k * M:(k+1  )*M, k * M: ( k + 1) * M] for k in range(num_classes)   ]
            
            # p12_estimated_covs.append(estimated_covs)


            player_FI_dets = []

            for player_index, player in enumerate(players):
                
                player_data = player_index_data_dict[player_index]
                
                # calling the custom estimate_FI for each player
                
                # for MNIST, there are multiple emp_Fishers, one for each class
                emp_Fishers = player.estimate_FI(player_data, [estimated_means, estimated_covs])
    

                player_FI_det = sum( [np.linalg.det(emp_Fisher) for emp_Fisher in emp_Fishers ])

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

            np.savetxt(oj(exp_dir, 'cumulative_{}.txt'.format(str(player_index+1))), player_sample_size_lists[player_index])
            
            np.savetxt(oj(exp_dir, 'shapley_fair_{}.txt'.format(str(player_index+1))), player_shapley_lists[player_index])
            
            np.savetxt(oj(exp_dir, 'FI_det_{}.txt'.format(str(player_index+1))), player_FI_lists[player_index])

        log_file.close()


        # np.savetxt(oj(exp_dir, "estimated_means.txt"), p12_estimated_means)
        # np.savetxt(oj(exp_dir, "estimated_covs.txt"), p12_estimated_covs)

