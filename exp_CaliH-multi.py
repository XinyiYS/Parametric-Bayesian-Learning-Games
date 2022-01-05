import os 
import sys
from os.path import join as oj

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.datasets import fetch_california_housing
data_fir = '.data/California_housing'
X, y = fetch_california_housing(data_home=data_fir, download_if_missing=True, return_X_y=True, as_frame=False)
bunch = fetch_california_housing(data_home=data_fir, download_if_missing=True, return_X_y=False, as_frame=True)
df = bunch['frame']

from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.model_selection import train_test_split

# print(df.columns)
df = df.drop(columns=['Latitude', 'Longitude'])

# print(df.describe())
# print(df.head())

y = df['MedHouseVal'].values
X = df.drop(columns=['MedHouseVal']).values

X = StandardScaler().fit_transform(X=X)
y = minmax_scale(y)

'''
curr_best = float('inf')
best_w = None
best_lambda = None
for reg_lambda in np.logspace(-15, 0, num=1000):
    w = np.linalg.inv(X.T @ X + reg_lambda * np.identity(X.shape[1])) @ X.T @ y
    risk = np.linalg.norm(w @ X.T - y)
    if risk < curr_best:
        best_w = w
        curr_best = risk
        best_lambda = reg_lambda
        print('updating at: ', reg_lambda, curr_best)

print("Best lambda: {}, empirical risk:{} , w:{}.".format(best_lambda, curr_best, best_w))
true_params = best_w
best_lambda = best_lambda
'''
# true_params = [0.21031031,  0.04282435, -0.10801392,  0.09709814,  0.00516233, -0.01044389]
# best_lambda = 2.47e-07


import parameters_CaliH as pr
import player_1 as player_1
import player_2 as player_2
import player_3 as player_3
import player_4 as player_4

from player_manager_LR_multi import sample_kl_divergences
from utils import powerset


from collections import defaultdict
from math import factorial as fac
import numpy as np
from scipy.stats import tvar

import matplotlib.pyplot as plt

import theano
import theano.tensor as T

reg_lambda = 1e-3
from linear_regression_sampling import leverage_iid_sampling

import random
random.seed(7913)
np.random.seed(7913)


def impute_with_mean(X):
    #Obtain mean of columns as you need, nanmean is convenient.
    col_mean = np.nanmean(X, axis=0)

    #Find indices that you need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X


# P1_DATA_SIZE = 10000 # 100, 500, 2000
# P1_LOCAL_SAMPLE_SIZE =  100 # 10, 100, 500
# P1_LOCAL_SAMPLE = 'iid' # iid, lvg_iid


P2_DATA_RATIO = 0.1  # 0.01, 0.1, 0.5
P2_NAN_RATIO = 0.2 # 0.1, 0.2

for P1_DATA_SIZE in [1000, 5000]:
    for P1_LOCAL_SAMPLE_SIZE in [100, 500]:
        # for P1_LOCAL_SAMPLE in ['iid' ,'lvg_iid']:
        exp_dir = oj('multiplayer', 'CaliH', "P1-{}_{}".format(str(P1_DATA_SIZE), str(P1_LOCAL_SAMPLE_SIZE)) )

        os.makedirs(exp_dir, exist_ok=True)

        log_file = open(oj(exp_dir, 'log')  ,"w")
        sys.stdout = log_file

        with open(oj(exp_dir, 'settings.txt'), 'w') as f:

            f.write("Experiment Parameters: \n")

            f.write("P1_DATA_SIZE =  " + str(P1_DATA_SIZE) + '\n')
            f.write("P1_LOCAL_SAMPLE_SIZE =  " + str(P1_LOCAL_SAMPLE_SIZE)+ '\n')
            # f.write("P1_LOCAL_SAMPLE =  " + str(P1_LOCAL_SAMPLE)+ '\n')
            f.write("P2_DATA_RATIO =  " + str(P2_DATA_RATIO)+ '\n')
            f.write("P2_NAN_RATIO =  " + str(P2_NAN_RATIO)+ '\n')
            f.write("P1_DATA_SIZE =  " + str(P1_DATA_SIZE)+ '\n')

            f.write("Algorithm Parameters: \n")

            f.write("fisher_sample_size =  " + str(pr.fisher_sample_size)+ '\n')
            f.write("posterior_sample_size =  " + str(pr.posterior_sample_size)+ '\n')
            f.write("tuning_step =  " + str(pr.tuning_step)+ '\n')
            f.write("num_params =  " + str(pr.num_params)+ '\n')
            f.write("true_param =  " + str(pr.true_param)+ '\n')
            f.write("best_lambda =  " + str(pr.best_lambda)+ '\n')

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

 
        indices_1 = np.random.choice(list(range(len(X))), P1_DATA_SIZE)
        X_1, y_1 = X[indices_1], y[indices_1]

        indices_3 = np.random.choice(list(range(len(X))), P1_DATA_SIZE)
        X_3, y_3 = X[indices_3], y[indices_3]


        # set player 2's data to be part of the full data with missing data
        indices_2 = np.random.choice(list(range(len(X))), int(P2_DATA_RATIO * len(X)))
        X_2, y_2 = X[indices_2], y[indices_2]

        # -- create nans and impute nan with averages -- #
        nan_count_2 = int(len(X_2) * P2_NAN_RATIO)  
        index_nan_2 = np.random.choice(X_2.size, nan_count_2, replace=False)  
        X_2.ravel()[index_nan_2] = np.nan
        assert nan_count_2 == np.count_nonzero(np.isnan(X_2))

        X_2 = impute_with_mean(X_2)
        assert 0 == np.count_nonzero(np.isnan(X_2))


        # set player 4's data to be part of the full data with missing data
        indices_4 = np.random.choice(list(range(len(X))), int(P2_DATA_RATIO * len(X)))
        X_4, y_4 = X[indices_4], y[indices_4]

        # -- create nans and impute nan with averages -- #
        nan_count_4 = int(len(X_4) * P2_NAN_RATIO)  
        index_nan_4 = np.random.choice(X_4.size, nan_count_4, replace=False)  
        X_4.ravel()[index_nan_4] = np.nan
        assert nan_count_4 == np.count_nonzero(np.isnan(X_4))

        X_4 = impute_with_mean(X_4)
        assert 0 == np.count_nonzero(np.isnan(X_4))

        def p1_generate_fcn(sample_size):
            ''' can we use volume sampling to guarantee unbiased-ness? '''
            sample_theta = np.zeros((sample_size, num_params))    
            for i in range(sample_size):
                # indices = fast_reg_vol_sampling(X_1, local_size, reg_lambda)
                
                # depends on the global variable of X_1
                indices = leverage_iid_sampling(X_1, P1_LOCAL_SAMPLE_SIZE, reg_lambda)

                x, y = X_1[indices], y_1[indices]

                theta_hat = np.linalg.inv(x.T @ x + reg_lambda * np.identity(x.shape[1])) @ x.T @ y  
                sample_theta[i] = theta_hat
            return sample_theta


        print("Global True param:", pr.true_param)
        beta = 1
        
        player_2.data_cov = np.diag(np.full(num_params, 2.5))

        # Player 2 maintains a posterior of BLR trained on the data
        player_2.data_cov_inv = np.linalg.inv(player_2.data_cov) + beta * (X_2.T @ X_2)
        player_2.data_cov = np.linalg.inv(player_2.data_cov_inv)
        player_2.data_mean = np.linalg.inv(X_2.T @ X_2 + reg_lambda * np.identity(X_2.shape[1])) @ X_2.T @ y_2

        print("Player 2 maintained prior mean:", player_2.data_mean)
        print("Player 2 maintained prior cov:", player_2.data_cov)

        def p2_generate_fcn(sample_size):
            return np.random.multivariate_normal(mean=player_2.data_mean, cov=player_2.data_cov, size=sample_size)

        def p3_generate_fcn(sample_size):
            ''' can we use volume sampling to guarantee unbiased-ness? '''
            sample_theta = np.zeros((sample_size, num_params))    
            for i in range(sample_size):
                # indices = fast_reg_vol_sampling(X_1, local_size, reg_lambda)
                
                # depends on the global variable of X_1
                indices = np.random.choice(np.arange(len(X_1)), size=P1_LOCAL_SAMPLE_SIZE)

                x, y = X_1[indices], y_1[indices]

                theta_hat = np.linalg.inv(x.T @ x + reg_lambda * np.identity(x.shape[1])) @ x.T @ y  
                sample_theta[i] = theta_hat
            return sample_theta

        # Player 4 maintains a posterior of BLR trained on the data
        player_4.data_cov = np.diag(np.full(num_params, 3.5))
        
        player_4.data_cov_inv = np.linalg.inv(player_4.data_cov) + beta * (X_4.T @ X_4)
        player_4.data_cov = np.linalg.inv(player_4.data_cov_inv)
        player_4.data_mean = np.linalg.inv(X_4.T @ X_4 + reg_lambda * np.identity(X_4.shape[1])) @ X_4.T @ y_4

        print("Player 4 maintained prior mean:", player_4.data_mean)
        print("Player 4 maintained prior cov:", player_4.data_cov)

        def p4_generate_fcn(sample_size):
            return np.random.multivariate_normal(mean=player_4.data_mean, cov=player_4.data_cov, size=sample_size)

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
            # sample_kl_1, data_x1, data_y1, data_theta1 = player_1.sample_kl_divergences(
            sample_kl_1, data_theta1 = player_1.sample_kl_divergences(
                [player_sample_sizes[0]], 1, posterior_sample_size, prior_mean, prior_cov, num_params, generate_fcn=p1_generate_fcn)
            
            sample_kl_2, data_x2 = player_2.sample_kl_divergences(
                [player_sample_sizes[1]], 1, posterior_sample_size, prior_mean, prior_cov, num_params, generate_fcn=p2_generate_fcn)
            

            sample_kl_3, data_theta3 = player_3.sample_kl_divergences(
                [player_sample_sizes[2]], 1, posterior_sample_size, prior_mean, prior_cov, num_params, generate_fcn=p3_generate_fcn)
            

            sample_kl_4, data_x4 = player_4.sample_kl_divergences(
                [player_sample_sizes[3]], 1, posterior_sample_size, prior_mean, prior_cov, num_params, generate_fcn=p4_generate_fcn)
            

            player_sample_kl_list = [sample_kl_1, sample_kl_2, sample_kl_3, sample_kl_4] # a list of N numbers for each iteration

            player_index_data_dict = {0:[data_theta1], 1:[data_x2],\
                2:[data_theta3], 3:[data_x4]}


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
                        posterior_sample_size, prior_mean, prior_cov, num_params,            
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
                         

            
            theano.config.compute_test_value = 'ignore'

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

            np.savetxt(oj(exp_dir, 'cumulative_{}.txt'.format(str(player_index+1))), player_sample_size_lists[player_index])
            
            np.savetxt(oj(exp_dir, 'shapley_fair_{}.txt'.format(str(player_index+1))), player_shapley_lists[player_index])
            
            np.savetxt(oj(exp_dir, 'FI_det_{}.txt'.format(str(player_index+1))), player_FI_lists[player_index])

        log_file.close()
