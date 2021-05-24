import os 
import sys
from os.path import join as oj

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.model_selection import train_test_split


X = pd.read_csv('.data/House_sales/KingH-NN_features.csv')
y = pd.read_csv('.data/House_sales/KingH-labels.csv').values

# X = MinMaxScaler().fit_transform(X=X.values)

X = StandardScaler().fit_transform(X=X)
y = minmax_scale(y)


import parameters_KingH as pr
import player_1 as player_1
import player_2 as player_2
import player_12 as player_12

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


P1_DATA_SIZE = 10000 # 100, 500, 2000
P1_LOCAL_SAMPLE_SIZE =  100 # 10, 100, 500
P1_LOCAL_SAMPLE = 'iid' # iid, lvg_iid

# P2_DATA_RATIO = 0.1  # 0.01,  0.1, 0.5
# P2_NAN_RATIO = 0.1 # 0.1, 0.2

name = 'KingH'

for P2_DATA_RATIO in [0.01, 0.1, 0.5]:
    for P2_NAN_RATIO in [0.05, 0.1, 0.4]:

        exp_dir = oj(name, "P2-{}_{}_{}".format(str(P1_DATA_SIZE), str(P1_LOCAL_SAMPLE_SIZE), str(P1_LOCAL_SAMPLE)) )
        
        os.makedirs(exp_dir, exist_ok=True)

        log_file = open(oj(exp_dir, 'log')  ,"w")
        sys.stdout = log_file

        with open(oj(exp_dir, 'settings.txt'), 'w') as f:

            f.write("Experiment Parameters: \n")

            f.write("P1_DATA_SIZE =  " + str(P1_DATA_SIZE) + '\n')
            f.write("P1_LOCAL_SAMPLE_SIZE =  " + str(P1_LOCAL_SAMPLE_SIZE)+ '\n')
            f.write("P1_LOCAL_SAMPLE =  " + str(P1_LOCAL_SAMPLE)+ '\n')
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


            indices_1 = np.random.choice(list(range(len(X))), P1_DATA_SIZE)
            # set player 1's data to be the full data
            X_1, y_1 = X[indices_1], y[indices_1]

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

            def p1_generate_fcn(sample_size):
                ''' can we use volume sampling to guarantee unbiased-ness? '''
                sample_theta = np.zeros((sample_size, num_params))    
                for i in range(sample_size):
                    # indices = fast_reg_vol_sampling(X_1, local_size, reg_lambda)
                    
                    # depends on the global variable of X_1
                    if P1_LOCAL_SAMPLE == 'lvg_iid':
                        indices = leverage_iid_sampling(X_1, P1_LOCAL_SAMPLE_SIZE, reg_lambda)
                    else:
                        indices = np.random.choice(np.arange(len(X_1)), size=P1_LOCAL_SAMPLE_SIZE)

                    x, y = X_1[indices], y_1[indices]

                    theta_hat = np.linalg.inv(x.T @ x + reg_lambda * np.identity(x.shape[1])) @ x.T @ y  
                    sample_theta[i] = theta_hat.squeeze()
                return sample_theta




            ''' Set up player 1 '''
            player_1.num_params = pr.num_params
            player_1.true_param = pr.true_param

            ''' Set up player 2 '''

            print("Global True param:", pr.true_param)
            beta = 1
            player_2.num_params = pr.num_params
            player_2.true_param = pr.true_param

            p2_data_cov = np.diag(np.full(pr.num_params, 2.5))

            player_2.data_cov_inv = p2_data_cov + beta * (X_2.T @ X_2)
            
            player_2.data_cov = np.linalg.inv(player_2.data_cov_inv)
            player_2.data_mean = np.linalg.inv(X_2.T @ X_2 + reg_lambda * np.identity(X_2.shape[1])) @ X_2.T @ y_2
            player_2.data_mean = np.squeeze(player_2.data_mean)

            ''' Set up player 12 '''
            player_12.num_params = pr.num_params
            player_12.true_param = pr.true_param
            print("Player 2 maintained prior mean:", player_2.data_mean)
            print("Player 2 maintained prior cov:", player_2.data_cov)


            def p2_generate_fcn(sample_size):
                x = np.random.multivariate_normal(mean=player_2.data_mean, cov=player_2.data_cov, size=sample_size)
                return x

            posterior_sample_size = pr.posterior_sample_size
            prior_mean = pr.prior_mean
            prior_cov = pr.prior_cov
            num_params = pr.num_params

            base_sample_size = pr.base_sample_size
            base_sample_increment = pr.base_sample_increment
            max_sample_increment = pr.max_sample_increment
            max_iteration = pr.max_iteration

            p1_sample_size = base_sample_size
            p2_sample_size = base_sample_size

            p1_sample_size_list = []
            p2_sample_size_list = []

            p1_FI_list = []
            p2_FI_list = []

            p1_shapley_list = []
            p2_shapley_list = []

            for i in range(max_iteration):
                # Progress
                print("----------------- Iteration: {}/{} -----------------".format(i + 1, max_iteration))
                print("Sample size: {} + {}".format(p1_sample_size, p2_sample_size))

                # Record current sample sizes
                p1_sample_size_list.append(p1_sample_size)
                p2_sample_size_list.append(p2_sample_size)
                
                # Generate the sample kl divergences
                # sample_kl_1, data_x1, data_y1, data_theta1 = player_1.sample_kl_divergences(
                sample_kl_1, data_theta1 = player_1.sample_kl_divergences(
                    [p1_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=p1_generate_fcn)
                sample_kl_2, data_x2 = player_2.sample_kl_divergences(
                    [p2_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=p2_generate_fcn)
                sample_kl_12, post_mean = player_12.sample_kl_divergences(
                    [p1_sample_size + p2_sample_size], 1,
                    posterior_sample_size, prior_mean, prior_cov, 
                    data_theta1, data_x2)
                

                print("Sample kl divergence 1:", sample_kl_1)
                print("Sample kl divergence 2:", sample_kl_2)
                print("Sample kl divergence 12:", sample_kl_12)
                # Current Shapley value
                sample_shapley_1 = np.multiply(0.5, sample_kl_1) + np.multiply(0.5, (np.subtract(sample_kl_12, sample_kl_2)))
                sample_shapley_2 = np.multiply(0.5, sample_kl_2) + np.multiply(0.5, (np.subtract(sample_kl_12, sample_kl_1)))

                print("Shapley value 1 vs. 2:", sample_shapley_1, sample_shapley_2)
                p1_shapley_list.append(sample_shapley_1.flatten())
                p2_shapley_list.append(sample_shapley_2.flatten())
                
                # Get the current parameter estimate
                estimated_param = post_mean

                # Estimate the Fisher informations at the estimated parameter
                # player 1
                emp_Fisher_1 = np.zeros((num_params, num_params))
                
                p1_theta_cov = np.cov(np.concatenate([sample for sample in data_theta1[0]] ), rowvar=False)

                for theta_hat in data_theta1[0][0]:
                    sample_dlogL = player_1.dlogL(theta_hat, estimated_param, p1_theta_cov)
                    sample_dlogL.shape = (num_params, 1)
                    emp_Fisher_1 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
                emp_Fisher_1 = emp_Fisher_1 / len(data_theta1[0][0])

                # print("Player 1 fisher:", emp_Fisher_1 )
                # player 2

                p2_cov_hat = np.cov(np.concatenate([sample for sample in data_x2[0]] ), rowvar=False)
                emp_Fisher_2 = np.zeros((num_params, num_params))
                for data_point_x in data_x2[0][0]:
                    sample_dlogL = player_2.dlogL(data_point_x, estimated_param, p2_cov_hat)
                    sample_dlogL.shape = (num_params, 1)
                    emp_Fisher_2 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
                emp_Fisher_2 = emp_Fisher_2 / len(data_x2[0][0])

                
                # Store the determinant of estimated Fisher information
                p1_FI_list.append(np.linalg.det(emp_Fisher_1))
                p2_FI_list.append(np.linalg.det(emp_Fisher_2))
                
                # Estimate the fair rate & next sample size
                det_F1 = np.linalg.det(emp_Fisher_1)
                det_F2 = np.linalg.det(emp_Fisher_2)
                print('Current det F1 vs F2: ', det_F1, det_F2)
                if det_F1 > det_F2:
                    rate = np.power(det_F1 / det_F2, 1.0/num_params)
                    p1_sample_size += base_sample_increment
                    target = round(p1_sample_size * rate)
                    
                    p2_to_add = 0
                    if p2_sample_size < target:
                        p2_to_add = int(min(target - p2_sample_size, max_sample_increment))
                        p2_sample_size += p2_to_add

                    print("Player 1 vs. 2 additional shared: ", base_sample_increment, p2_to_add)

                    p2_sample_size = int(p2_sample_size)

                elif det_F2 > det_F1:
                        rate = np.power(det_F2 / det_F1, 1.0/num_params)
                        p2_sample_size += base_sample_increment
                        target = round(p2_sample_size * rate)
                        
                        p1_to_add = 0
                        if p1_sample_size < target:
                            p1_to_add = int(min(target - p1_sample_size, max_sample_increment))
                            p1_sample_size += p1_to_add

                        print("Player 1 vs. 2 additional shared: ", p1_to_add, base_sample_increment)
                        p1_sample_size = int(p1_sample_size)
                else:
                    p1_sample_size += base_sample_increment
                    p2_sample_size += base_sample_increment   


            np.savetxt(oj(exp_dir, "cumulative_1.txt"), p1_sample_size_list)
            np.savetxt(oj(exp_dir, "cumulative_2.txt"), p2_sample_size_list)

            np.savetxt(oj(exp_dir, "shapley_fair_1.txt"), p1_shapley_list)
            np.savetxt(oj(exp_dir, "shapley_fair_2.txt"), p2_shapley_list)

            # Plot sample sizes
            plt.plot(p1_sample_size_list, linestyle='--', color='red', label='player 1')
            plt.plot(p2_sample_size_list, linestyle='--', color='blue', label='player 2')
            plt.xlabel('Iteration', fontsize=16)
            plt.ylabel('Cumulative number of points shared', fontsize=16)
            plt.legend()
            plt.savefig(oj(exp_dir, 'output_sharing_rate.pdf'))
            # plt.show()
            plt.clf()    

            # Plot the shapley value
            plt.plot(p1_shapley_list, linestyle='--', color='red', label='player 1')
            plt.plot(p2_shapley_list, linestyle='--', color='blue', label='player 2')
            plt.xlabel('Iteration', fontsize=16)
            plt.ylabel('Shapley value', fontsize=16)
            plt.legend()
            plt.savefig(oj(exp_dir, 'output_shapley_fair.pdf'))
            # plt.show()
            plt.clf()
            plt.close()    

            log_file.close()
