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


print(df.columns)
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

import random
random.seed(7913)

from sklearn.model_selection import train_test_split
X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.4, random_state=42)

ratio_1, ratio_2 = 0.2, 0.05


nan_count_1 = int(len(X_1) * ratio_1)
nan_count_2 = int(len(X_2) * ratio_2)  
# choosing random indexes to put NaN
index_nan_1 = np.random.choice(X_1.size, nan_count_1, replace=False)
index_nan_2 = np.random.choice(X_2.size, nan_count_2, replace=False)  
# adding nan to the data.
X_1.ravel()[index_nan_1] = np.nan
X_2.ravel()[index_nan_2] = np.nan

assert nan_count_1 == np.count_nonzero(np.isnan(X_1))
assert nan_count_2 == np.count_nonzero(np.isnan(X_2))

def impute_with_mean(X):
    #Obtain mean of columns as you need, nanmean is convenient.
    col_mean = np.nanmean(X, axis=0)

    #Find indices that you need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])

    return X

# -- impute nan with averages -- #

X_1 = impute_with_mean(X_1)
X_2 = impute_with_mean(X_2)

assert 0 == np.count_nonzero(np.isnan(X_1))
assert 0 == np.count_nonzero(np.isnan(X_2))



import parameters as pr
import player_1 as player_1
import player_2 as player_2
import player_12 as player_12

import theano
import theano.tensor as T


reg_lambda = 1e-3

from linear_regression_sampling import leverage_iid_sampling

def p1_generate_fcn(sample_size):
    ''' can we use volume sampling to guarantee unbiased-ness? '''
    local_size = 100
    sample_theta = np.zeros((sample_size, num_params))    
    sample_x, sample_y = [], []  
    for i in range(sample_size):
        # indices = np.random.choice(np.arange(len(X_1)), size=local_size)
        # indices = fast_reg_vol_sampling(X_1, local_size, reg_lambda)
        indices = leverage_iid_sampling(X_1, local_size, reg_lambda)
        x, y = X_1[indices], y_1[indices]
        theta_hat = np.linalg.inv(x.T @ x + reg_lambda * np.identity(x.shape[1])) @ x.T @ y  
        sample_theta[i] = theta_hat
        sample_x.append(x)
        sample_y.append(y)

    sample_x = np.asarray(sample_x).reshape(sample_size, local_size, -1)
    sample_y = np.asarray(sample_y).reshape(sample_size, local_size)
    return sample_x, sample_y, sample_theta


def p2_generate_fcn(sample_size):
    x = np.random.multivariate_normal(mean=player_2.true_param, cov=player_2.data_cov, size=sample_size)
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
    print("Iteration: {}/{} ".format(i + 1, max_iteration))
    print("Sample size: {} + {}".format(p1_sample_size, p2_sample_size))

    # Record current sample sizes
    p1_sample_size_list.append(p1_sample_size)
    p2_sample_size_list.append(p2_sample_size)
    
    # Generate the sample kl divergences
    sample_kl_1, data_x1, data_y1, data_theta1 = player_1.sample_kl_divergences(
        [p1_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=p1_generate_fcn)
    sample_kl_2, data_x2 = player_2.sample_kl_divergences(
        [p2_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=None)
    sample_kl_12, post_mean = player_12.sample_kl_divergences(
        [p1_sample_size + p2_sample_size], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_theta1, data_x2)
    
    # Current Shapley value
    sample_shapley_1 = np.multiply(0.5, sample_kl_1) + np.multiply(0.5, (np.subtract(sample_kl_12, sample_kl_2)))
    sample_shapley_2 = np.multiply(0.5, sample_kl_2) + np.multiply(0.5, (np.subtract(sample_kl_12, sample_kl_1)))
    p1_shapley_list.append(sample_shapley_1.flatten())
    p2_shapley_list.append(sample_shapley_2.flatten())
    
    # Get the current parameter estimate
    estimated_param = post_mean


    # Estimate the Fisher informations at the estimated parameter
    # player 1
    emp_Fisher_1 = np.zeros((num_params, num_params))
    for j in range(len(data_theta1[0][0])):
        sample_dlogL = player_1.dlogL(data_x1[0][0][j], data_y1[0][0][j], estimated_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_1 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_1 = emp_Fisher_1 / len(data_theta1[0][0])
    
    # print("Player 1 fisher:", emp_Fisher_1 )
    # player 2
    emp_Fisher_2 = np.zeros((num_params, num_params))
    for data_point_x in data_x2[0][0]:
        sample_dlogL = player_2.dlogL(data_point_x, estimated_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_2 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_2 = emp_Fisher_2 / len(data_x2[0][0])
    
    # Store the determinant of estimated Fisher information
    p1_FI_list.append(np.linalg.det(emp_Fisher_1))
    p2_FI_list.append(np.linalg.det(emp_Fisher_2))
    
    # Estimate the fair rate & next sample size
    det_F1 = np.linalg.det(emp_Fisher_1)
    det_F2 = np.linalg.det(emp_Fisher_2)
    if det_F1 > det_F2:
        rate = np.power(det_F1 / det_F2, 1.0/num_params)
        p1_sample_size += base_sample_increment
        target = round(p1_sample_size * rate)
        if p2_sample_size < target:
            p2_sample_size += min(target - p2_sample_size, max_sample_increment)
    else: 
        if det_F2 > det_F1:
            rate = np.power(det_F2 / det_F1, 1.0/num_params)
            p2_sample_size += base_sample_increment
            target = round(p2_sample_size * rate)
            if p1_sample_size < target:
                p1_sample_size += min(target - p1_sample_size, max_sample_increment)
        else:
            p1_sample_size += base_sample_increment
            p2_sample_size += base_sample_increment   

exp_dir = 'CaliH/lvg_iid'

import os 
from os.path import join as oj
os.makedirs(exp_dir, exist_ok=True)


np.savetxt(oj(exp_dir, "cumulative_1.txt"), p1_sample_size_list)
np.savetxt(oj(exp_dir, "cumulative_2.txt"), p2_sample_size_list)

np.savetxt(oj(exp_dir, "shapley_fair_1.txt"), p1_shapley_list)
np.savetxt(oj(exp_dir, "shapley_fair_2.txt"), p2_shapley_list)

# Plot sample sizes
plt.plot(p1_sample_size_list, linestyle='--', color='red', label='1')
plt.plot(p2_sample_size_list, linestyle='--', color='blue', label='2')
plt.legend()
plt.savefig(oj(exp_dir, 'output_sharing_rate.pdf'))
plt.show()
plt.clf()    

# Plot the shapley value
plt.plot(p1_shapley_list, linestyle='--', color='red', label='1')
plt.plot(p2_shapley_list, linestyle='--', color='blue', label='1')
plt.legend()
plt.savefig(oj(exp_dir, 'output_shapley_fair.pdf'))
plt.show()
plt.clf()    

