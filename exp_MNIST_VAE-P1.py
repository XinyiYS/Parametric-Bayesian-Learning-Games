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

    
    digit_counts = np.random.multinomial(data_size, P1_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P1_DATA_SIZE]  , count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    return sampled_vae_mus, sampled_vae_logvars


def sample2(data_size):
    # digit_counts = np.ceil(data_size * np.asarray(P2_PROBS)).astype(int)

    digit_counts = np.random.multinomial(data_size, P2_PROBS, size=1).squeeze()

    sampled_digit_indices = [choice(indices_by_digit[c][:P2_DATA_SIZE], count) for c, count in enumerate(digit_counts)]

    sampled_vae_mus = [ mus[digit_indices] for digit_indices in sampled_digit_indices]
    sampled_vae_logvars = [ logvars[digit_indices] for digit_indices in sampled_digit_indices]


    return sampled_vae_mus, sampled_vae_logvars



import theanofrom params import parameters_MNIST as pr
from players import player_1_gmm as player_1, player_2_gmm as player_2, player_12_gmm as player_12

import theano.tensor as T



P2_BALANCE = 0.1
P2_DATA_SIZE = 1000

name = 'MNIST_VAE'

for P1_DATA_SIZE, P2_DATA_SIZE in [(1000, 5000), (5000, 5000)]:   
# for P1_DATA_SIZE in [1000, 5000]:
    # for P2_DATA_SIZE in [1000, 5000]:
    for P1_BALANCE in np.linspace(0.1, 0.9, 9):

        P1_PROBS = [1 - P1_BALANCE, P1_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]
        P2_PROBS = [1 - P2_BALANCE, P2_BALANCE,  0, 0, 0, 0, 0, 0, 0, 0]

        exp_dir = oj(name, "P1-size-{}_P2-size-{}_P1-ratio-{}".format(str(P1_DATA_SIZE), str(P2_DATA_SIZE), str(P1_BALANCE)) )

        os.makedirs(exp_dir, exist_ok=True)

        log_file = open(oj(exp_dir, 'log')  ,"w")
        sys.stdout = log_file

        with open(oj(exp_dir, 'settings.txt'), 'w') as f:

            f.write("Experiment Parameters: \n")

            f.write("P1_DATA_SIZE =  " + str(P1_DATA_SIZE) + '\n')
            f.write("P2_DATA_SIZE =  " + str(P2_DATA_SIZE)+ '\n')

            f.write("P1_BALANCE =  " + str(P1_BALANCE)+ '\n')
            f.write("P2_BALANCE =  " + str(P2_BALANCE)+ '\n')

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

        num_classes = ((np.asarray(P1_PROBS) + np.asarray(P1_PROBS)) > 0).sum()
        print("Total number of non-zero probs classes:", num_classes)
        pr.num_classes = num_classes


        sampled_vae_mus_1, sampled_vae_logvars_1 = sample1(5)
        sampled_vae_mus_2, sampled_vae_logvars_2 = sample2(5)


        pr.latent_dim = latent_dim
        true_means = np.zeros((num_classes, latent_dim)) # 10 dimensional mean for each class

        player_1.num_classes = num_classes
        player_1.num_params = num_classes * latent_dim
        player_1.latent_dim = latent_dim
        

        player_2.num_classes = num_classes
        player_2.num_params = num_classes * latent_dim
        player_2.latent_dim = latent_dim

        player_12.num_classes = num_classes
        player_12.num_params = num_classes * latent_dim
        player_12.latent_dim = latent_dim


        true_logvars = np.zeros((num_classes, latent_dim)) 
        for i in range(num_classes):
            indices = y == i
            true_means[i] = mus[indices].mean(axis=0)
            true_logvars[i] = logvars[indices].mean(axis=0)                


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


        p12_estimated_means = []
        p12_estimated_covs = []

        for i in range(max_iteration):
            # Progress
            print("----------------- Iteration: {}/{} -----------------".format(i + 1, max_iteration))
            print("Sample size: {} + {}".format(p1_sample_size, p2_sample_size))

            # Record current sample sizes
            p1_sample_size_list.append(p1_sample_size)
            p2_sample_size_list.append(p2_sample_size)
            
            # Generate the sample kl divergences
            sample_kl_1, sampled_vae_mus_1, sampled_vae_logvars_1 = player_1.sample_kl_divergences(
                [p1_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample1)

            sample_kl_2, sampled_vae_mus_2, sampled_vae_logvars_2 = player_2.sample_kl_divergences(
                [p2_sample_size], 1, posterior_sample_size, prior_mean, prior_cov, generate_fcn=sample2)

            sample_kl_12, post_mean, post_cov, summary = player_12.sample_kl_divergences(
                [p1_sample_size + p2_sample_size], 1,
                posterior_sample_size, prior_mean, prior_cov, 
                sampled_vae_mus_1, sampled_vae_logvars_1,
                sampled_vae_mus_2, sampled_vae_logvars_2
                )
            

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
            # shape: c * latent_dim
            estimated_means = post_mean[:num_classes*latent_dim].reshape(num_classes, latent_dim)

            p12_estimated_means.append(estimated_means)

            # shape: c * (latent_dim * latent_dim)
            M = latent_dim
            estimated_covs  = [post_cov[k * M:(k+1  )*M, k * M: ( k + 1) * M] for k in range(num_classes)   ]
            
            p12_estimated_covs.append(estimated_covs)

            # Estimate the Fisher informations at the estimated parameter
            # player 1
            emp_Fisher_1s = [np.zeros((latent_dim, latent_dim))  for c in range(num_classes)]

            for c in range(num_classes):    
                if len(sampled_vae_mus_1[0][0][c]) == 0: continue

                sample_dlogL = player_1.dlogL(estimated_means[c].reshape(1, latent_dim), sampled_vae_mus_1[0][0][c], estimated_covs[c])

                sample_dlogL.shape = (latent_dim, 1)
                
                emp_Fisher_1s[c] += np.matmul(sample_dlogL, np.transpose(sample_dlogL))

                emp_Fisher_1s[c] = emp_Fisher_1s[c] / len(sampled_vae_logvars_1[0][0][c])


            emp_Fisher_2s = [np.zeros((latent_dim, latent_dim))  for c in range(num_classes)]

            for c in range(num_classes):    

                if len(sampled_vae_mus_2[0][0][c]) == 0: continue

                sample_dlogL = player_1.dlogL(estimated_means[c].reshape(1, latent_dim), sampled_vae_mus_2[0][0][c], estimated_covs[c])
                sample_dlogL.shape = (latent_dim, 1)
                
                emp_Fisher_2s[c] += np.matmul(sample_dlogL, np.transpose(sample_dlogL))

                emp_Fisher_2s[c] = emp_Fisher_1s[c] / len(sampled_vae_logvars_2[0][0][c])


            # Store the determinant of estimated Fisher information
            det_F1 = sum( [np.linalg.det(emp_Fisher_1) for emp_Fisher_1 in emp_Fisher_1s ])
            det_F2 = sum( [np.linalg.det(emp_Fisher_2) for emp_Fisher_2 in emp_Fisher_2s ])

            p1_FI_list.append( det_F1 )
            p2_FI_list.append( det_F2 )

            # Estimate the fair rate & next sample size
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

        np.savetxt(oj(exp_dir, "estimated_means.txt"), p12_estimated_means)
        np.savetxt(oj(exp_dir, "estimated_covs.txt"), p12_estimated_covs)

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
