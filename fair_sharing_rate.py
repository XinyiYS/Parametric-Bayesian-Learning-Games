import numpy as np
import matplotlib.pyplot as plt

import parameters as pr
import player_1 as player_1
import player_2 as player_2
import player_12 as player_12

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
    sample_kl_1, data_x1, data_y1 = player_1.sample_kl_divergences(
        [p1_sample_size], 1, posterior_sample_size, prior_mean, prior_cov)
    sample_kl_2, data_x2 = player_2.sample_kl_divergences(
        [p2_sample_size], 1, posterior_sample_size, prior_mean, prior_cov)
    sample_kl_12, post_mean = player_12.sample_kl_divergences(
        [p1_sample_size + p2_sample_size], 1,
        posterior_sample_size, prior_mean, prior_cov, 
        data_x1, data_y1, data_x2)
    
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

np.savetxt("cumulative_1.txt", p1_sample_size_list)
np.savetxt("cumulative_2.txt", p2_sample_size_list)

np.savetxt("shapley_fair_1.txt", p1_shapley_list)
np.savetxt("shapley_fair_2.txt", p2_shapley_list)

# Plot sample sizes
plt.plot(p1_sample_size_list, linestyle='--', color='red')
plt.plot(p2_sample_size_list, linestyle='--', color='blue')
plt.savefig('output_sharing_rate.pdf')
plt.show()
plt.clf()    

# Plot the shapley value
plt.plot(p1_shapley_list, linestyle='--', color='red')
plt.plot(p2_shapley_list, linestyle='--', color='blue')
plt.savefig('output_shapley_fair.pdf')
plt.show()
plt.clf()    

