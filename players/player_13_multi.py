

# This is exactly the same as the original player_12.py
# repurposed to test out the multi-player scenario for 3 players

# TODO: modify to suit player 13


import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

import player_1_multi as p1
import player_3_multi as p3
import regression_game as div
import fourier as fr

# Copy the global parameters
import parameters as pr
num_params = pr.num_params
true_param = pr.true_param

def estimate_fisher_information(sample_size):
    
    sample_x1, sample_y1 = p1.generate(sample_size)
    sample_x3, sample_y3 = p3.generate(sample_size)
    
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        
        dlogL1 = p1.dlogL(sample_x1[i], sample_y1[i], true_param)
        dlogL3 = p3.dlogL(sample_x3[i], sample_y3[i], true_param)

        dlogL = dlogL1 + dlogL3
        
        dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(dlogL, np.transpose(dlogL))
    emp_Fisher = emp_Fisher / sample_size
    return emp_Fisher

### Sample KL divergences from player 1+2 ###
# Function for computing the mean of output given an input and a parameter
# This is for player 1
def compute_mean(x, theta):
    mu_y = 0
    L = p1.domain[1] - p1.domain[0]
    freq_cos = 1
    freq_sin = 1
    
    for i in range(num_params):
        # Compute the basis value
        value = 1
        if i>0:
            if ((i % 2) == 1):
                # Odd
                value = np.cos(2 * freq_cos * np.pi / L * x)                
                freq_cos = freq_cos + 1
            else:
                # Even
                value = np.sin(2 * freq_sin * np.pi / L * x)                
                freq_sin = freq_sin + 1
        # Multiply by its coefficient
        value = theta[0][i] * value        
        # Sum all values
        mu_y = mu_y + value
    return mu_y

def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov,
                          data_x_1, data_y_1,
                          data_x_3, data_y_3, noise_std_estimate=p3.noise_std_prior): 
    # Computed outputs
    estimated_kl_values = []
    
    # For tracking progress
    progress = 1
    
    # Index for tracking which sample size is now being considered
    i = -1
    
    # For the sample range
    for sample_size in sample_size_range:
        
        # Update the index
        i += 1
        
        # The estimated divergences for a sample size
        estimated_kl = []
        
        # Build the pymc3 model
        with pm.Model() as model:
            
            # Specify the prior
            pmTheta = pm.MvNormal('pmTheta', mu=prior_mean, cov=prior_cov, 
                                  shape=(1, num_params))
            
            # Data holders
            pmData_x_1 = pm.Data('pmData_x_1', data_x_1[i][0])
            pmData_y_1 = pm.Data('pmData_y_1', data_y_1[i][0])

            pmData_x_3 = pm.Data('pmData_x_3', data_x_3[i][0])
            pmData_y_3 = pm.Data('pmData_y_3', data_y_3[i][0])


            # Specify the observed variables
            # From player 1
            pm_x_1 = pm.Normal('pm_x_1', mu=p1.x_mean, sigma=p1.x_std_dev, observed=pmData_x_1)
            pm_y_1 = pm.Normal('pm_y_1', mu=compute_mean(pm_x_1, pmTheta), 
                             sigma=p1.noise_std_dev, observed=pmData_y_1)
            # From player 3
            pm_x_3 = pm.Normal('pm_x_3', mu=p3.x_mean, sigma=p3.x_std_dev, observed=pmData_x_3)
            pm_y_3 = pm.Normal('pm_y_3', mu=compute_mean(pm_x_3, pmTheta), 
                             sigma=noise_std_estimate, observed=pmData_y_3)

            
            for j in range(num_samples):
                # Show progress
                print('player 1+3 progress: {}/{}'.format(
                    progress, len(sample_size_range) * num_samples))
                
                # Assign the data
                pm.set_data({'pmData_x_1': data_x_1[i][j]})
                pm.set_data({'pmData_y_1': data_y_1[i][j]})

                pm.set_data({'pmData_x_3': data_x_3[i][j]})
                pm.set_data({'pmData_y_3': data_y_3[i][j]})
                
                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=0)
                summary = pm.stats.summary(pmTrace)
                
                # Get the sample mean and covariance of the samples
                post_mean = np.array(summary.loc[:,'mean'])
                post_cov = pm.trace_cov(pmTrace)
                                
                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                estimated_kl.append(div.compute_KL(post_mean, post_cov, 
                                                   prior_mean, prior_cov))
                
                # Update progress
                progress += 1
    
        estimated_kl_values.append(estimated_kl)  
    
    return estimated_kl_values, post_mean
