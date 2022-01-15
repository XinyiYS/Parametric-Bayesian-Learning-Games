
# This is exactly the same as the original player_2.py
# repurposed to test out the multi-player scenario for 3 players


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
theano.config.compute_test_value = 'off'
import pymc3 as pm

import regression_game as div
import parameters as pr

# Copy global parameters
num_params = pr.num_params
true_param = pr.true_param

# Get a random covariance
cov = 2.5
data_cov = np.diag(np.full(num_params, cov))

# Generate data from player 2
def generate(sample_size):
    # Player 2's data comes from the normal distribution
    # Where the mean is the true parameter and covariance is known (sampled above)
    x = np.random.multivariate_normal(mean=true_param, cov=data_cov, size=sample_size)
    return x

def generate_and_plot(sample_size, ylim):
    pass

"""Define the dlogL function using Theano's auto gradient"""    
# Input variables
x = T.dvector('x')
theta = T.dvector('theta')    

# Define the logL function
logL_enum = T.dot(T.transpose(x - theta), np.linalg.inv(data_cov))
logL_enum = T.dot(logL_enum, x - theta) / -2
logL_enum = T.exp(logL_enum)
logL_denom = T.sqrt(T.pow((2 * np.pi), num_params) * np.linalg.det(data_cov)) 
logL = T.log(logL_enum/logL_denom)

# The dlog likelihood function
dlogL = theano.function([x, theta], T.grad(logL, theta))

'''
def estimate_fisher_information(sample_size):
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        # Generate one sample
        sample_x = generate(1)
        
        # Estimate the Fisher information
        sample_dlogL = dlogL(sample_x[0], true_param)
        
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher = emp_Fisher / sample_size
    return emp_Fisher
'''

def estimate_FI(player_data, estimated_param, num_params):
    # player 2
    [data_x2] = player_data
    emp_Fisher_2 = np.zeros((num_params, num_params))
    for data_point_x in data_x2[0][0]:
        sample_dlogL = dlogL(data_point_x, estimated_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher_2 += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher_2 = emp_Fisher_2 / len(data_x2[0][0])
    
    return emp_Fisher_2



### Sample KL divergences from the player ###
def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov):  
    # Computed outputs
    estimated_kl_values = []
    
    # Store the generated data
    generated_data_x = []
    
    # For tracking progress
    progress = 1
    
    # For the sample range
    for sample_size in sample_size_range:
        
        # The estimated divergences for a sample size
        estimated_kl = []
        # For storing data with the same sample_size
        data_x = []
        
        # Build the pymc3 model
        with pm.Model() as model:
            
            # Specify the prior
            pmTheta = pm.MvNormal('pmTheta', mu=prior_mean, cov=prior_cov, shape=(1, num_params))
                                       
            # Data holders
            sample_x = generate(sample_size)
            pmData_x = pm.Data('pmData_x', sample_x)

            # Specify the observed variables)
            pm_x = pm.MvNormal('pm_x', mu=pmTheta, cov=data_cov, observed=pmData_x)
            
            for j in range(num_samples):
                # Show progress
                print('player 2 progress: {}/{}'.format(
                    progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                sample_x = generate(sample_size)
                pm.set_data({'pmData_x': sample_x})
                
                # Store the data
                data_x.append(sample_x)
                
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
        
        generated_data_x.append(data_x)
        
    return estimated_kl_values, generated_data_x
