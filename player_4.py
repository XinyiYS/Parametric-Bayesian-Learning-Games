import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

import pymc3 as pm
theano.config.compute_test_value = 'off'
theano.config.compute_test_value = 'ignore'


import regression_game as div
import parameters as pr





# Essentially the same as player 2 for directly observing a noisy verson of true parameters


# -- NEW -- # estimates the cov

def get_dLogL(num_params):
    x = T.dvector('x')
    theta = T.dvector('theta')    
    cov_hat = T.dmatrix('cov_hat')

    # Define the logL function
    logL_enum = T.dot(T.transpose(x - theta), T.nlinalg.matrix_inverse(cov_hat))
    logL_enum = T.dot(logL_enum, x - theta) / -2
    logL_enum = T.exp(logL_enum)
    logL_denom = T.sqrt(T.pow((2 * np.pi), num_params) * T.nlinalg.det(cov_hat)) 

    logL = T.log(logL_enum/logL_denom)

    # The dlog likelihood function
    dlogL = theano.function([x, theta, cov_hat], T.grad(logL, theta))

    return dlogL

def estimate_FI(player_data, estimated_param, num_params):

    [data_x4] = player_data
    p4_cov_hat = np.cov(np.concatenate([sample for sample in data_x4[0]] ), rowvar=False)
    emp_Fisher = np.zeros((num_params, num_params))
    
    dlogL = get_dLogL(num_params)
    for data_point_x in data_x4[0][0]:
        sample_dlogL = dlogL(data_point_x, estimated_param, p4_cov_hat)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher = emp_Fisher / len(data_x4[0][0])

    return emp_Fisher


### Sample KL divergences from the player ###
def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov, num_params,
                          generate_fcn=None):

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
            sample_x = generate_fcn(sample_size)
            pmData_x = pm.Data('pmData_x', sample_x)

            # Specify the observed variables)
            
            p4_x_cov = np.cov(np.vstack([sample for sample in sample_x]), rowvar=False)
            p4_x_cov = np.diag(np.diag(p4_x_cov))
            
            pm_x = pm.MvNormal('pm_x', mu=pmTheta, cov=p4_x_cov, observed=pmData_x)
            
            for j in range(num_samples):
                # Show progress
                print('player 4 progress: {}/{}'.format(
                    progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                sample_x = generate_fcn(sample_size)
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
