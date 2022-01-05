import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

import player_1 as p1
import player_2 as p2
import regression_game as div
import fourier as fr


import theano
import theano.tensor as T
theano.config.compute_test_value = 'ignore'

# Copy the global parameters
import parameters as pr
num_params = pr.num_params
true_param = pr.true_param

def estimate_fisher_information(sample_size, p1_generate_fcn=None, p2_generate_fcn=None):

    if p1_generate_fcn is not None:

        sample_x1, sample_y1 = p1_generate_fcn.generate(sample_size)
    else:
        sample_x1, sample_y1 = p1.generate(sample_size)


    if p2_generate_fcn is not None:
        sample_x2 = p2_generate_fcn(sample_size)
    else:
        sample_x2 = p2.generate(sample_size)
    
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        
        dlogL1 = p1.dlogL(sample_x1[i], sample_y1[i], true_param)
        dlogL2 = p2.dlogL(sample_x2[i], true_param)
        dlogL = dlogL1 + dlogL2
        
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



def get_theta_hat(x, y, reg_lambda=1e-3):
    return np.linalg.inv(x.T @ x + reg_lambda * np.identity(x.shape[1])) @ x.T @ y  


def set_data_holder(player_index, sample_index_i, parameter_variable, player_data):

    if player_index == 0:
        return _set_data_holder_0(sample_index_i, parameter_variable, player_data)
    
    elif player_index == 1:
        return _set_data_holder_1(sample_index_i, parameter_variable, player_data)
    
    elif player_index == 2:
        return _set_data_holder_2(sample_index_i, parameter_variable, player_data)
    
    elif player_index == 3:
        return _set_data_holder_3(sample_index_i, parameter_variable, player_data)

    elif player_index == 4:
        return _set_data_holder_4(sample_index_i, parameter_variable, player_data)

    else:
        raise NotImplementedError("Only up to 5 players.")


def _set_data_holder_0(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_theta_1] = player_data

    pmData_theta_1 = pm.Data('pmData_theta_1', data_theta_1[i][0])


    p1_theta_cov = np.cov(np.concatenate([sample for sample in data_theta_1[i]] ), rowvar=False)
    p1_theta_cov = np.diag(np.diag(p1_theta_cov))
    pmTheta_1 = pm.MvNormal('pmTheta_1', mu=pmTheta, cov=p1_theta_cov, observed=pmData_theta_1)

    return [data_theta_1], ['pmData_theta_1']



def _set_data_holder_1(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_x_2] = player_data

    pmData_x_2 = pm.Data('pmData_x_2', data_x_2[i][0])

    p2_x_cov = np.cov(np.vstack([sample for sample in data_x_2[i]]), rowvar=False)
    p2_x_cov = np.diag(np.diag(p2_x_cov))
    pm_x_2 = pm.MvNormal('pm_x_2', mu=pmTheta, cov=p2_x_cov, observed=pmData_x_2)

    return [data_x_2], ['pmData_x_2']


def _set_data_holder_2(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_theta_3] = player_data

    pmData_theta_3 = pm.Data('pmData_theta_3', data_theta_3[i][0])


    p3_theta_cov = np.cov(np.concatenate([sample for sample in data_theta_3[i]] ), rowvar=False)
    p3_theta_cov = np.diag(np.diag(p3_theta_cov))
    pmTheta_3 = pm.MvNormal('pmTheta_3', mu=pmTheta, cov=p3_theta_cov, observed=pmData_theta_3)

    return [data_theta_3], ['pmData_theta_3']


def _set_data_holder_3(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_x_4] = player_data

    pmData_x_4 = pm.Data('pmData_x_4', data_x_4[i][0])

    p4_x_cov = np.cov(np.vstack([sample for sample in data_x_4[i]]), rowvar=False)
    p4_x_cov = np.diag(np.diag(p4_x_cov))
    pm_x_4 = pm.MvNormal('pm_x_4', mu=pmTheta, cov=p4_x_cov, observed=pmData_x_4)

    return [data_x_4], ['pmData_x_4']



def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov, num_params,
                          player_index_data_dict):  
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
            
            data_holders_list = [] # a list (length = number of players) of list (each corresponding to the number of holders the player needs)

            variables_list = [] # a list (length = number of players) of list (each corresponding to the number of variables the player needs)

            for player_index, player_data in player_index_data_dict.items():

                # a list of data holders, since some player index may need more than one
                # a list of data holders' variable names in pymc3 model, later used for setting the data
                observed_datas, data_holder_names = set_data_holder(player_index, i, pmTheta, player_data)

                data_holders_list.append([observed_datas, data_holder_names])


            for j in range(num_samples):
                # Show progress
                print('player {} progress: {}/{}'.format('+'.join([str(index) for index in player_index_data_dict.keys()]),
                    progress, len(sample_size_range) * num_samples))
                    
                # Assign the data
                for (observed_datas, data_holder_names) in data_holders_list:
                    for observed_data, data_holder_name in zip(observed_datas, data_holder_names):
                        pm.set_data({data_holder_name: observed_data[i][j]})

                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=False)
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
