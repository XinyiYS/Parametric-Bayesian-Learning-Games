

# This is exactly the same as the original player_12.py
# repurposed to test out the multi-player scenario for 3 players

# TODO: modify to suit player 23


import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

import player_1_multi as p1
import player_2_multi as p2
import player_3_multi as p3
import regression_game as div
import fourier as fr

# Copy the global parameters
import parameters as pr
num_params = pr.num_params
true_param = pr.true_param

def estimate_fisher_information(sample_size):
    
    sample_x3, sample_y3 = p3.generate(sample_size)
    sample_x2 = p2.generate(sample_size)
    
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        
        dlogL3 = p3.dlogL(sample_x3[i], sample_y3[i], true_param)
        dlogL2 = p2.dlogL(sample_x2[i], true_param)
        dlogL = dlogL3 + dlogL2
        
        dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(dlogL, np.transpose(dlogL))
    emp_Fisher = emp_Fisher / sample_size
    return emp_Fisher


### Sample KL divergences ###

def _set_data_holder_0(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_x_1, data_y_1] = player_data

    pmData_x_1 = pm.Data('pmData_x_1', data_x_1[i][0])
    pmData_y_1 = pm.Data('pmData_y_1', data_y_1[i][0])


    pm_x_1 = pm.Normal('pm_x_1', mu=p1.x_mean, sigma=p1.x_std_dev, observed=pmData_x_1)
    pm_y_1 = pm.Normal('pm_y_1', mu=p1.compute_mean(pm_x_1, pmTheta), 
                     sigma=p1.noise_std_dev, observed=pmData_y_1)

    return [data_x_1, data_y_1], ['pmData_x_1', 'pmData_y_1']



def _set_data_holder_1(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_x_2] = player_data

    pmData_x_2 = pm.Data('pmData_x_2', data_x_2[i][0])

    pm_x_2 = pm.MvNormal('pm_x_2', mu=pmTheta, cov=p2.data_cov, observed=pmData_x_2)

    return [data_x_2], ['pmData_x_2']


def _set_data_holder_2(i, parameter_variable, player_data):
    pmTheta = parameter_variable
    [data_x_3, data_y_3, noise_std_estimate] = player_data

    pmData_x_3 = pm.Data('pmData_x_3', data_x_3[i][0])
    pmData_y_3 = pm.Data('pmData_y_3', data_y_3[i][0])

    pm_x_3 = pm.Normal('pm_x_3', mu=p3.x_mean, sigma=p3.x_std_dev, observed=pmData_x_3)
    pm_y_3 = pm.Normal('pm_y_3', mu=p3.compute_mean(pm_x_3, pmTheta), 
                     sigma=noise_std_estimate, observed=pmData_y_3)

    return [data_x_3, data_y_3], ['pmData_x_3', 'pmData_y_3']


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


def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov,
                          player_index_data_dict):
                          # data_x_2,
                          # data_x_3, data_y_3, noise_std_estimate=p3.noise_std_prior):  
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

            '''
            pmData_x_2 = pm.Data('pmData_x_2', data_x_2[i][0])

            pmData_x_3 = pm.Data('pmData_x_3', data_x_3[i][0])
            pmData_y_3 = pm.Data('pmData_y_3', data_y_3[i][0])
            
            # Specify the observed variables
            

            # From player 2
            pm_x_2 = pm.MvNormal('pm_x_2', mu=pmTheta, cov=p2.data_cov, observed=pmData_x_2)
            

            # From player 3
            pm_x_3 = pm.Normal('pm_x_3', mu=p3.x_mean, sigma=p3.x_std_dev, observed=pmData_x_3)
            pm_y_3 = pm.Normal('pm_y_3', mu=compute_mean(pm_x_3, pmTheta), 
                             sigma=noise_std_estimate, observed=pmData_y_3)
            '''
            
            for j in range(num_samples):
                # Show progress
                print('player {} progress: {}/{}'.format('+'.join([str(index) for index in player_index_data_dict.keys()]),
                    progress, len(sample_size_range) * num_samples))
                

                for (observed_datas, data_holder_names) in data_holders_list:
                    for observed_data, data_holder_name in zip(observed_datas, data_holder_names):
                        pm.set_data({data_holder_name: observed_data[i][j]})


                # Assign the data
                '''
                pm.set_data({'pmData_x_2': data_x_2[i][j]})


                pm.set_data({'pmData_x_3': data_x_3[i][j]})
                pm.set_data({'pmData_y_3': data_y_3[i][j]})
                '''


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
