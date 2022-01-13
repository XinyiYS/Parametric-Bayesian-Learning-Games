import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

import player_1 as p1
import player_2 as p2
import regression_game as div
import fourier as fr

# Copy the global parameters
import parameters_MNIST as pr
num_params = pr.num_params
true_param = pr.true_param
latent_dim = pr.latent_dim

### Sample KL divergences from player 1+2 ###
# Function for computing the mean of output given an input and a parameter

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
   
    [mvnl, p] = parameter_variable
    [sampled_vae_mus, _] = player_data

    sampled_vae_mus_all = []
    for sampled_vae_mus_digit in sampled_vae_mus[i][0]:
        if len(sampled_vae_mus_digit) > 0:
            sampled_vae_mus_all.extend(list(np.vstack(sampled_vae_mus_digit)))

    gmm_data_1 = pm.Data('gmm_data_1', sampled_vae_mus_all[0])

    gmm1 = pm.Mixture('GMM1', w=p, comp_dists=mvnl, observed=gmm_data_1)

    return [sampled_vae_mus_all], ['gmm_data_1']



def _set_data_holder_1(i, parameter_variable, player_data):
   
    [mvnl, p] = parameter_variable
    [sampled_vae_mus, _] = player_data
    sampled_vae_mus_all = []

    for sampled_vae_mus_digit in sampled_vae_mus[i][0]:
        if len(sampled_vae_mus_digit) > 0:
            sampled_vae_mus_all.extend(list(np.vstack(sampled_vae_mus_digit)))

    gmm_data_2 = pm.Data('gmm_data_2', sampled_vae_mus_all[0])

    gmm2 = pm.Mixture('GMM2', w=p, comp_dists=mvnl, observed=gmm_data_2)

    return [sampled_vae_mus_all], ['gmm_data_2']



def _set_data_holder_2(i, parameter_variable, player_data):
   
    [mvnl, p] = parameter_variable
    [sampled_vae_mus, _] = player_data
    sampled_vae_mus_all = []

    for sampled_vae_mus_digit in sampled_vae_mus[i][0]:
        if len(sampled_vae_mus_digit) > 0:
            sampled_vae_mus_all.extend(list(np.vstack(sampled_vae_mus_digit)))

    gmm_data_3 = pm.Data('gmm_data_3', sampled_vae_mus_all[0])

    gmm3 = pm.Mixture('GMM3', w=p, comp_dists=mvnl, observed=gmm_data_3)

    return [sampled_vae_mus_all], ['gmm_data_3']


def _set_data_holder_3(i, parameter_variable, player_data):
   
    [mvnl, p] = parameter_variable
    [sampled_vae_mus, _] = player_data
    sampled_vae_mus_all = []

    for sampled_vae_mus_digit in sampled_vae_mus[i][0]:
        if len(sampled_vae_mus_digit) > 0:
            sampled_vae_mus_all.extend(list(np.vstack(sampled_vae_mus_digit)))

    gmm_data_4 = pm.Data('gmm_data_4', sampled_vae_mus_all[0])

    gmm4 = pm.Mixture('GMM4', w=p, comp_dists=mvnl, observed=gmm_data_4)

    return [sampled_vae_mus_all], ['gmm_data_4']


def _set_data_holder_4(i, parameter_variable, player_data):
   
    [mvnl, p] = parameter_variable
    [sampled_vae_mus, _] = player_data
    sampled_vae_mus_all = []

    for sampled_vae_mus_digit in sampled_vae_mus[i][0]:
        if len(sampled_vae_mus_digit) > 0:
            sampled_vae_mus_all.extend(list(np.vstack(sampled_vae_mus_digit)))

    gmm_data_5 = pm.Data('gmm_data_5', sampled_vae_mus_all[0])

    gmm5 = pm.Mixture('GMM5', w=p, comp_dists=mvnl, observed=gmm_data_5)

    return [sampled_vae_mus_all], ['gmm_data_5']


def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov,
                          player_index_data_dict,):
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
            mus = [pm.MvNormal('mu_%d' % c,
                             mu=pm.floatX(np.zeros(latent_dim)),
                             tau=pm.floatX(0.1 * np.eye(latent_dim)),
                             shape=(latent_dim,),
                             )
                 for c in range(num_classes)]

            mvnl = [pm.MvNormal.dist(mu=mus[c],cov = np.eye(latent_dim)) for c in range(num_classes)]

            p = pm.Dirichlet('p', a=np.array([1.]*num_classes))



            # gmm_data_1 = pm.Data('gmm_data_1', sampled_vae_mus_1_all[0])
            # gmm_data_2 = pm.Data('gmm_data_2', sampled_vae_mus_1_all[0])

            # gmm1 = pm.Mixture('GMM1', w=p, comp_dists=mvnl, observed=gmm_data_1)
            # gmm2 = pm.Mixture('GMM2', w=p, comp_dists=mvnl, observed=gmm_data_2)

            data_holders_list = [] # a list (length = number of players) of list (each corresponding to the number of holders the player needs)

            variables_list = [] # a list (length = number of players) of list (each corresponding to the number of variables the player needs)

            for player_index, player_data in player_index_data_dict.items():

                # a list of data holders, since some player index may need more than one
                # a list of data holders' variable names in pymc3 model, later used for setting the data
                observed_datas, data_holder_names = set_data_holder(player_index, i, [mvnl, p], player_data)

                data_holders_list.append([observed_datas, data_holder_names])


            for j in range(num_samples):

                print('player {} progress: {}/{}'.format('+'.join([str(index) for index in player_index_data_dict.keys()]),
                    progress, len(sample_size_range) * num_samples))
                    
                # Assign the data
                # pm.set_data({'gmm_data_1': sampled_vae_mus_1_all[j]})
                # pm.set_data({'gmm_data_2': sampled_vae_mus_2_all[j]})
                
                for (observed_datas, data_holder_names) in data_holders_list:
                    for observed_data, data_holder_name in zip(observed_datas, data_holder_names):
                        pm.set_data({data_holder_name: observed_data[j]})


                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=False)
                summary = pm.stats.summary(pmTrace)
                
                # Get the sample mean and covariance of the samples
                post_mean = np.array(summary.loc[:,'mean'])
                post_cov = pm.trace_cov(pmTrace)

                post_means = post_mean[:num_classes*latent_dim].reshape(num_classes, latent_dim)

                M = latent_dim
                post_covs  = [ post_cov[k * M:(k+1) * M, k * M: ( k + 1) * M] for k in range(num_classes)   ]
 

                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                # estimated_kl.append(div.compute_KL(post_mean, post_cov, # prior_mean, prior_cov))

                kl = sum([div.compute_KL(post_mean, post_cov, prior_mean, prior_cov) for post_mean, post_cov in zip(post_means, post_covs)])
                estimated_kl.append(kl)
                
                # Update progress
                progress += 1


        estimated_kl_values.append(estimated_kl)  
    
    return estimated_kl_values, post_mean, post_cov, summary