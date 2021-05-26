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


def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov,
                          sampled_vae_mus_1, sampled_vae_logvars_1,
                          sampled_vae_mus_2, sampled_vae_logvars_2):  
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

            sampled_vae_mus_1_all, sampled_vae_mus_2_all = [], []

            for sampled_vae_mus_1_digit in sampled_vae_mus_1[i][0]:
                if len(sampled_vae_mus_1_digit) > 0:
                    sampled_vae_mus_1_all.extend(list(np.vstack(sampled_vae_mus_1_digit)))

            for sampled_vae_mus_2_digit in sampled_vae_mus_1[i][0]:
                if len(sampled_vae_mus_2_digit) > 0:
                    sampled_vae_mus_2_all.extend(list(np.vstack(sampled_vae_mus_2_digit)))

            gmm_data_1 = pm.Data('gmm_data_1', sampled_vae_mus_1_all[0])
            gmm_data_2 = pm.Data('gmm_data_2', sampled_vae_mus_1_all[0])


            gmm1 = pm.Mixture('GMM1', w=p, comp_dists=mvnl, observed=gmm_data_1)
            gmm2 = pm.Mixture('GMM2', w=p, comp_dists=mvnl, observed=gmm_data_2)

            for j in range(num_samples):

                print('player 1+2 progress: {}/{}'.format(
                    progress, len(sample_size_range) * num_samples))
                
                # Assign the data
                pm.set_data({'gmm_data_1': sampled_vae_mus_1_all[j]})
                pm.set_data({'gmm_data_2': sampled_vae_mus_2_all[j]})
                
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
                post_covs  = [ post_cov[k * M:(k+1  )*M, k * M: ( k + 1) * M] for k in range(num_classes)   ]
 

                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                # estimated_kl.append(div.compute_KL(post_mean, post_cov, # prior_mean, prior_cov))

                kl = sum([div.compute_KL(post_mean, post_cov, prior_mean, prior_cov) for post_mean, post_cov in zip(post_means, post_covs)  ])
                estimated_kl.append(kl)
                
                # Update progress
                progress += 1


        estimated_kl_values.append(estimated_kl)  
    
    return estimated_kl_values, post_mean, post_cov, summary