import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nlinalg import det
theano.config.compute_test_value = 'off'
# theano.config.compute_test_value = "ignore"

import pymc3 as pm

import regression_game as div

import parameters_MNIST as pr


# Initialize the data generation parameters & function
num_params = pr.num_params
true_param = pr.true_param

num_classes = pr.num_classes
latent_dim = pr.latent_dim

def get_dLogL():
    value = T.dmatrix('value') # observations:  n * d, d is the latent dimension
    mu_opt = T.dmatrix('mu_opt') # mu estimates: d
    tau_opt = T.dmatrix('tau_opt') # mu estimates: d * d

    k = tau_opt.shape[0]

    deltas, updates = theano.scan(lambda v: v - mu_opt, sequences = value)

    logL = T.sum((-1 / 2.0) * (k * T.log(2 * np.pi)
                         + T.log(1.0 / det(tau_opt))
                         + (deltas.dot(tau_opt) * deltas).sum(axis=1)
                        )
                )

    dlogL = theano.function([mu_opt, value, tau_opt], T.grad(logL, mu_opt))
    return dlogL

def estimate_FI(player_data, estimated_param):

    [estimated_means, estimated_covs] = estimated_param
    [sampled_vae_mus, sampled_vae_logvars] = player_data

    emp_Fishers = [np.zeros((latent_dim, latent_dim))  for c in range(num_classes)]

    dlogL = get_dLogL()

    for c in range(num_classes):    
        if len(sampled_vae_mus[0][0][c]) == 0: continue

        sample_dlogL = dlogL(estimated_means[c].reshape(1, latent_dim), sampled_vae_mus[0][0][c], estimated_covs[c])

        sample_dlogL.shape = (latent_dim, 1)
        
        emp_Fishers[c] += np.matmul(sample_dlogL, np.transpose(sample_dlogL))

        emp_Fishers[c] = emp_Fishers[c] / len(sampled_vae_logvars[0][0][c])

    return emp_Fishers




"""Sample KL divergences from player 4"""
# Function for computing the mean of output given an input and a parameter
# The same for all players
from theano.tensor.nlinalg import matrix_inverse

testvals = [[-2,-2],[0,0],[2,2], [1,1] , [-1,2] , [-2,-2],[0,0],[2,2], [1,1], [-1,2]]

def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov, generate_fcn=None):
    
    if generate_fcn is None:
        generate_fcn = generate

    # Computed outputs
    estimated_kl_values = []
    
    # Store the generated data

    total_sampled_vae_mus = []
    total_sampled_vae_logars = []


    # For tracking progress
    progress = 1
    
    # For the sample range
    for sample_size in sample_size_range:
        
        # The estimated divergences for a sample size
        estimated_kl = []
        # For storing data with the same sample_size
        data_theta = []

        data_sampled_mus = []
        data_sampled_logvars = []
        
        # Build the pymc3 model
        with pm.Model() as model:
            
            sampled_vae_mus, sampled_vae_logvars = generate_fcn(sample_size)

            mus = [pm.MvNormal('mu_%d' % c,
                             mu=pm.floatX(np.zeros(latent_dim)),
                             tau=pm.floatX(0.1 * np.eye(latent_dim)),
                             shape=(latent_dim,),
                             testval=pm.floatX(testvals[c]))
                 for c in range(num_classes)]
            

            mvnl = [pm.MvNormal.dist(mu=mus[c],cov = np.eye(latent_dim)) for c in range(num_classes)]

            
            p = pm.Dirichlet('p', a=np.array([1.]*num_classes))
            
            gmm_data = pm.Data('gmm_data', np.concatenate(sampled_vae_mus))

            gmm = pm.Mixture('GMM', w=p, comp_dists=mvnl, observed=gmm_data)


            
            for j in range(num_samples):
                # Show progress
                print('player 4 progress: {}/{}'.format(progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                # sample_means, sample_logvars = generate_fcn(sample_size)
                sampled_vae_mus, sampled_vae_logvars = generate_fcn(sample_size)

                print("player 4 sampled counts per class: ", [len(sampled_vae_mus[c]) for c in range(num_classes) ]) 
                print("player 4 sampled shape per class: ", [sampled_vae_mus[c].shape for c in range(num_classes) ]) 
                
                # for c in range(num_classes):
                    # pm.set_data({'pmData_Mu_'+ str(c): sampled_vae_mus[c]})
                  
                pm.set_data({'gmm_data': np.concatenate(sampled_vae_mus)})

                # Store the data

                data_sampled_mus.append(sampled_vae_mus)
                data_sampled_logvars.append(sampled_vae_logvars)


                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=0)
                summary = pm.stats.summary(pmTrace)
                
                # Get the sample mean and covariance of the samples

                post_mean = np.array(summary.loc[:, 'mean'])
                post_cov = pm.trace_cov(pmTrace)

                print('Player 4\'s prior and posterior shapes:', prior_mean.shape, prior_cov.shape, post_mean.shape, post_cov.shape)

                post_means = post_mean[:num_classes*latent_dim].reshape(num_classes, latent_dim)

                M = latent_dim
                post_covs  = [ post_cov[k * M:(k+1  )*M, k * M: ( k + 1) * M] for k in range(num_classes)   ]

                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                # post_covs = [ post_cov[k*M:(k+1)*M, k*M:(k+1)*M] for k in range(post_cov.shape[0] // M)]                
                # post_means = post_mean.reshape(num_classes, latent_dim)

                kl = sum([div.compute_KL(post_mean, post_cov, prior_mean, prior_cov) for post_mean, post_cov in zip(post_means, post_covs)  ])
                estimated_kl.append(kl)
                
                # Update progress
                progress += 1
    
        estimated_kl_values.append(estimated_kl)
        

        total_sampled_vae_mus.append(data_sampled_mus)
        total_sampled_vae_logars.append(data_sampled_logvars)


    return estimated_kl_values, total_sampled_vae_mus, total_sampled_vae_logars

