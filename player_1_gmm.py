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

'''
# -- NEW -- # with estimated cov
# Input variables
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
'''



# compute dlogL for each class independently

# value = T.dmatrix('value') # observations:  n * d, d is the latent dimension
# mu_opt = T.dmatrix('mu_opt') # mu estimates: d
# tau_opt = T.dmatrix('tau_opt') # mu estimates: d * d

# k = tau_opt.shape[0]

# deltas, updates = theano.scan(lambda v: v - mu_opt, sequences = value)

# logL = T.sum((-1 / 2.0) * (k * T.log(2 * np.pi)
#                      + T.log(1.0 / det(tau_opt))
#                      + (deltas.dot(tau_opt) * deltas).sum(axis=1)
#                     )
#             )

# dlogL = theano.function([mu_opt, value, tau_opt], T.grad(logL, mu_opt))


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

'''
from pymc3.math import logsumexp

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]

    # tau is the inverse of the Sigma estimate/covariance
    delta = lambda mu: value - mu
    return (-1 / 2.0) * (
        k * T.log(2 * np.pi)
        + T.log(1.0 / det(tau))
        + (delta(mu).dot(tau) * delta(mu)).sum(axis=1)
    )

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau):
    # assuming all normal distributions only differ in mean, but not in Sigma or tau, i.e., cov
    def logp_(value):
        logps = [T.log(pi[i]) + logp_normal(mu, tau, value) for i, mu in enumerate(mus)]

        return T.sum(logsumexp(T.stacklists(logps)[:, :n_samples], axis=0))

    return logp_


# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, taus):
    # assuming all normal distributions have a distinct mean, and cov: tau or Sigma 
    def logp_(value):
        logps = [T.log(pi[i]) + logp_normal(mu, tau, value) for i, (mu, tau) in enumerate(zip(mus, taus))]

        return T.sum(logsumexp(T.stacklists(logps)[:, :n_samples], axis=0))

    return logp_
'''




def estimate_fisher_information(sample_size):
    # Generate some samples
    sample_theta = generate(sample_size)
    
    # Estimate the Fisher information
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        theta_hat = sample_theta[i]
        cov_hat = None

        sample_dlogL = dlogL(None , true_param)
        

        sample_dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher = emp_Fisher / sample_size
    return emp_Fisher



"""Sample KL divergences from player 1"""
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
            
            # packed_L = [pm.LKJCholeskyCov('packed_L_%d' % c, n=latent_dim, eta=2., 
                                          # sd_dist=pm.HalfCauchy.dist(1))
                        # for c in range(num_classes)]
            
            # L = [pm.expand_packed_triangular(latent_dim, packed_L[c])
                 # for c in range(num_classes)]
            
            # sigma = [pm.Deterministic('sigma_%d' % c ,L[c].dot(L[c].T))
                     # for c in range(num_classes)]
            # tau = [pm.Deterministic('tau_%d' % c,matrix_inverse(sigma[c]))
                   # for c in range(num_classes)]

            # mvnl = [pm.MvNormal.dist(mu=mus[c],chol=L[c]) for c in range(num_classes)]


            mvnl = [pm.MvNormal.dist(mu=mus[c],cov = np.eye(latent_dim)) for c in range(num_classes)]

            
            p = pm.Dirichlet('p', a=np.array([1.]*num_classes))
            
            gmm_data = pm.Data('gmm_data', np.concatenate(sampled_vae_mus))

            gmm = pm.Mixture('GMM', w=p, comp_dists=mvnl, observed=gmm_data)


            '''

            # Specify the prior
            # pmTheta = pm.MvNormal('pmTheta', mu=prior_mean, cov=prior_cov, shape=(1, num_params))
            
            pmMus= [pm.MvNormal('pmMu_'+ str(c), \
                mu=prior_mean, cov=prior_cov, shape=(1, latent_dim)) for c in range(num_classes)]

            # Data holders
            
            sampled_vae_mus, sampled_vae_logvars = generate_fcn(sample_size)
            # sample_means, sample_logvars = generate_fcn(sample_size)

            pmMus_data = [pm.Data('pmData_Mu_'+ str(c), sampled_vae_mus[c]) for c in range(num_classes) ]

            pm_samples = []
            # Specify the observed variables)
            for c in range(num_classes):
                
                class_cov = np.diag(np.exp(sampled_vae_logvars[c].mean(axis=0)))
                
                pm_sample = pm.MvNormal('pm_sample_'+str(c), mu=pmMus[c], cov=class_cov,\
                    observed=sampled_vae_mus[c])

                pm_samples.append(pm_sample)

            '''    
            
            for j in range(num_samples):
                # Show progress
                print('player 1 progress: {}/{}'.format(progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                # sample_means, sample_logvars = generate_fcn(sample_size)
                sampled_vae_mus, sampled_vae_logvars = generate_fcn(sample_size)

                print("player 1 sampled counts per class: ", [len(sampled_vae_mus[c]) for c in range(num_classes) ]) 
                print("player 1 sampled shape per class: ", [sampled_vae_mus[c].shape for c in range(num_classes) ]) 
                
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

                print('Player 1\'s prior and posterior shapes:', prior_mean.shape, prior_cov.shape, post_mean.shape, post_cov.shape)

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

