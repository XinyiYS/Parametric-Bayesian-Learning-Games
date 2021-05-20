import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
theano.config.compute_test_value = 'off'
import pymc3 as pm

import regression_game as div
import fourier as fr

import parameters as pr

# Player 1's data generation model
x_mean = 0
x_std_dev = 2.25
x_var = x_std_dev * x_std_dev
domain = [-10, 10]
noise_std_dev = 1
noise_var = noise_std_dev * noise_std_dev

# Initialize the data generation parameters & function
num_params = pr.num_params
true_param = pr.true_param
true_function = fr.Fourier(domain, num_params-1)
true_function.set_coefficients(true_param)

# Generate data from player 1
def generate(sample_size):
    # player 1's input is normally distributed
    x = np.random.normal(loc=x_mean, scale=x_std_dev, size=sample_size)
    # generate the output
    y = true_function.evaluate(x)
    y = y + np.random.normal(loc=0.0, scale=noise_std_dev, size=sample_size)    
    return x,y

def generate_and_plot(sample_size, ylim):
    x,y = generate(sample_size)
    plt.xlim(domain)
    plt.ylim(ylim)
    plt.plot(x,y, marker='o', linestyle='none', color='red', fillstyle='none')
    plt.savefig('player_1_dist.pdf')
    plt.show()
    plt.clf()
    return x,y


'''
"""Define the dlogL function"""    
# Input variables
x = T.dscalar('x')
y = T.dscalar('y')
theta = T.dvector('theta')

# Compute the mean of the density
L = domain[1] - domain[0]
freq_cos = 1
freq_sin = 1

basis_values = []
for i in range(num_params):        
    # Compute the basis values
    value = 1
    if i>0:
        if ((i % 2) == 1):
            # Odd
            value = T.cos(2 * freq_cos * np.pi * x / L)                
            freq_cos = freq_cos + 1
        else:
            # Even
            value = T.sin(2 * freq_sin * np.pi * x / L)                
            freq_sin = freq_sin + 1
    # Multiply by its coefficient
    basis_values.append(value)        
# Sum all values
mean = T.dot(theta, basis_values)

# Log likelihood (normal density)
logL = 1 / T.sqrt(2 * np.pi * noise_var)
term = (y - mean) ** 2
term = term / noise_var
term = term * (-0.5)
term = T.exp(term)
logL = logL * term
logL = T.log(logL)

# The dlog likelihood function
dlogL = theano.function([x, y, theta], T.grad(logL, theta))
'''



# https://jwmi.github.io/SL/11-Penalized-regression.pdf

# https://medium.com/quick-code/maximum-likelihood-estimation-for-regression-65f9c99f815d
data_cov = np.diag(np.full(num_params, 2.5))


# from player 2
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


def estimate_fisher_information(sample_size):
    # Generate some samples
    sample_x, sample_y = generate(sample_size)
    
    # Estimate the Fisher information
    emp_Fisher = np.zeros((num_params, num_params))
    for i in range(sample_size):
        sample_dlogL = dlogL(sample_x[i], sample_y[i], true_param)
        sample_dlogL.shape = (num_params, 1)
        emp_Fisher += np.matmul(sample_dlogL, np.transpose(sample_dlogL))
    emp_Fisher = emp_Fisher / sample_size
    return emp_Fisher

"""Sample KL divergences from player 1"""
# Function for computing the mean of output given an input and a parameter
# The same for all players
def compute_mean(x, theta):
    mu_y = 0
    L = domain[1] - domain[0]
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
                          prior_mean, prior_cov, generate_fcn=None):
    
    if generate_fcn is None:
        generate_fcn = generate

    # Computed outputs
    estimated_kl_values = []
    
    # Store the generated data
    generated_data_x = []
    generated_data_y = []
    generated_data_theta = []

    # For tracking progress
    progress = 1
    
    # For the sample range
    for sample_size in sample_size_range:
        
        # The estimated divergences for a sample size
        estimated_kl = []
        # For storing data with the same sample_size
        # data_x = []
        # data_y = []
        data_theta = []
        
        # Build the pymc3 model
        with pm.Model() as model:
            
            # Specify the prior
            pmTheta = pm.MvNormal('pmTheta', mu=prior_mean, cov=prior_cov, shape=(1, num_params))
            
            # Data holders
            sample_theta = generate_fcn(sample_size)

            pmData_theta = pm.Data('pmData_theta', sample_theta)

                    
            # Specify the observed variables
            
            # use an estimated cov

            p1_theta_cov = np.cov(np.vstack([sample for sample in sample_theta]), rowvar=False)
            p1_theta_cov = np.diag(np.diag(p1_theta_cov))
            p1_theta = pm.MvNormal('p1_theta', mu=pmTheta, cov=p1_theta_cov, observed=pmData_theta)
            
            # use a predetermined cov
            # p1_theta = pm.MvNormal('p1_theta', mu=pmTheta, cov=data_cov, observed=pmData_theta)

            
            for j in range(num_samples):
                # Show progress
                print('player 1 progress: {}/{}'.format(progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                sample_theta = generate_fcn(sample_size)
                pm.set_data({'pmData_theta': sample_theta})


                # sample_x, sample_y = generate_fcn(sample_size)
                # pm.set_data({'pmData_x': sample_x})
                # pm.set_data({'pmData_y': sample_y})
                
                # Plot the data against the true function
                """
                plt.xlim(domain)
                x = np.arange(domain[0], domain[1], 0.1)
                plt.plot(x, true_function.evaluate(x), linestyle='-', color='black', alpha=0.5)
                plt.plot(sample_x, sample_y, marker='o', fillstyle='none', color='red',
                         linestyle='none')
                plt.savefig('example_data_p1.pdf')
                plt.show()
                plt.clf()
                """
                
                # Store the data
                data_theta.append(sample_theta)
                
                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=0)
                summary = pm.stats.summary(pmTrace)
                
                # Get the sample mean and covariance of the samples
                post_mean = np.array(summary.loc[:,'mean'])
                post_cov = pm.trace_cov(pmTrace)
                
                # Plot some sample functions
                """
                x = np.arange(domain[0], domain[1], 0.1)
                fn = fr.Fourier(domain, num_params-1)
                fn.set_coefficients(post_mean)
                y = fn.evaluate(x)
                plt.plot(x, y, color='black')
                for k in range(10):
                    fn.set_coefficients(pmTrace['pmTheta'][k][0])
                    y = fn.evaluate(x)
                    plt.plot(x,y, color='black', alpha=0.3)
                plt.plot(sample_x, sample_y, linestyle='none', marker='o', color='red', fillstyle='none')
                plt.savefig('example_posterior_p1.pdf')
                plt.show()
                plt.clf()
                """
                
                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                estimated_kl.append(div.compute_KL(post_mean, post_cov, 
                                                   prior_mean, prior_cov))
                
                # Update progress
                progress += 1
    
        estimated_kl_values.append(estimated_kl)
        
        generated_data_theta.append(data_theta)
    
    return estimated_kl_values, generated_data_theta



'''

# original definition for observing two random variables X, Y or A, B
def sample_kl_divergences(sample_size_range, num_samples, num_draws,
                          prior_mean, prior_cov):
    

    # Computed outputs
    estimated_kl_values = []
    
    # Store the generated data
    generated_data_x = []
    generated_data_y = []
    
    # For tracking progress
    progress = 1
    
    # For the sample range
    for sample_size in sample_size_range:
        
        # The estimated divergences for a sample size
        estimated_kl = []
        # For storing data with the same sample_size
        data_x = []
        data_y = []
        
        # Build the pymc3 model
        with pm.Model() as model:
            
            # Specify the prior
            pmTheta = pm.MvNormal('pmTheta', mu=prior_mean, cov=prior_cov, 
                                  shape=(1, num_params))
            
            # Data holders
            sample_x, sample_y = generate(sample_size)
            print('player 1 sample -------------')
            print('sample_x, sample_y:', sample_x.shape, sample_y.shape)
            print('-------------')

            pmData_x = pm.Data('pmData_x', sample_x)
            pmData_y = pm.Data('pmData_y', sample_y)
                    
            # Specify the observed variables
            pm_x = pm.Normal('pm_x', mu=x_mean, sigma=x_std_dev, observed=pmData_x)
            pm_y = pm.Normal('pm_y', mu=compute_mean(pm_x, pmTheta), 
                             sigma=noise_std_dev, observed=pmData_y)
            
            for j in range(num_samples):
                # Show progress
                print('player 1 progress: {}/{}'.format(progress, len(sample_size_range) * num_samples))
                
                # Generate the data
                sample_x, sample_y = generate(sample_size)
                pm.set_data({'pmData_x': sample_x})
                pm.set_data({'pmData_y': sample_y})
                
                # Plot the data against the true function
                """
                plt.xlim(domain)
                x = np.arange(domain[0], domain[1], 0.1)
                plt.plot(x, true_function.evaluate(x), linestyle='-', color='black', alpha=0.5)
                plt.plot(sample_x, sample_y, marker='o', fillstyle='none', color='red',
                         linestyle='none')
                plt.savefig('example_data_p1.pdf')
                plt.show()
                plt.clf()
                """
                
                # Store the data
                data_x.append(sample_x)
                data_y.append(sample_y)
                
                # Sample from the posterior
                pmTrace = pm.sample(draws=num_draws, 
                                    cores=pr.max_num_cores, 
                                    tune=pr.tuning_step, 
                                    progressbar=0)
                summary = pm.stats.summary(pmTrace)
                
                # Get the sample mean and covariance of the samples
                post_mean = np.array(summary.loc[:,'mean'])
                post_cov = pm.trace_cov(pmTrace)
                
                # Plot some sample functions
                """
                x = np.arange(domain[0], domain[1], 0.1)
                fn = fr.Fourier(domain, num_params-1)
                fn.set_coefficients(post_mean)
                y = fn.evaluate(x)
                plt.plot(x, y, color='black')
                for k in range(10):
                    fn.set_coefficients(pmTrace['pmTheta'][k][0])
                    y = fn.evaluate(x)
                    plt.plot(x,y, color='black', alpha=0.3)
                plt.plot(sample_x, sample_y, linestyle='none', marker='o', color='red', fillstyle='none')
                plt.savefig('example_posterior_p1.pdf')
                plt.show()
                plt.clf()
                """
                
                # Assuming Normal distribution for the posterior,
                # estimate the KL divergence
                estimated_kl.append(div.compute_KL(post_mean, post_cov, 
                                                   prior_mean, prior_cov))
                
                # Update progress
                progress += 1
    
        estimated_kl_values.append(estimated_kl)
        
        generated_data_x.append(data_x)
        generated_data_y.append(data_y)
    
    return estimated_kl_values, generated_data_x, generated_data_y


'''