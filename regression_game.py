import numpy as np
import matplotlib.pyplot as plt
import plotter as plotter


# Compute the Gaussian posterior from a Gaussian prior & likelihood
def compute_posterior(prior_mean, prior_covariance, X, y, noise_variance):
    inverted_prior_covariance = np.linalg.inv(prior_covariance)
    # Compute posterior's convariance
    posterior_covariance = np.matmul(np.multiply(np.matrix.transpose(X), noise_variance), X)
    posterior_covariance = inverted_prior_covariance + posterior_covariance
    
    # This is the matrix to be inverted
    # print(posterior_covariance)
    
    # Compute its Cholesky factor
    # cholesky = np.linalg.cholesky(posterior_covariance)
    #print(cholesky)
    
    # Compute the inverse of the Cholesky factor
    # cholesky_inv = np.linalg.inv(cholesky)
    #print(cholesky_inv)
    
    # Compute its inverse directly
    posterior_covariance = np.linalg.inv(posterior_covariance)
    
    # Compute posterior's covariance
    posterior_mean = np.multiply(np.matrix.transpose(X), noise_variance)
    posterior_mean = np.matmul(posterior_mean, y)
    posterior_mean = np.add(posterior_mean, np.matmul(inverted_prior_covariance, prior_mean))
    posterior_mean = np.matmul(posterior_covariance, posterior_mean)
    # Return the computed posterior
    return posterior_mean, posterior_covariance

# Compute KL divergence between two multi-variate Gaussian
# mean_0: posterior mean
# cov_0: posterior covariance
# mean_1: prior mean
# cov_1: prior covariance
def compute_KL(mean_0, cov_0, mean_1, cov_1):
    inverted_cov_1 = np.linalg.inv(cov_1)
    term = np.matmul(inverted_cov_1, cov_0)
    divergence = np.trace(term)
    term = np.subtract(mean_1, mean_0)
    term = np.transpose(term)
    term = np.matmul(term, inverted_cov_1)
    term = np.matmul(term, np.subtract(mean_1, mean_0))
    divergence = divergence + term
    divergence = divergence - len(mean_0)
    term = np.linalg.det(cov_1)
    term = term / np.linalg.det(cov_0)
    term = np.log(term)
    divergence = divergence + term
    divergence = divergence / 2
    divergence = divergence.flatten()[0]
    return divergence

# Compute the Renyi divergence between two multi-variate Gaussian
# mean_0: posterior mean
# cov_0: posterior covariance
# mean_1: prior mean
# cov_1: prior covariance
def compute_Renyi(mean_0, cov_0, mean_1, cov_1, alpha):
    sigma_star = np.multiply(alpha, cov_1) + np.multiply((1-alpha), cov_0)
    
    delta_mu = np.subtract(mean_0, mean_1)
    
    term_1 = np.matmul(np.transpose(delta_mu), sigma_star)
    term_1 = np.matmul(term_1, delta_mu)
    term_1 = term_1 * alpha / 2
    
    det_sigma_star = np.linalg.det(sigma_star)
    det_cov_0 = np.linalg.det(cov_0)
    det_cov_1 = np.linalg.det(cov_1)
    
    term_2 = np.power(det_cov_0, 1 - alpha)
    term_2 = term_2 * np.power(det_cov_1, alpha)
    term_2 = det_sigma_star / term_2
    term_2 = np.log(term_2)
    term_2 = term_2 / ( 2 * (alpha - 1))
    
    divergence = term_1 - term_2
    return divergence

#-- Check again, this might be wrong --#

# Compute the Hellinger distance between two multi-variate Normal
def compute_Hellinger(mean_0, cov_0, mean_1, cov_1):
    mean_diff = np.subtract(mean_0, mean_1)
    weight = np.add(cov_0, cov_1)
    weight = np.multiply(0.5, weight)
    weight = np.linalg.inv(weight)
    exponent = np.matmul(np.transpose(mean_diff), weight)
    exponent = np.matmul(exponent, mean_diff)
    exponent = -(1/8) * exponent
    exponent = np.exp(exponent)
    
    det_cov_0 = np.linalg.det(cov_0)
    det_cov_0 = np.power(det_cov_0, 0.25)
    det_cov_1 = np.linalg.det(cov_1)
    det_cov_1 = np.power(det_cov_1, 0.25)
    enumerator = det_cov_0 * det_cov_1
    
    denominator = np.add(cov_0, cov_1)
    denominator = np.multiply(0.5, denominator)
    denominator = np.linalg.det(denominator)
    denominator = np.sqrt(denominator)
    
    H2 = enumerator / denominator
    H2 = H2 * exponent
    H2 = 1 - H2
    
    return H2

#-----#

class RegressionGame:
    
    def __init__(self, data_generator, model, degree, prior, value_eval, order):
        self.data_generator = data_generator
        
        # Parameters to be analyzed
        self.model = model
        self.degree = degree
        self.prior = prior
        self.value_eval = value_eval
        self.order = order
        
        self.num_players = data_generator.get_num_players()
    
    def get_num_players(self):
        return self.num_players
    
    def compute_posterior(self, index):
        
        variance = self.data_generator.get_variance()
        domain = self.data_generator.get_domain()
        
        # Get the combined data of players chosen
        X_S, Y_S = self.data_generator.get_serialized_data(index)
        
        # Compute the posterior of S from prior
        X_S_reshaped = self.model.distribute(domain, self.degree, X_S)
        Y_S_reshaped = np.reshape(Y_S, (len(Y_S), 1))
        posterior_S_mean, posterior_S_covariance = compute_posterior(
                self.prior['mean'], self.prior['cov'], 
                X_S_reshaped, Y_S_reshaped, variance)
    
        return posterior_S_mean, posterior_S_covariance
    
    def compute_value(self, index, verbose=False):
        if self.value_eval == 'KL': return self.compute_value_Renyi(index, 1.0, verbose)
        if self.value_eval == 'Renyi': return self.compute_value_Renyi(index, self.order, verbose)
        exit(1)
    
    def compute_value_Renyi(self, index, order, verbose=False):
        
        #num_players = self.num_players    
        variance = self.data_generator.get_variance()
        domain = self.data_generator.get_domain()
        
        # Get the combined data of players chosen
        X_S, Y_S = self.data_generator.get_serialized_data(index)
        
        # Compute the posterior of S from prior
        X_S_reshaped = self.model.distribute(domain, self.degree, X_S)
        Y_S_reshaped = np.reshape(Y_S, (len(Y_S), 1))
        posterior_S_mean, posterior_S_covariance = compute_posterior(
                self.prior['mean'], self.prior['cov'], 
                X_S_reshaped, Y_S_reshaped, variance)
        
        # Compute the divergence of S from prior
        """
        KL_S = compute_KL(posterior_S_mean, posterior_S_covariance, 
                          self.prior['mean'], self.prior['cov'])
        """
        
        # Compute the Renyi divergence of S from prior
        divergence = 0
        
        if order == 1:
            divergence = compute_KL(posterior_S_mean, posterior_S_covariance, 
                                    self.prior['mean'], self.prior['cov'])
        else:
            divergence = compute_Renyi(posterior_S_mean, 
                                       posterior_S_covariance, 
                                       self.prior['mean'], 
                                       self.prior['cov'],
                                       order)
        
        # Try Hellinger distance instead
        # KL_S = compute_Hellinger(posterior_S_mean, posterior_S_covariance,
        #           self.prior['mean'], self.prior['cov'])
        
        """
        # Compute the KL using the second method
        Lambda = np.diag(np.ones(self.degree + 1))
        H = np.matmul(np.transpose(X_S_reshaped), X_S_reshaped)
        Lambda = np.add(Lambda, H)
        
        detLambda1 = np.linalg.det(Lambda)
        
        # Compute determinant via the eigenvalues
        eigvalH, eigvecH = np.linalg.eig(H)
        detLambda2 = np.prod(np.add(1, eigvalH))
        # Compare the two
        # print("detLambda1 = {:}, detLambda2 = {:}".format(detLambda1, detLambda2))
        
        print("{:} {:}".format(np.sum(eigvalH), np.trace(H)))
        
        term1 = np.log(detLambda1)
        term2 = np.trace(np.matmul(H, posterior_S_covariance))
        term3 = np.matmul(np.transpose(posterior_S_mean), posterior_S_mean)
        
        KL_S_2 = term1 - term2 + term3
        KL_S_2 = KL_S_2 / 2
        """
        
        # Check values
        if verbose:
            print ("V {:} = {:}".format(index, divergence))
        
        # Plot to verify
        if verbose:
            plt.axis([domain[0]-0.5, domain[1]+0.5, -13.0, 8.5])
            plotter.plot_sample_functions(posterior_S_mean, posterior_S_covariance, self.model, domain, 'black', plt)
            self.data_generator.plot_data(index, plt)
            plt.xlabel("Renyi({:}) = {:}".format(order, divergence)) 
            plt.savefig("{:}{:}{:}.pdf".format(int(index[0]), int(index[1]), int(index[2])))
            plt.show()
            
        # Return the computed value
        return divergence
    
    """
    def compute_value_Div_S_NS(self, index, verbose=False):
        
        num_players = self.num_players
        
        variance = self.data_generator.get_variance()
        domain = self.data_generator.get_domain()
        
        # Get the combined data of players not choosen
        index_complement = np.ones(num_players) - index
        X_NS, Y_NS = self.data_generator.get_serialized_data(index_complement)
        
        # Compute the posterior 
        X_NS_reshaped = self.model.distribute(domain, self.degree, X_NS)
        Y_NS_reshaped = np.reshape(Y_NS, (len(Y_NS), 1))
        posterior_NS_mean, posterior_NS_covariance = compute_posterior(
                self.prior['mean'], self.prior['cov'], 
                X_NS_reshaped, Y_NS_reshaped, variance)
        
        # Plot to verify
        if verbose:
            plt.axis([domain[0]-0.5, domain[1]+0.5, -13.0, 8.5])
            plotter.plot_sample_functions(posterior_NS_mean, posterior_NS_covariance, self.model, domain, 'green', plt)
            plt.plot(X_NS, Y_NS, 'bo')
            plt.show()
        
        # Get the combined data of players chosen
        X_S, Y_S = self.data_generator.get_serialized_data(index)
        
        # Compute the posterior of S givent NS
        X_S_reshaped = self.model.distribute(domain, self.degree, X_S)
        Y_S_reshaped = np.reshape(Y_S, (len(Y_S), 1))
        posterior_S_mean, posterior_S_covariance = compute_posterior(
                posterior_NS_mean, posterior_NS_covariance, 
                X_S_reshaped, Y_S_reshaped, variance)
    
        # Plot to verify
        if verbose:
            plt.axis([domain[0]-0.5, domain[1]+0.5, -13.0, 8.5])
            plotter.plot_sample_functions(posterior_S_mean, posterior_S_covariance, self.model, domain, 'red', plt)
            plt.plot(X_NS, Y_NS, 'bo')
            plt.plot(X_S, Y_S, 'ro')
            plt.show()
        
        
        # Compute the divergence of NS from prior
        KL_NS = compute_KL(posterior_NS_mean, posterior_NS_covariance, self.prior['mean'], self.prior['cov'])
        
        # Compute the divergence of S from NS
        KL_S_NS = compute_KL(posterior_S_mean, posterior_S_covariance,
                   posterior_NS_mean, posterior_NS_covariance)
        
        # Check values
        if verbose:
            print ("V {:} = {:} -> {:}".format(index, KL_NS, KL_S_NS))
        
        # Return the computed value
        return KL_S_NS
    """

