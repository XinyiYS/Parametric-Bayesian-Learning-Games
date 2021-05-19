import numpy as np

def plot_sample_functions(param_mean, param_covariance, model, domain, color, plt):

    num_samples = 7
    
    sample_functions_X = []
    sample_functions_Y = []

    function = model(domain, len(param_mean) - 1)

    for i in range(num_samples + 1):
        # Draw a parameter
        # Put the mean in the first one
        parameter = None
        if i==0:
            parameter = param_mean
        else:
            parameter = np.random.multivariate_normal(param_mean.flatten(), param_covariance, size=1)
        
        # Construct the function
        function.set_coefficients(parameter.flatten())
        
        # Generate the points
        X = np.arange(domain[0], domain[1] + 0.1, 0.05)
        sample_functions_X.append(X)
        sample_functions_Y.append(function.evaluate(X))
    
    for i in range(len(sample_functions_Y)):
        # Highlight the mean
        if i==0:
            plt.plot(sample_functions_X[i], sample_functions_Y[i], color=color, linewidth=1.5, alpha=1.0)
        else:
            plt.plot(sample_functions_X[i], sample_functions_Y[i], color=color, linewidth=0.5, alpha=0.3)

