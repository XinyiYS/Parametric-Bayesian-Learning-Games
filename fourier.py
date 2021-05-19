import numpy as np

class Fourier:
    
    coefficients = None
    
    def __init__(self, domain, degree):
        self.domain = domain
        self.degree = degree
    
    def set_coefficients(self, coefficients):
        self.coefficients = coefficients
    
    # Function for evaluating multiple inputs
    def evaluate(self, x):
        final = np.zeros(len(x))
        L = self.domain[1] - self.domain[0]
        
        freq_cos = 1
        freq_sin = 1
        
        for i in range(self.degree+1):        
            # Compute the basis values
            values = np.ones(len(x))
            if i>0:
                if ((i % 2) == 1):
                    # Odd
                    values = np.cos(np.multiply(2 * freq_cos * np.pi / L, x))                
                    freq_cos = freq_cos + 1
                else:
                    # Even
                    values = np.sin(np.multiply(2 * freq_sin * np.pi / L, x))                
                    freq_sin = freq_sin + 1
            # Multiply by its coefficient
            values = np.multiply(self.coefficients[i], values)        
            # Sum all values
            final = np.add(values, final)
        return final

    # Function for evaluating single unit
    # Mainly for specifying the data generation model in pymc3
    def evaluate_unit(self, x):
        y = 0
        L = self.domain[1] - self.domain[0]
        freq_cos = 1
        freq_sin = 1
        
        for i in range(self.degree+1):
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
            value = self.coefficients[i] * value        
            # Sum all values
            y = y + value
        return y
    
    @staticmethod
    def distribute(domain, degree, x):
        final = []
        L = domain[1] - domain[0]
        
        freq_cos = 1
        freq_sin = 1
        
        for i in range(degree+1):
            # Compute the basis values
            values = np.ones(len(x))
            if i>0:
                if ((i % 2) == 1):
                    # Odd
                    values = np.cos(np.multiply(2 * freq_cos * np.pi / L, x))                
                    freq_cos = freq_cos + 1
                else:
                    # Even
                    values = np.sin(np.multiply(2 * freq_sin * np.pi / L, x))                
                    freq_sin = freq_sin + 1
            # Combine the distributed basis
            final.append(values)
        
        return np.transpose(np.array(final))

