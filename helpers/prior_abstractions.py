import numpy as np
from abc import abstractmethod

from utils import gig_rvs

#function to define gig random variable, even for the edge cases:
#a class for running, printing, and calculating error for different priors for the SDE estimation
class Prior:
    def __init__(self, init_data, b_func, diffusion, kernel_name='gauss', gibbs_iters=40):
        self._init_data = init_data
        self._b_func = b_func
        self.diffusion = diffusion
        self._kernel = self._set_kernel(kernel_name)
        self.gibbs_iters = gibbs_iters
        self.lambda_mat_record = []
        self.beta_record = []

    #Using the specifications from initialization to set the kernel for the object
    def _set_kernel(self, kernel_name):
        if kernel_name.lower() == 'gauss':
            def kern(x,y): return np.exp(-((x-y)**2)/2)
            return kern
        if kernel_name.lower() == 'laplace':
            def kern(x,y): return np.exp(-abs(x-y)/np.sqrt(2))
            return kern
        if kernel_name.lower() == 'poly':
            order = 3 #TODO: add a way to specify this order when defining the class
            def kern(x,y): return np.exp(-abs(x-y)/np.sqrt(2)) (x*y + 1)**order
            return kern

    #define the matrix of k-values (kernel output for each possible pair of data points)
    def get_k_matrix(self):
        data = self._init_data
        k_mat_len = len(data)-1
        k_mat = np.zeros((k_mat_len, k_mat_len))
        for i in range(k_mat_len):
            for j in range(k_mat_len):
                k_mat[i][j] = self._kernel(data[i], data[j])
        return np.array(k_mat)

    #create the c vector (pretty self-explanatory what this is)
    def get_c_vect(self):
        data = self._init_data
        c_vect = []
        for i in range(1, len(data)):
            c_vect.append(data[i] - data[i-1])
        return np.array(c_vect)

    #returns the variance matrix
    #zeta and t_delta come from the formula that was used to create the data set
    #in the case of gibbs sampling, zeta changes for each iteration
    def get_variance_matrix(self, k_mat, zeta, lambda_mat):
        #perform calculations, as per the formulas that jinpu derived
        part_1 = (self.t_delta/zeta)*(k_mat.T @ k_mat)
        part_2 = np.linalg.inv(lambda_mat)
        return np.linalg.inv(part_1 + part_2)

    #outputs the vector of mean values for each beta_i
    def get_mu_vector(self, variance_matrix, zeta, c_vect, k_mat):
        kt_times_c = k_mat.T @ c_vect
        return np.matmul(variance_matrix/zeta, kt_times_c)

    @abstractmethod
    def run_gibbs(self, verbose = True):
        pass

#a child of the Prior class; Gig (generalized inverse gaussian)
class Gig(Prior):
    def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20, gig_a = 2, gig_b = 0,
                 gig_p = 1):
        super().__init__(init_data, b_func, diffusion, kernel_name='gauss', gibbs_iters=40)
        self.gig_a = gig_a
        self.gig_b = gig_b
        self.gig_p = gig_p
        # Default for last three is a=2 b=0 p=1 for LASSO prior (see run_gibbs method)

    def run_gibbs(self, verbose = True):
        k_mat = self.get_k_matrix()
        c_vect = self.get_c_vect()
        data_len= len(self._init_data)
        #TODO add a way to specify these following starting priors;
        #right now just set to diffusion = 0.5 and beta = np.zeros()
        zeta = 0.5**2 #a scalar (noise coeff)
        beta = np.zeros(data_len) #a vector (beta coeffs)
        lambda_mat = np.identity(data_len - 1)

        #NOTE: _gig_params in order [c,d,a,b,p] (the last three specify gig.rvs)
        a,b,p = self.gig_a, self.gig_b, self.gig_p
        c, d = 2,2
        c_prime = c + (data_len-1)/2
        t_delta = self.t_delta
        assert self.lambda_mat_record == [] and self.beta_record == [] 'you have already ran the gibbs process for this object; run obj.reset to forget these results'

        for i in range(self.gibbs_iters):
            #draw a diagonal matrix of lambda^2 values, using the inverse gamma distribution
            #this replaces the 1 values inside of lambda_mat
            np.fill_diagonal(lambda_mat, [gig_rvs(a, (beta[i]**2+b), p - 1/2, 1) for i in range(data_len - 1)])
            #Don't know if this shoul dbe there; it seems to really mess up results:
            # lambda_mat = lambda_mat**2
            #use this matrix to compute the matrix and covariance for beta vector
            #sample from these values
            variance_matrix = self.get_variance_matrix(k_mat, zeta, lambda_mat)
            mu_vect = self.get_mu_vector(variance_matrix, zeta, c_vect, k_mat)
            beta = np.random.multivariate_normal(mu_vect,variance_matrix)

            #draw a new guess for zeta given the beta and lambda_mat values
            #it is an inverse gamme of c_prime and d_prime, defined below
            d_prime_1 = (1/(2*t_delta))*np.sum(c_vect**2)
            d_prime_2 = t_delta/2* (beta.T @ (k_mat.T @ (k_mat @ beta))) - c_vect.T @ (k_mat @ beta) + d
            d_prime = d_prime_1 + d_prime_2
            zeta = 1/np.random.gamma(shape = c_prime, scale = 1/d_prime)

            #keeping track of things here
            if verbose:
                if i%25 == 0:
                    print(f'step {i} completed')
            self.lambda_mat_record.append(lambda_mat)
            self.beta_record.append(beta)

#a very similar class, but now for inverse gamma gibbs smapling, rather than generalized inverse gauss (IG vs GIG)
class Ig(Prior):
    def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20, class_params = [2,2,1,2]):
        super().__init__(init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20)
        self._class_params = class_params #the default is to test the t-prior here

    def run_gibbs(self, verbose = True):
        k_mat = self.get_k_matrix()
        c_vect = self.get_c_vect()
        data_len= len(self._init_data)
        #TODO add a way to specify these following starting priors;
        #right now just set to diffusion = 0.5 and beta = np.zeros()
        zeta = 0.5**2 #a scalar (noise coeff)
        beta = np.zeros(data_len) #a vector (beta coeffs)
        lambda_mat = np.identity(data_len - 1)

        c,d,a,b = self._class_params
        c_prime = c + (data_len-1)/2
        t_delta = self.t_delta
        assert self.lambda_mat_record == [] and self.beta_record == [] and self.diffusion_est_record == [], 'you have already ran the gibbs process for this object; run obj.reset to forget these results'

        for i in range(self.gibbs_iters):
            #draw a diagonal matrix of lambda^2 values, using the inverse gamma distribution
            #this replaces the 1 values inside of lambda_mat
            np.fill_diagonal(lambda_mat, [1/np.random.gamma(a + 1/2, 1/(b + (1/2) * beta[i]**2)) for i in range(data_len - 1)])
            #Don't know if this shoul dbe there; it seems to really mess up results:
            # lambda_mat = lambda_mat**2
            #use this matrix to compute the matrix and covariance for beta vector
            #sample from these values
            variance_matrix = self.get_variance_matrix(k_mat, zeta, lambda_mat)
            mu_vect = self.get_mu_vector(variance_matrix, zeta, c_vect, k_mat)
            beta = np.random.multivariate_normal(mu_vect,variance_matrix)

            #draw a new guess for zeta given the beta and lambda_mat values
            #it is an inverse gamme of c_prime and d_prime, defined below
            d_prime_1 = (1/(2*t_delta))*np.sum(c_vect**2)
            d_prime_2 = t_delta/2* (beta.T @ (k_mat.T @ (k_mat @ beta))) - c_vect.T @ (k_mat @ beta) + d
            d_prime = d_prime_1 + d_prime_2
            zeta = 1/np.random.gamma(shape = c_prime, scale = 1/d_prime)

            #keeping track of things here
            if verbose:
                if i%25 == 0:
                    print(f'step {i} completed')
            self.lambda_mat_record.append(lambda_mat)
            self.beta_record.append(beta)
            self.diffusion_est_record.append(zeta)

class Shoe(Prior):
    def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 2000, n = 20):
        super().__init__(init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20)

    def run_gibbs(self, verbose = True):
        k_mat = self.get_k_matrix()
        c_vect = self.get_c_vect()
        data_len= len(self._init_data)
        #TODO add a way to specify these following starting priors;
        #right now just set to diffusion = 0.5 and beta = np.zeros()
        zeta_sq = 1/np.random.gamma(1, 1/2) #The noise coefficient (squared), prior is from inverse gamma (a = 1, b = 2; these are not specific)
        #new variables for the gibbs process
        xi = np.random.gamma(1/2,1)
        eps = np.array([np.random.gamma(1/2, 1) for i in range(data_len- 1)])
        tau_sq = 1/np.random.gamma(1/2, xi)
        lambda_mat = np.identity(data_len - 1)
        #classic beta vector and lambda matrix for gibbs:
        np.fill_diagonal(lambda_mat, [1/np.random.gamma(1/2, eps[i]) for i in range(data_len - 1)])
        beta = np.array([np.random.normal(0,lambda_mat[i,i]*tau_sq) for i in range(data_len - 1)]) #a vector (beta coeffs)
        #idk if these next variables are right; I made them up to fit the code but they could be improper
        a, c, d = 0.5, 1, 1
        c_prime = c + (data_len-1)/2
        t_delta = self.t_delta
        assert self.lambda_mat_record == [] and self.beta_record == [] and self.diffusion_est_record == [], 'you have already ran the gibbs process for this object; run obj.reset to forget these results'

        for i in range(self.gibbs_iters):
            #draw a diagonal matrix of lambda^2 values, using the inverse gamma distribution
            #this replaces the 1 values inside of lambda_mat
            np.fill_diagonal(lambda_mat, [1/np.random.gamma(1, 1/(1/eps[i] + (1/2)*tau_sq*beta[i]**2)) for i in range(data_len - 1)])
            #Don't know if this shoul dbe there; it seems to really mess up results:
            # lambda_mat = lambda_mat**2
            #use this matrix to compute the matrix and covariance for beta vector
            #sample from these values
            variance_matrix = self.get_variance_matrix(k_mat, zeta_sq, lambda_mat)
            mu_vect = self.get_mu_vector(variance_matrix, zeta_sq, c_vect, k_mat)
            beta = np.random.multivariate_normal(mu_vect,variance_matrix)

            #draw a new guess for zeta_sq given the beta and lambda_mat values
            #it is an inverse gamme of c_prime and d_prime, defined below
            d_prime_1 = (1/(2*t_delta))*np.sum(c_vect**2)
            d_prime_2 = t_delta/2* (beta.T @ (k_mat.T @ (k_mat @ beta))) - c_vect.T @ (k_mat @ beta) + d
            d_prime = d_prime_1 + d_prime_2
            zeta_sq = 1/np.random.gamma(shape = c_prime, scale = 1/d_prime)

            #new variable changes
            tau_arg2 = max([0.0001,1/(1/2*np.sum(np.array([beta[i]/lambda_mat[i,i] for i in range(data_len - 1)]))+1/xi)])
            tau_sq = 1/np.random.gamma(data_len/2, tau_arg2)
            eps = np.array([1/np.random.gamma(1, 1/(1/lambda_mat[i,i] + 1)) for i in range(data_len - 1)])
            xi = 1/np.random.gamma(1, 1/(1/tau_sq + 1))

            #keeping track of things here
            if verbose:
                if i%25 == 0:
                    print(f'step {i} completed')
            self.lambda_mat_record.append(lambda_mat)
            self.beta_record.append(beta)
            self.diffusion_est_record.append(zeta_sq)