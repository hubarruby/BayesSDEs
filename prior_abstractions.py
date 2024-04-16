import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geninvgauss, gamma, invgamma
from datetime import datetime
from abc import abstractmethod
import pickle
import pandas as pd

def gig_rvs(a, b, p, size):
    '''
    Generate Generalized Inverse Gaussian random variables

    pdf: f(x, a, b, p) = ( (a/b)^{p/2} / (2 K_p(\sqrt{ab})) ) * x^{p-1} \exp(-ax + b/x)
          where a > 0, b > 0, p is a real number

    when a --> 0, GIG --> InvGamma;
    when b --> 0, GIG --> Gamma;

    Special cases:
        Gamma(shape=alpha, rate=beta) = GIG(2*beta, 0, alpha)
        InvGamma(alpha, beta) = GIG(0, 2*beta, -alpha)
        InvGaussian(mu, lambda) = GIG(lambda/mu^2, lambda, −1/2)
        Park and Casella’s Lasso(alpha^2) = GIG(alpha^2, 0, 1)
    '''
    if a == 0 and b == 0:
        raise ValueError('GIG can not have input with both a and b being 0')

    if b == 0: # Gamma
        shape_gamma = p
        rate_gamma = a / 2
        rvs = gamma.rvs(a=shape_gamma, scale=1/rate_gamma, size=size)

    if a == 0: # InvGamma
        shape_invgamma = -p
        scale_invgamma = b / 2
        rvs = invgamma.rvs(a=shape_invgamma, scale=scale_invgamma, size=size)

    if a != 0 and b != 0:
        p_ss = p
        b_ss = np.sqrt(a*b)
        scale_ss = np.sqrt(b/a)
        rvs = geninvgauss.rvs(p=p_ss, b=b_ss, loc=0, scale=scale_ss, size=size)

    return rvs

#function to define gig random variable, even for the edge cases:
#a class for running, printing, and calculating error for different priors for the SDE estimation
class Prior:
    def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 15000, n = 20):
        self._init_data = init_data
        self._kernel = self._set_kernel(kernel_name)
        self._b_func = b_func
        self.gibbs_iters = 150
        self.diffusion = diffusion
        self.lambda_mat_record = []
        self.beta_record = []
        self.diffusion_est_record = []
        #this is based on how the input data was generated:
        #I dont have it as an input because all of the data we've generated so far has used t_delta=0.05
        self.t_delta = 0.05
        self._m = m
        self._n = n
        self._b_mdata = None
        self._bhat_mdata = None
        self._b_cdf = None
        self._bhat_cdf = None
        self._mse = None
        self._kolmogorov = None

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

    def _create_data(self, s):
        np.random.seed(999)
        t = 0
        val = 0
        data = []
        t_delta = self.t_delta

        if s.lower() == 'b':
            def create_next(val):
                return val + self._b_func(val)*t_delta + self.diffusion*np.random.normal(0,np.sqrt(t_delta))

            for i in range(int(self._m)):
                data.append(val)
                val = create_next(val)
                t += t_delta
            return data

        elif s.lower() == 'bhat':
            assert self.beta_record != [], "Error: trying to generate data with bhat before bhat exists."
            def create_next(val):
                #this should change to use the last x% of data samples, not just beta_record[-20:]
                return val + self.get_b_hat(val, np.mean(self.beta_record[-20:], axis = 0))*t_delta + self.diffusion*np.random.normal(0,np.sqrt(t_delta))
            for i in range(int(self._m)):
                data.append(val)
                val = create_next(val)
                t += t_delta
            return data

    def get_b_hat(self, x, beta_vect):
        data = self._init_data
        return np.sum([beta_vect[i]*self._kernel(x, data[i]) for i in range(len(data)-1)])

    #define the matrix of k-values (kernel output for each possible pair of data points)
    #TODO: I want to try to vectorize this function, which will probably dramatically speed up the simulation speeds
    def get_k_matrix(self):
        data = self._init_data
        k_mat_len = len(data)-1
        k_mat = np.zeros((k_mat_len, k_mat_len))
        for i in range(k_mat_len):
            for j in range(k_mat_len):
                k_mat[i][j] = self._kernel(data[i], data[j])
        return np.array(k_mat)

    #create the c vector (pretty self-explanatory what this is)
    #TODO: would also be good to vectorize this
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

    def set_metric_data_size(self, m):
        assert self._b_mdata != [] and self._bhat_mdata != [], f'Metric test data has already been generated using m = {self.__m}; manually reset _b_mdata and __bhat_mdata to change m'
        self._m = m

    def _create_mdata(self, s):
        #frankly, this could be turned into a single function,instead of calling the hidden funciton too
        if 'bhat' in s.strip():
            self._bhat_mdata = self._create_data(s)[int(self._m/10):]
        elif 'b' in s.strip():
            #the indexing at the end is to remove the first 10% of data
            self._b_mdata = self._create_data(s)[int(self._m/10):]
        else:
            raise Exception('invalid keyword for _create_mdata')

    def return_mdata(self, s):
        if 'bhat' == s.lower() and self._bhat_mdata != None:
            return self._bhat_mdata
        elif 'b' == s.lower() and self._b_mdata != None:
            return self._b_mdata
        else:
            raise Exception('invalid keyword for _return_mdata; use "bhat" or "b"')

    #plots a histogram of the data that is available
    def data_hist(self,s, smoothing = True):
        if smoothing:
            if 'bhat' == s.lower():
                if self._bhat_mdata != None:
                    sns.displot(self._bhat_mdata, kind = "kde", fill = True)
                    plt.title("bhat generated data histogram")
                    plt.show()
                else: raise Exception("cannot make a histogram for data that doesn't exist yet (bhat)")
            elif "b" == s.lower():
                if self._b_mdata != None:
                    sns.displot(self._b_mdata, kind = "kde", fill = True)
                    plt.title("true b generated data histogram")
                    plt.show()
                else:
                    raise Exception("cannot make a histogram for data that doesn't exist yet (b)")
            elif s.lower() == 'original':
                sns.displot(self._init_data, kind = "kde", fill = True)
                plt.title("True b original learning data histogram")
                plt.show()
            else:
                raise Exception('invalid keyword for data_hist; use "bhat", "b", or "original".')
        else:
            if 'bhat' == s.lower():
                if self._bhat_mdata != None:
                    sns.displot(self._bhat_mdata, bins = 100, stat ='density')
                    plt.title("bhat generated data histogram")
                    plt.show()
                else: raise Exception("cannot make a histogram for data that doesn't exist yet (bhat)")
            elif "b" == s.lower():
                if self._b_mdata != None:
                    sns.displot(self._b_mdata, bins = 100, stat ='density')
                    plt.title("true b generated data histogram")
                    plt.show()
                else:
                    raise Exception("cannot make a histogram for data that doesn't exist yet (b)")
            elif s.lower() == 'original':
                sns.displot(self._init_data, bins = 100, stat ='density')
                plt.title("True b original learning data histogram")
                plt.show()
            else:
                raise Exception('invalid keyword for data_hist; use "bhat", "b", or "original".')

    def plot_b_vs_bhat(self):
        # plotting bhat using the averaged last n values recorded from gibbs
        # can be made more efficient
        n = self._n
        x_vals = self._init_data
        last_n_beta_avg = np.mean(np.array(self.beta_record[-n:]), axis = 0)
        avgd_beta_est = [self.get_b_hat(x, last_n_beta_avg) for x in x_vals]
        plt.scatter(x_vals, avgd_beta_est, c = 'red', label = f'Approximated b, averaged last {n} gibbs samples', s= 2)
        plt.scatter(x_vals, [self._b_func(x) for x in x_vals], c = 'blue', label = 'True b', s= 2)

    def plot_shrinkage(self):
        last_beta = self.beta_record[-1]
        plt.hist(last_beta, bins = 50)

    def find_mse(self):
        #when sampling the last few values of beta to get bhat, do we
        # 1) average n beta vectors, then calc the bhat values using the average, or
        #2) calc bhat values for n different beta vectors, and average the bhat values?
        #hypothesis: betas are the thing being sampled, so we should average them first.
        if self._mse == None:
            n = self._n
            data = self._init_data
            self._mse = (1/len(data))*np.sum(np.array([self.get_b_hat(x,
                                                                      np.mean(np.array(self.beta_record[-n:]), axis = 0)) - self._b_func(x) for x in data])**2)
        else:
            raise Exception('no point recalculating mse error; you can use "return_mse()" to output it.')

    def return_mse(self):
        return self._mse

    def _calc_cdf(self, s):
        if 'bhat' == s.lower():
            self._bhat_cdf = (1/len(self._bhat_mdata))*np.array([np.sum(np.where(np.array(self._bhat_mdata) <= x, 1, 0)) for x in self._init_data])
        elif 'b' == s.lower():
            self._b_cdf = (1/len(self._b_mdata))*np.array([np.sum(np.where(np.array(self._b_mdata) <= x, 1, 0)) for x in self._init_data])
        else:
            raise Exception('must call _calc_cdf using string "bhat" or "b".')

    def find_kolmogorov(self):
        if self._kolmogorov == None:
            self._create_mdata('b')
            self._create_mdata('bhat')
            self._calc_cdf('bhat')
            self._calc_cdf('b')
            self._kolmogorov = np.max(np.absolute(self._b_cdf - self._bhat_cdf))

    def return_kolmogorov(self):
        return self._kolmogorov

#a child of the Prior class; Gig (generalized inverse gaussian)
class Gig(Prior):
    def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20, class_params = [2,2,2,0,1]):
        super().__init__(init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20)
        self._class_params = class_params #first two are default c = 2, d =2. Default for last three is a=2 b=0 p=1 for LASSO prior (see run_gibbs method)

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
        c,d,a,b,p = self._class_params
        c_prime = c + (data_len-1)/2
        t_delta = self.t_delta
        assert self.lambda_mat_record == [] and self.beta_record == [] and self.diffusion_est_record == [], 'you have already ran the gibbs process for this object; run obj.reset to forget these results'

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
            self.diffusion_est_record.append(zeta)

    def plot_b_vs_bhat(self):
        super().plot_b_vs_bhat()
        plt.title(f'graph using GIG Prior (params a,b,p = {self._class_params[2:]})')
        plt.legend()
        plt.show()

    def plot_shrinkage(self):
        super().plot_shrinkage()
        plt.title('GIG: final beta sample/shrinkage plot')
        plt.show()


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

    def plot_b_vs_bhat(self):
        # plotting bhat using the averaged last n values recorded from gibbs
        super().plot_b_vs_bhat()
        plt.title(f'graph using Inverse Gamma prior (params [a, b] = {self._ig_params[-2:]}')
        plt.legend()
        plt.show()

    def plot_shrinkage(self):
        super().plot_shrinkage()
        plt.title('IG: final beta sample/shrinkage plot')
        plt.show()

#a very similar class, but now for inverse gamma gibbs smapling, rather than generalized inverse gauss (IG vs GIG)
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

    def plot_b_vs_bhat(self):
        # plotting bhat using the averaged last n values recorded from gibbs
        super().plot_b_vs_bhat()
        plt.title(f'graph using Horseshoe Prior')
        plt.legend()
        plt.show()

    def plot_shrinkage(self):
        super().plot_shrinkage()
        plt.title('Horseshoe: final beta shrinkage plot')
        plt.show()


    class GL_Gig(Prior):
        def __init__(self, init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20, local_gig_params = [0.5, 0, 2], global_gig_params = [2,0,1]):
            super().__init__(init_data, b_func, diffusion, kernel_name = 'gauss', m = 100000, n = 20)
            self._local_gig_params = local_gig_params #typically, we'd want this to have a heavy tail (it is the form [a,b,p], default is [0.5,0,2], which has a large tail (but not much shrinkage))
            self._global_gig_params = global_gig_params #typically, we'd want this to have high shrinkage (it is the form [a,b,p], default is the Lasso, which has high shrinkage)

        def run_gibbs(self, verbose = True):
            k_mat = self.get_k_matrix()
            c_vect = self.get_c_vect()
            data_len= len(self._init_data)
            #TODO add a way to specify these following starting priors;
            #right now just set to diffusion = 0.5 and beta = np.zeros()
            zeta = 0.5**2 #a scalar (noise coeff)
            beta = np.zeros(data_len) #a vector (beta coeffs)
            lambda_mat = np.identity(data_len - 1)

            #NOTE: _gig_params in order [a,b,p] (the three specify gig.rvs)
            a_loc,b_loc,p_loc = self._local_gig_params
            a_glob,b_glob,p_glob = self._global_gig_params
            c = 2 #somewhat arbitrary, at least for right now
            c_prime = c + (data_len-1)/2
            t_delta = self.t_delta
            assert self.lambda_mat_record == [] and self.beta_record == [] and self.diffusion_est_record == [], 'you have already ran the gibbs process for this object; run obj.reset to forget these results'

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
                self.diffusion_est_record.append(zeta)

        def plot_b_vs_bhat(self):
            super().plot_b_vs_bhat()
            plt.title(f'graph using GIG Prior (params a,b,p = {self._class_params[2:]})')
            plt.legend()
            plt.show()

        def plot_shrinkage(self):
            super().plot_shrinkage()
            plt.title('GIG: final beta sample/shrinkage plot')
            plt.show()