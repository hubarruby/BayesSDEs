import numpy as np
import time
from scipy.optimize import minimize
from scipy import integrate

from helpers.utils import gig_rvs


def estimated_b_function_matrix(range_linspace, init_data, kernel, known_b, y_domain, int_n=5000, chunk_size=100):
    """
    function for computing the matrix for estimate b_mat, over a particular linspace
        (usually for plotting/MSE calculation purposes)
    :param range_linspace:
    :param init_data:
    :param kernel:
    :param known_b:
    :param y_domain:
    :param int_n:
    :param chunk_size:
    :return:
    """
    # y_domain is the range from which to randomly sample (ie [-1000, 1000] in example 3.1)
    # int_n is the size of sampling for conducting the integration

    # for ease of writing code, defining the linspace such that it has the same number of points as the init_data
    linspace_vals = np.linspace(range_linspace[0], range_linspace[1], len(init_data))

    # Chunking init_data processing
    n_total = len(init_data) - 1
    n = int_n
    b_aggregated = np.zeros((n_total, n_total), dtype=np.float32)  # Placeholder for aggregated results

    y_z = np.random.uniform(y_domain[0], y_domain[1], size=(int_n, 2)).astype(np.float32)
    kern_vals = np.diag(kernel(y_z[:, 0], y_z[:, 1])).astype(np.float32)

    for start_idx_i in range(0, n_total, chunk_size):
        end_idx_i = min(start_idx_i + chunk_size, n_total)
        b_y_chunk = known_b(np.asarray(linspace_vals, dtype=np.float32)[start_idx_i:end_idx_i], y_z[:, 0]).astype(
            np.float32)
        mb_y = b_y_chunk[:, np.newaxis, :]

        for start_idx_j in range(0, n_total, chunk_size):
            end_idx_j = min(start_idx_j + chunk_size, n_total)

            # Compute b_y and b_z for the current chunk
            b_z_chunk = known_b(np.asarray(init_data, dtype=np.float32)[start_idx_j:end_idx_j], y_z[:, 1]).astype(
                np.float32)
            mb_z = b_z_chunk[np.newaxis, :, :]

            b_chunk = mb_y * mb_z * kern_vals

            # Sum over the n dimension (the integral approximation) and accumulate
            b_aggregated[start_idx_i:end_idx_i, start_idx_j:end_idx_j] += np.sum(b_chunk, axis=2) / n

    b_mat = b_aggregated
    return b_mat


def estimated_b_function_mat_calc(b_ij, betas):
    # now we use the betas and multiply them into the matrix along the columns axis
    # (since j is columns, and each beta (or c) corresponds to a j value
    return np.matmul(b_ij, betas)


# 1D version of the B estimation
def estimated_b_function_vect(val, init_data, kernel, known_b, y_rvs, int_n=5000):
    """
    function for computing a single estimate b_mat
    val: float
    y_rvs: is a scipy.stats sampling method (must have a callable .rvs())
    int_n is the size of sampling for conducting the integration
    """

    # for ease of writing code, defining the linspace such that it has the same number of points as the init_data
    y_z = y_rvs.rvs(size=(int_n, 2)).astype(np.float32)
    kern_vals = np.diag(kernel(y_z[:, 0], y_z[:, 1])).astype(np.float32)
    # print(kern_vals.shape)

    # Chunking init_data processing
    n = int_n

    # Compute b_y and b_z
    b_y = known_b(val, y_z[:, 0]).astype(np.float32)  # shape (n,)
    b_z = known_b(np.asarray(init_data, dtype=np.float32), y_z[:, 1]).astype(np.float32)  # shape (n_total, n)

    # mb_y = b_y[:, np.newaxis]
    # mb_z = b_z[np.newaxis, :]
    kern_vals = kern_vals
    b = b_y * b_z * kern_vals  # try doing this with built-in broadcasting in numpy
    # print(b.shape)
    #
    # # Sum over the n dimension (the integral approximation) and accumulate
    b_vec = np.sum(b, axis=1)[:-1] / n

    return b_vec


def b1_est_func_vect_calc(val, betas, init_data, kernel, known_b, y_rvs, int_n=5000):
    return betas @ estimated_b_function_vect(val, init_data, kernel, known_b, y_rvs, int_n=int_n)


# a class for running, different priors for the SDE estimation usign the Fredholm Method
class FredholmGlobLoc:
    def __init__(self,
                 init_data,
                 known_b,  # this should be a function that is compatible with matrix versions of the operation
                 diffusion,  # included because we are assuming we know the noise in this global-local scenario
                 y_domain,
                 kernel_name='gauss',
                 gibbs_iters=150,
                 t_delta=0.05,
                 a_loc=2,  # lambda
                 b_loc=0,  # lambda
                 p_loc=1,  # lambda
                 a_glob=2,  # tau
                 b_glob=0,  # tau
                 p_glob=1,  # tau
                 int_n=5000,  # number of values to generate when numerically estimated the b integral
                 chunk_size=100,
                 timer=True
                 ):
        self._local_gig_a = a_loc  # desired low shrinkage
        self._local_gig_b = b_loc
        self._local_gig_p = p_loc
        self._global_gig_a = a_glob  # desired high shrinkage.
        self._global_gig_b = b_glob  # For the case where we only want one prior, we ignore the global.
        self._global_gig_p = p_glob  # Instead, set them all to 0 to make the gibbs process only use the local params.
        self.init_data = init_data
        self._kernel = self._set_kernel(kernel_name)
        self.known_b = known_b
        self.y_domain = y_domain
        self.gibbs_iters = gibbs_iters
        self.diffusion = diffusion
        self.lambda_mat_record = []
        self.beta_record = []
        self.t_delta = t_delta
        self.chunk_size = chunk_size
        self.int_n = int_n
        self.timer = timer
        self.b_mat = None

    # Using the specifications from initialization to set the kernel for the object
    @staticmethod
    def _set_kernel(kernel_name):
        if kernel_name.lower() == 'gauss':
            def kern(x, y):
                """
                Computes the Gaussian kernel exp(-((x - y)^2) / 2) for each combination of elements
                in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
                a matrix when both x and y are vectors, a vector when one of them is a scalar, and
                a scalar when both are scalars.

                Args:
                - x (numpy.array or scalar): Input vector or scalar x.
                - y (numpy.array or scalar): Input vector or scalar y.

                Returns:
                - numpy.array or scalar: Gaussian kernel values.
                """
                # Wrap x and y in arrays if they are scalars
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)

                # Broadcasting x and y to form all pairs (x[i] - y[j])
                x_expanded = x[:, np.newaxis]
                y_expanded = y[np.newaxis, :]

                # Applying the Gaussian kernel to each pair
                result = np.exp(-((x_expanded - y_expanded) ** 2) / 2)

                # Return result in original shape (scalar if both inputs were scalars)
                if result.size == 1:
                    return result.item()  # Return as scalar
                elif len(x) == 1 or len(y) == 1:
                    return result.flatten()  # Return 1D array if one input was scalar
                return result

            return kern

        if kernel_name.lower() == 'laplace':
            def kern(x, y): return np.exp(-abs(x - y) / np.sqrt(2))

            return kern

        if kernel_name.lower() == 'poly':
            order = 3  # TODO: add a way to specify this order when defining the class

            def kern(x, y): return np.exp(-abs(x - y) / np.sqrt(2))(x * y + 1) ** order

            return kern

        if kernel_name.lower() == 'mult_exp':
            def kern(x, y):
                """
                Computes the Kernel exp(-(x^2 + y^2) / 2) for each combination of elements
                in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
                a matrix when both x and y are vectors, a vector when one of them is a scalar, and
                a scalar when both are scalars.

                Args:
                - x (numpy.array or scalar): Input vector or scalar x.
                - y (numpy.array or scalar): Input vector or scalar y.

                Returns:
                - numpy.array or scalar: multiplicative exponential kernel values.
                """
                # Wrap x and y in arrays if they are scalars
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)

                # Broadcasting x and y to form all pairs (x[i] - y[j])
                x_expanded = x[:, np.newaxis]
                y_expanded = y[np.newaxis, :]

                # Applying the kernel to each pair
                result = np.exp(- (x_expanded ** 2 + y_expanded ** 2) / 2)

                # Return result in original shape (scalar if both inputs were scalars)
                if result.size == 1:
                    return result.item()  # Return as scalar
                elif len(x) == 1 or len(y) == 1:
                    return result.flatten()  # Return 1D array if one input was scalar
                return result

            return kern

    # define the matrix of b_integrals (matrix output with a value for each possible pair of data points in init_data)
    # this is the slow version of the
    def get_b_integral(self):
        # start_time = time.time()
        # create y and z samples for calculating the integral
        y_z = np.random.uniform(self.y_domain[0], self.y_domain[1], size=(self.int_n, 2))
        # calculate the kernel for each of these y and z values together
        kern_vals = np.diag(self._kernel(y_z[:, 0], y_z[:, 1]))

        b_int = np.zeros((len(self.init_data) - 1, len(self.init_data) - 1))
        # doing this as a loop first to make it easier to conceptualize
        for i, xi in enumerate(self.init_data[:-1]):
            if i % 10 == 0:
                print(f'For B, at outer loop point {i + 1}/{len(self.init_data) + 1} in init_data')
            for j, xj in enumerate(self.init_data[:-1]):
                dy_b_result = self.known_b(xi, y_z[:, 0])
                dz_b_result = self.known_b(xj, y_z[:, 1])
                # if i==0: print(kern_vals.shape, dy_b_result.shape, dz_b_result.shape)
                elementise_mult = dy_b_result * dz_b_result * kern_vals
                b_int[i, j] = np.sum(elementise_mult) / self.int_n
        # print('done making B')
        # print(b_int.shape)
        self.b_mat = np.array(b_int)
        # end_time = time.time()
        # Calculate the total time taken and print it
        # total_time = end_time - start_time
        # print(f"Total time taken for non-vectorized B gen: {total_time} seconds")
        return np.array(b_int)

    def get_b_integral_v_chunked(self):
        # Create y and z samples for calculating the integral
        y_z = np.random.uniform(self.y_domain[0], self.y_domain[1], size=(self.int_n, 2)).astype(np.float32)
        kern_vals = np.diag(self._kernel(y_z[:, 0], y_z[:, 1])).astype(np.float32)

        # Chunking init_data processing
        chunk_size = self.chunk_size
        n_total = len(self.init_data) - 1
        n = self.int_n
        b_aggregated = np.zeros((n_total, n_total), dtype=np.float32)  # Placeholder for aggregated results

        for start_idx_i in range(0, n_total, chunk_size):
            end_idx_i = min(start_idx_i + chunk_size, n_total)
            b_y_chunk = self.known_b(np.asarray(self.init_data, dtype=np.float32)[start_idx_i:end_idx_i],
                                     y_z[:, 0]).astype(np.float32)
            mb_y = b_y_chunk[:, np.newaxis, :]

            for start_idx_j in range(0, n_total, chunk_size):
                end_idx_j = min(start_idx_j + chunk_size, n_total)

                b_z_chunk = self.known_b(np.asarray(self.init_data, dtype=np.float32)[start_idx_j:end_idx_j],
                                         y_z[:, 1]).astype(np.float32)
                mb_z = b_z_chunk[np.newaxis, :, :]

                b_chunk = mb_y * mb_z * kern_vals

                # Sum over the n dimension (the integral approximation) and accumulate
                b_aggregated[start_idx_i:end_idx_i, start_idx_j:end_idx_j] += np.sum(b_chunk, axis=2) / n

        self.b_mat = b_aggregated
        return b_aggregated

    def get_b_integral_scipy(self):
        n_total = len(self.init_data) - 1
        # this version takes a lot longer than get_b_integral_v_chunked, but it is more accurate
        b_mat = np.zeros((n_total, n_total), dtype=np.float32)
        start_time = time.time()
        for coi, xi in enumerate(self.init_data[:-1]):
            print(f'calculating b_integral values for row {coi}/{len(self.init_data[:-1])}')
            if self.timer:
                current_time = time.time()
                elapsed_time = current_time - start_time
                progress = (coi + 1) / n_total
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
                print(f"Elapsed time: {elapsed_time/60:.2f} min, Estimated total time: "
                      f"{estimated_total_time/60:.2f} min, Remaining time: {remaining_time/60:.2f} min")

            for coj, xj in enumerate(self.init_data[:-1]):
                b_mat[coi, coj] = integrate.dblquad(
                    lambda y, z: self.known_b(xi, y) * self.known_b(xj, z) * self._kernel(y, z),
                    self.y_domain[0], self.y_domain[1], lambda z: self.y_domain[0], lambda z: self.y_domain[1])[0]
        self.b_mat = b_mat
        return b_mat

    # returns the variance matrix
    def get_diagonal_matrix(self):
        data_len = len(self.init_data) - 1
        d_mat = np.identity(data_len) * 1 / self.diffusion ** 2
        return d_mat

    def get_eta(self, lambda_mat, tau):
        eta = np.identity(len(self.init_data) - 1)
        eta = eta * lambda_mat  # elementwise multiplication to put lambda vals along the diagonal
        eta = eta * tau  # since tau is a scalar this applies itself to the whole matrix (along the diag)
        return eta

    def get_c_mat(self, b_int, d_mat, eta):
        # print(type(b_int))
        c_inv = self.t_delta * ((b_int.T @ d_mat) @ b_int) + np.linalg.inv(eta)
        return np.linalg.inv(c_inv)  # return C by taking the inverse of c_inv

    def get_theta(self):  # theta looks like a fancy "v" with a curl on the right side (for reference to the note)
        constructor1 = np.asarray(self.init_data)[1:]
        constructor2 = np.asarray(self.init_data)[:-1]
        theta = constructor1 - constructor2
        return theta

    # outputs the vector of mean values for each beta_i
    @staticmethod
    def get_mu_vector(c_mat, b_int, d_mat, theta):
        mu = c_mat @ b_int.T @ d_mat @ theta
        return mu

    def run_gibbs(self, verbose=True, matrix_calc='chunked'):
        a_loc = self._local_gig_a  # lambda
        b_loc = self._local_gig_b  # lambda
        p_loc = self._local_gig_p  # lambda
        a_glob = self._global_gig_a  # tau
        b_glob = self._global_gig_b  # tau
        p_glob = self._global_gig_p  # tau
        print('a_loc', a_loc, 'b_loc', b_loc, 'p loc', p_loc)
        print('a_glob', a_glob, 'b_glob', b_glob, 'p glob', p_glob)

        if matrix_calc == 'chunked':
            b_int = self.get_b_integral_v_chunked()
        elif matrix_calc == 'long':
            b_int = self.get_b_integral()
        elif matrix_calc == 'scipy':
            b_int = self.get_b_integral_scipy()
        else:
            raise AttributeError

        d_mat = self.get_diagonal_matrix()
        data_len = len(self.init_data)

        # initialize tau (semi-randomly)
        if a_glob == b_glob == p_glob == 0:
            # in this case, we are not using global priors so tau is not modified during the gibbs process
            global_prior = False
            tau = 1
        else:
            global_prior = True
            tau = gig_rvs(a_glob, b_glob, p_glob, 1)
        # randomly initialize beta
        beta = np.random.normal(0, 0.05, data_len - 1)  # a vector (beta coeffs)
        lambda_mat = np.identity(data_len - 1)
        theta = self.get_theta()

        for i in range(self.gibbs_iters):
            # draw a diagonal matrix of lambda^2
            np.fill_diagonal(lambda_mat,
                             [gig_rvs(a_loc, (1 / tau) * (beta[i] ** 2) + b_loc, p_loc - 1 / 2 - 1, 1) for i in
                              range(data_len - 1)])

            # lambda then informs the distribution of tau
            tau = gig_rvs(a_glob, np.sum(
                [(1 / lamb) * (beta[i] ** 2) for i, lamb in enumerate(np.diagonal(lambda_mat))]) + b_glob,
                          p_glob - 1 / 2 - 1, 1) if global_prior else 1

            # now, using these, we sample beta
            eta = self.get_eta(lambda_mat, tau)
            c_mat = self.get_c_mat(b_int, d_mat, eta)  # covariance matrix of beta
            mu = self.get_mu_vector(c_mat, b_int, d_mat, theta)  # mean vector of beta
            # the beta comes from a multivariate normal
            beta = np.random.multivariate_normal(mu, c_mat)

            # keeping track of progress here
            if verbose:
                if i % 25 == 0:
                    print(f'step {i}/{self.gibbs_iters} completed')
            self.lambda_mat_record.append(lambda_mat)
            self.beta_record.append(beta)


def gauss_kernel(x, y):
    """
    Computes the Gaussian kernel exp(-((x - y)^2) / 2) for each combination of elements
    in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
    a matrix when both x and y are vectors, a vector when one of them is a scalar, and
    a scalar when both are scalars.

    Args:
    - x (numpy.array or scalar): Input vector or scalar x.
    - y (numpy.array or scalar): Input vector or scalar y.

    Returns:
    - numpy.array or scalar: Gaussian kernel values.
    """
    # Wrap x and y in arrays if they are scalars
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Broadcasting x and y to form all pairs (x[i] - y[j])
    x_expanded = x[:, np.newaxis]
    y_expanded = y[np.newaxis, :]

    # Applying the Gaussian kernel to each pair
    result = np.exp(-((x_expanded - y_expanded) ** 2) / 2)

    # Return result in original shape (scalar if both inputs were scalars)
    if result.size == 1:
        return result.item()  # Return as scalar
    elif len(x) == 1 or len(y) == 1:
        return result.flatten()  # Return 1D array if one input was scalar
    return result


def multiplicative_exponential_kernel(x, y):
    """
    Computes the Kernel exp(-(x^2 + y^2) / 2) for each combination of elements
    in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
    a matrix when both x and y are vectors, a vector when one of them is a scalar, and
    a scalar when both are scalars.

    Args:
    - x (numpy.array or scalar): Input vector or scalar x.
    - y (numpy.array or scalar): Input vector or scalar y.

    Returns:
    - numpy.array or scalar: multiplicative exponential kernel values.
    """
    # Wrap x and y in arrays if they are scalars
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Broadcasting x and y to form all pairs (x[i] - y[j])
    x_expanded = x[:, np.newaxis]
    y_expanded = y[np.newaxis, :]

    # Applying the kernel to each pair
    result = np.exp(- (x_expanded ** 2 + y_expanded ** 2) / 2)

    # Return result in original shape (scalar if both inputs were scalars)
    if result.size == 1:
        return result.item()  # Return as scalar
    elif len(x) == 1 or len(y) == 1:
        return result.flatten()  # Return 1D array if one input was scalar
    return result


class FredholmOptimize:
    def __init__(self,
                 init_data,
                 known_b,  # this should be a function that is compatible with matrix versions of th operation
                 diffusion,  # included because we are assuming we know the noise in this global-local scenario
                 y_domain,
                 kernel_name='gauss',
                 t_delta=0.05,
                 int_n=5000,  # number of values to generate when numerically estimated the b integral
                 chunk_size=100  # for calculating the B Integral
                 ):
        self.init_data = init_data
        self.known_b = known_b
        self.y_domain = y_domain
        self._kernel = self._set_kernel(kernel_name)
        self.diffusion = diffusion
        self.t_delta = t_delta
        self.int_n = int_n
        self.chunk_size = chunk_size
        self.b_mat = None
        self.optimal_beta = None  # this is referred to as c in the Fredholm note
        self.optimization_history = None

    @staticmethod
    def _set_kernel(kernel_name):
        if kernel_name.lower() == 'gauss':
            def kern(x, y):
                """
                Computes the Gaussian kernel exp(-((x - y)^2) / 2) for each combination of elements
                in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
                a matrix when both x and y are vectors, a vector when one of them is a scalar, and
                a scalar when both are scalars.

                Args:
                - x (numpy.array or scalar): Input vector or scalar x.
                - y (numpy.array or scalar): Input vector or scalar y.

                Returns:
                - numpy.array or scalar: Gaussian kernel values.
                """
                # Wrap x and y in arrays if they are scalars
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)

                # Broadcasting x and y to form all pairs (x[i] - y[j])
                x_expanded = x[:, np.newaxis]
                y_expanded = y[np.newaxis, :]

                # Applying the Gaussian kernel to each pair
                result = np.exp(-((x_expanded - y_expanded) ** 2) / 2)

                # Return result in original shape (scalar if both inputs were scalars)
                if result.size == 1:
                    return result.item()  # Return as scalar
                elif len(x) == 1 or len(y) == 1:
                    return result.flatten()  # Return 1D array if one input was scalar
                return result

            return kern

        if kernel_name.lower() == 'laplace':
            def kern(x, y): return np.exp(-abs(x - y) / np.sqrt(2))

            return kern

        if kernel_name.lower() == 'poly':
            order = 3  # TODO: add a way to specify this order when defining the class

            def kern(x, y): return np.exp(-abs(x - y) / np.sqrt(2))(x * y + 1) ** order

            return kern

        if kernel_name.lower() == 'mult_exp':
            def kern(x, y):
                """
                Computes the Kernel exp(-(x^2 + y^2) / 2) for each combination of elements
                in x and y, which can be vectors (numpy arrays) or single scalar values. The result is
                a matrix when both x and y are vectors, a vector when one of them is a scalar, and
                a scalar when both are scalars.

                Args:
                - x (numpy.array or scalar): Input vector or scalar x.
                - y (numpy.array or scalar): Input vector or scalar y.

                Returns:
                - numpy.array or scalar: multiplicative exponential kernel values.
                """
                # Wrap x and y in arrays if they are scalars
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)

                # Broadcasting x and y to form all pairs (x[i] - y[j])
                x_expanded = x[:, np.newaxis]
                y_expanded = y[np.newaxis, :]

                # Applying the kernel to each pair
                result = np.exp(- (x_expanded ** 2 + y_expanded ** 2) / 2)

                # Return result in original shape (scalar if both inputs were scalars)
                if result.size == 1:
                    return result.item()  # Return as scalar
                elif len(x) == 1 or len(y) == 1:
                    return result.flatten()  # Return 1D array if one input was scalar
                return result

            return kern

    # define the matrix of b_integrals (matrix output with a value for each possible pair of data points in init_data)
    # this is the slow version of the
    def get_b_integral(self):
        # start_time = time.time()
        # create y and z samples for calculating the integral
        y_z = np.random.uniform(self.y_domain[0], self.y_domain[1], size=(self.int_n, 2))
        # calculate the kernel for each of these y and z values together
        kern_vals = np.diag(self._kernel(y_z[:, 0], y_z[:, 1]))

        b_int = np.zeros((len(self.init_data) - 1, len(self.init_data) - 1))
        # doing this as a loop first to make it easier to conceptualize
        for i, xi in enumerate(self.init_data[:-1]):
            if i % 10 == 0:
                print(f'For B, at outer loop point {i + 1}/{len(self.init_data) + 1} in init_data')
            for j, xj in enumerate(self.init_data[:-1]):
                dy_b_result = self.known_b(xi, y_z[:, 0])
                dz_b_result = self.known_b(xj, y_z[:, 1])
                # if i==0: print(kern_vals.shape, dy_b_result.shape, dz_b_result.shape)
                elementise_mult = dy_b_result * dz_b_result * kern_vals
                b_int[i, j] = np.sum(elementise_mult) / self.int_n
        # print('done making B')
        print(b_int.shape)
        self.b_mat = np.array(b_int)
        # end_time = time.time()
        # Calculate the total time taken and print it
        # total_time = end_time - start_time
        # print(f"Total time taken for non-vectorized B gen: {total_time} seconds")
        return np.array(b_int)

    def get_b_integral_v_chunked(self):
        # Create y and z samples for calculating the integral
        y_z = np.random.uniform(self.y_domain[0], self.y_domain[1], size=(self.int_n, 2)).astype(np.float32)
        kern_vals = np.diag(self._kernel(y_z[:, 0], y_z[:, 1])).astype(np.float32)

        # Chunking init_data processing
        chunk_size = self.chunk_size
        n_total = len(self.init_data) - 1
        n = self.int_n
        b_aggregated = np.zeros((n_total, n_total), dtype=np.float32)  # Placeholder for aggregated results

        for start_idx_i in range(0, n_total, chunk_size):
            end_idx_i = min(start_idx_i + chunk_size, n_total)
            b_y_chunk = self.known_b(np.asarray(self.init_data, dtype=np.float32)[start_idx_i:end_idx_i],
                                     y_z[:, 0]).astype(np.float32)
            mb_y = b_y_chunk[:, np.newaxis, :]

            for start_idx_j in range(0, n_total, chunk_size):
                end_idx_j = min(start_idx_j + chunk_size, n_total)

                b_z_chunk = self.known_b(np.asarray(self.init_data, dtype=np.float32)[start_idx_j:end_idx_j],
                                         y_z[:, 1]).astype(np.float32)
                mb_z = b_z_chunk[np.newaxis, :, :]

                b_chunk = mb_y * mb_z * kern_vals

                # Sum over the n dimension (the integral approximation) and accumulate
                b_aggregated[start_idx_i:end_idx_i, start_idx_j:end_idx_j] += np.sum(b_chunk, axis=2) / n

        self.b_mat = b_aggregated
        return b_aggregated

    # returns the variance matrix
    def get_diagonal_matrix(self):
        data_len = len(self.init_data) - 1
        d_mat = np.identity(data_len) * 1 / self.diffusion ** 2
        return d_mat

    def get_theta(self):  # theta looks like a fancy "v" with a curl on the right side (for reference to the note)
        constructor1 = np.asarray(self.init_data)[1:]
        constructor2 = np.asarray(self.init_data)[:-1]
        theta = constructor1 - constructor2
        return theta

    def optimize_c(self, matrix_calc='chunked', **kwargs):
        if matrix_calc == 'chunked':
            b_int = self.get_b_integral_v_chunked()
        elif matrix_calc == 'long':
            b_int = self.get_b_integral()
        d_mat = self.get_diagonal_matrix()
        theta = self.get_theta()
        dt = self.t_delta

        def f(c):
            part1 = dt * (c.T @ b_int.T @ d_mat @ b_int @ c)
            part2 = 2 * (theta.T @ d_mat @ b_int @ c)
            part3 = np.linalg.norm(c)
            return part1 - part2 + part3

        min_c = minimize(f, np.zeros(len(d_mat)), **kwargs)
        self.optimal_beta = min_c
        return min_c
