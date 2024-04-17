import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geninvgauss, gamma, invgamma

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

def create_data(diffusion, b_func, t_delta, t_end, start_val, verbose=False, **kwargs):
    """
    Simulates and returns a sequence of data points representing the evolution of a system over time, based on a specified deterministic function and a stochastic diffusion component.
    The simulation omits the initial 5% of data points to focus on the stabilized behavior of the system.

    Parameters:
    - diffusion (float): Intensity of the stochastic fluctuation component.
    - b_func (callable): Deterministic function that defines the system's evolution, accepting the current system value and optional keyword arguments.
    - t_delta (float): Time increment for each simulation step.
    - t_end (float): Total duration of the simulation.
    - start_val (float): Initial value of the system.
    - **kwargs: Additional keyword arguments passed to the deterministic function.

    Returns:
    - list of float: A list of simulated system values after the initial transient phase.
    """
    t = 0
    val = start_val
    data = []

    def create_next(val):
        return val + b_func(val, **kwargs) * t_delta + diffusion * np.random.normal(0, np.sqrt(t_delta))

    while t < t_end - t_delta:
        data.append(val)
        val = create_next(val)
        t += t_delta
        if verbose: # and (t-int(t) == 0 or t-int(t)==0.5):
            print(f'Generated data up to t = {t}/{t_end}')

    return data[len(data) // 20:] #only take the last 19/20 of the data because the first little bit is just to establish a random start point

