import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geninvgauss, gamma, invgamma
from datetime import datetime

from prior_abstractions import Gig, Shoe, Ig

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

#quick wrapper for generating data and showing histogram
def gen_data_and_hist(b, diff, **kwargs):
    init_data = create_data(diff, b, **kwargs)  # Pass kwargs directly to create_data
    plt.hist(init_data, bins=100, density=True)
    plt.show()
    return init_data

def check_start_data(b_func, diffusion, **kwargs):
    init_data = gen_data_and_hist(b_func, diffusion, **kwargs)  # Pass kwargs directly to gen_data_and_hist
    s = input('Are you happy with the way this data looks? [y/n]\n')
    while s.lower() != 'y':
        init_data = gen_data_and_hist(b_func, diffusion, **kwargs)  # Pass kwargs directly again
        s = input('Are you happy with the way this data looks? [y/n] \n')
    return init_data

def do_calcs_and_find_errors(sde_obj, name):
    if input(f'Do you want to run the gibbs process for {name}? [y/n] \n').lower() =='y':
        sde_obj.run_gibbs()
        sde_obj.plot_b_vs_bhat()
        sde_obj.find_mse()
        sde_obj.find_kolmogorov() #in this function, check if CDF exists and ask to create it if not
        mse = sde_obj.return_mse()
        kolmo = sde_obj.return_kolmogorov()
        sde_obj.data_hist("b")
        sde_obj.data_hist("bhat")
        sde_obj.plot_shrinkage()
        print(f'MSE for {name} = {mse}.\nKolmogorov Error for {name} = {kolmo}')
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M")
        np.save(f'{name}_MSE_Kolmo_{dt_string}', np.array(mse, kolmo))
    else:
        raise Exception('Gibbs process cancelled')

#code that does n runs of the gibbs process for an empty prior class, and returns the average mse/kol values from those runs
#optional param: other_info: saves the initial data, generated bhat data, generated b data, and beta_record from the last run
def avg_n(n, init_data, b, diffusion, Cls, include_other_info = True, show_shrinkage = False, m = 20000, class_params = False):
    trials = n
    mse = np.zeros(trials)
    kol = np.zeros(trials)
    for i in range(trials):
        print(f'run {i}')
        if class_params:
            sde_obj = Cls(init_data, b, diffusion, m=m, class_params = class_params)
        else:
            sde_obj = Cls(init_data, b, diffusion, m=m)
        sde_obj.run_gibbs(verbose = False)
        sde_obj.find_mse()
        sde_obj.find_kolmogorov()
        mse[i] = sde_obj.return_mse()
        kol[i] = sde_obj.return_kolmogorov()
    if show_shrinkage:
        sde_obj.plot_shrinkage()
    if include_other_info:
        return mse, kol, init_data, sde_obj.return_mdata('bhat'), sde_obj.return_mdata('b'), sde_obj.beta_record
    else:
        return mse, kol


#function for making the list of params to run through the run_all_processes function
#basically, for each b, find a diffusion coefficient and data set that looks good and add them to the list
#input is a 2d array (nx4) where each entry is [func_name,b,diffusion] (there are n number of b functions we are looking at)
def make_params(b_diff_list):
    params = []
    for func_name, b, diff, datagen_hyperparams in b_diff_list:
        init_data = gen_data_and_hist(b, diff, **datagen_hyperparams)
        while True:
            s = input('Enter "y" to use this data, "n" to make data again with this diffusion, \nor a number from 0.1 - 1.5 to make new data with diff = num: ')
            if s == 'y':
                break
            elif s == 'n':
                init_data = gen_data_and_hist(b, diff,**datagen_hyperparams)
            elif float(s) <= 1.5 and float(s) >= 0.1:
                diff = float(s)
                init_data = gen_data_and_hist(b, diff, **datagen_hyperparams)
        params.append([func_name, init_data, b, diff])
    return params

def make_params_multi(b_diff_list, num_init_data=1):
    params = []
    for func_name, b, diff, datagen_hyperparams in b_diff_list:
        init_data_versions = []  # Initialize a list to hold all versions of init_data

        for _ in range(num_init_data):  # Loop as per the specified number of init_data versions
            init_data = gen_data_and_hist(b, diff, **datagen_hyperparams)
            while True:
                s = input('Enter "y" to use this data, "n" to make data again with this diffusion, \nor a number from 0.1 - 1.5 to make new data with diff = num: ')
                if s == 'y':
                    init_data_versions.append(init_data)  # Append the approved init_data to the versions list
                    break
                elif s == 'n':
                    init_data = gen_data_and_hist(b, diff,**datagen_hyperparams)
                elif float(s) <= 1.5 and float(s) >= 0.1:
                    diff = float(s)
                    init_data = gen_data_and_hist(b, diff, **datagen_hyperparams)

        params.append([func_name, init_data_versions, b, diff])  # Append parameters along with all versions of init_data
    return params


#a natural segue from the previous function
# def run_all_processes(params, n = 5):
#     results = {}
#     grid = [(func_name, init_data, b, diffusion, Cls) for func_name, init_data, b, diffusion in params for Cls in [Gig, Ig, Shoe]]
#     for func_name, init_data, b, diffusion, Cls in grid:
#         print(f'with b = {func_name} and Cls = {Cls}...')
#         #need to write mse and kolmo values to some dataframe (dict for now) to view it
#         mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls)
#         if Cls == Ig: id = func_name + '; Ig'
#         elif Cls == Gig: id = func_name + '; Gig'
#         elif Cls == Shoe: id = func_name + '; Shoe'
#         results[id] = mse, kol, init_data, bhat_data, b_data, beta_record
#     return results


#A different version of the previous run-all_processes function - pretty janky, made in a hurry for testing out different parameters for the Gig class
# class_params is a 2d list with all of the sets of Gig parameters we want to test. Default is None, which means only testing LASSO

def run_all_processes(params, n = 5, priors = [Gig, Ig, Shoe], class_params = None):
    results = {}
    grid = [(func_name, init_data, b, diffusion, Cls) for func_name, init_data, b, diffusion in params for Cls in priors]
    for func_name, init_data, b, diffusion, Cls in grid:
        #need to write mse and kolmo values to some dataframe (dict for now) to view it
        if Cls == Ig:
            print(f'with b = {func_name} and Cls = {Cls}...')
            id = func_name + '; Ig'
            mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls)
            results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

        elif Cls == Gig: #this is where the jankiness is born; trying different GIG priors here
            if class_params:
                for p in class_params:
                    print(f'with b = {func_name} and Cls = {Cls} and params a,b,p = {p[2:]}...')
                    id = f'{func_name} ; Gig with {p[2:]}'
                    mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls, class_params = p)
                    results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

        elif Cls == Shoe:
            print(f'with b = {func_name} and Cls = {Cls}...')
            id = func_name + '; Shoe'
            mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls)
            results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

    return results

def run_all_processes_multi(params, n=5, priors=[Gig, Ig, Shoe], class_params=None):
    results = {}
    # Extend the grid to include an index for each version of init_data
    grid = [(func_name, init_data_version, index, b, diffusion, Cls)
            for func_name, init_data_list, b, diffusion in params
            for index, init_data_version in enumerate(init_data_list)  # Iterate over each version of init_data
            # for init_data_version in init_data_list
            for Cls in priors]

    for item in grid:
        func_name, init_data, init_data_index, b, diffusion, Cls = item  # Unpack the grid
        # print(item)
        # Generate a unique id that now includes the init_data index
        if Cls == Ig:
            print(f'with b = {b}, function = {func_name}, Cls = {Cls}, and init_data version = {init_data_index}...')
            id = f'{func_name}; Ig; {init_data_index}'
            mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls)
            results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

        elif Cls == Gig:  # trying different GIG priors here
            if class_params:
                for p in class_params:
                    print(f'with b = {b}, function = {func_name}, Cls = {Cls}, params a,b,p = {p[2:]}, and init_data version = {init_data_index}...')
                    id = f'{func_name}; Gig with {p[2:]}; {init_data_index}'
                    mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls, class_params=p)
                    results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

        elif Cls == Shoe:
            print(f'with b = {b}, function = {func_name}, Cls = {Cls}, and init_data version = {init_data_index}...')
            id = f'{func_name}; Shoe; {init_data_index}'
            mse, kol, init_data, bhat_data, b_data, beta_record = avg_n(n, init_data, b, diffusion, Cls)
            results[id] = mse, kol, init_data, bhat_data, b_data, beta_record

    return results
