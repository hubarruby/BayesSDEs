import matplotlib.pyplot as plt
import pickle
from scipy.stats import kurtosis, beta, truncnorm
import pandas as pd
import sympy as sp

from fredholm_datagen_utils import *
from fredholm_utils import *


def get_results_dictionaries(filename_list, filepath='fredholm_results/'):
    """
    Turns a list of .pkl filenames into the dictionaries at each address
    :param filename_list:
    :param filepath
    :return: a list of dictionaries containing data and metadata from the simulation results listed
    """
    dict_list = []
    for filename in filename_list:
        with open(filepath + filename, 'rb') as file:
            dict_list.append(pickle.load(file))

    return dict_list


def compare_betas(dict_list, beta_idx=-1, kurt=True, **kwargs):
    """
    Plots a comparison shrinkage histogram of the betas in each dictionary in the input list.

    :param dict_list: a list of dictionaries to iterate on, which were generated from the fredholm_simulation.py file
    :param beta_idx: the index in the gibbs process that the beta values will be taken from. Defaults to -1 (the final beta vals)
    :param kurtosis (bool): determines whether the kurtosis values of the betas are plotted in the legend labels.
    :param **kwargs: args to pass into the np.histogram or plt.hist functions, such as e.g. bins=50
    :return: None
    """
    betas_list = []
    for data_dict in dict_list:
        betas_list.append(data_dict['beta_record'][beta_idx])

    bins = np.histogram(np.hstack(tuple(betas_list)), bins=50)[1]

    for data_dict in dict_list:
        meta = data_dict['meta_data']
        if kurt:
            # quick check if there are any global parameters to worry about plotting in the labels
            if meta["global_gig_a"] == meta["global_gig_b"] == meta["global_gig_p"] == 0:
                plt.hist(data_dict['beta_record'][beta_idx], bins=bins, alpha=0.5,
                         label=f'Func: {data_dict["known_b"].__name__}, '
                               f'Loc: {meta["local_gig_a"]}, {meta["local_gig_b"]}, {meta["local_gig_p"]}, '
                               f'Kurt: {kurtosis(data_dict["beta_record"][beta_idx]):.1f}')
            else:
                plt.hist(data_dict['beta_record'][beta_idx], bins=bins, alpha=0.5,
                         label=f'Func: {data_dict["known_b"].__name__}, '
                               f'Loc: {meta["local_gig_a"]}, {meta["local_gig_b"]}, {meta["local_gig_p"]}, '
                               f'Glob: {meta["global_gig_a"]}, {meta["global_gig_b"]}, {meta["global_gig_p"]}, '
                               f'Kurt: {kurtosis(data_dict["beta_record"][beta_idx]):.1f}')
        else:
            if meta["global_gig_a"] == meta["global_gig_b"] == meta["global_gig_p"] == 0:
                plt.hist(data_dict['beta_record'][beta_idx], bins=bins, alpha=0.5,
                         label=f'Func: {data_dict["known_b"].__name__}, '
                               f'Loc: {meta["local_gig_a"]}, {meta["local_gig_b"]}, {meta["local_gig_p"]}')
            else:
                plt.hist(data_dict['beta_record'][beta_idx], bins=bins, alpha=0.5,
                         label=f'Func: {data_dict["known_b"].__name__}, '
                               f'Loc: {meta["local_gig_a"]}, {meta["local_gig_b"]}, {meta["local_gig_p"]}, '
                               f'Glob: {meta["global_gig_a"]}, {meta["global_gig_b"]}, {meta["global_gig_p"]}')

    plt.legend()
    plt.show()


def plot_bbar_estimates(dict_list, b_mat, beta_idx=-1, range_linspace=(-5,5), bbar_integral=True, integral_n=5000,
                        bbar_manual=None, bbar_manual_name=None, y_lim=None):
    """
    Plots the estimated b_bar using the betas in each results dictionary, comparing with the true function.
    Assumes that all data being passed in the main input list require the same b_mat (use the same init_data and b)
    :param dict_list:  a list of dictionaries to iterate on, which were generated from the fredholm_simulation.py file
    :param b_mat: calculated from fredholm_utils.estimated_b_function_matrix
    :param include_true_func: 
    :param beta_idx: 
    :param range_linspace:
    "param bbar_integral (bool): if True, this function will numerically integrate (using the b function and pi
        distribution from the final dictionary entry in dict_list) to provide a comparison for the estimated functions.
    :param bbar_manual (callable): takes in x_vals across the linspace range and outputs the true integral.
        example, if b(x, y) = (x+y)-(x+y)^3 and pi is a truncated normal with moments of 1, 11, 31 respectively:
                def bbar_manual(x_vals):
                    E_y = 1
                    E_y2 = 11
                    E_y3 = 31
                    return x_vals + E_y - x_vals**3 - (3*x_vals**2)*E_y - 3*x_vals*E_y2 - E_y3
    :param bbar_manual_name (str): legend label for the plot of bbar_manual
        Example: "x_vals + E_y - x_vals**3 - (3*x_vals**2)*E_y - 3*x_vals*E_y2 - E_y3"
    # todo :param mse (str): if 'integral', the function calculates and reports the MSE between the numerically estimated
    integral bbar_integral and the beta estimates from each item in the dict_list.
    If 'manual', the function calculates and reports the MSE between the manually computed function bbar_manual and the beta
    estimates from each item in the dict_list
    :return: 
    """

    # assumes that len(init_data) is the same across the different entires of the list
    linspace_size = len(dict_list[0]['init_data']) - 1
    x_vals=np.linspace(range_linspace[0], range_linspace[1], linspace_size)

    for data_dict in dict_list:
        meta = data_dict["meta_data"]
        # quick check if there are any global parameters to worry about plotting
        if meta["global_gig_a"] == meta["global_gig_b"] == meta["global_gig_p"] == 0:
            plt.scatter(x_vals, estimated_b_function_mat_calc(b_mat, data_dict['beta_record'][beta_idx]), alpha = 0.6,
                        s=0.5, label=f'Func: {data_dict["known_b"].__name__}, Loc: {meta["local_gig_a"]}, '
                                     f'{meta["local_gig_b"]}, {meta["local_gig_p"]}, '
                                     f'Kurt: {kurtosis(data_dict["beta_record"][beta_idx]):.1f}')
        else:
            plt.scatter(x_vals, estimated_b_function_mat_calc(b_mat, data_dict['beta_record'][beta_idx]), alpha = 0.6,
                        s=0.5, label=f'Func: {data_dict["known_b"].__name__}, Loc: {meta["local_gig_a"]}, '
                                     f'{meta["local_gig_b"]}, {meta["local_gig_p"]}, '
                              f'Glob: {meta["global_gig_a"]}, {meta["global_gig_b"]}, {meta["global_gig_p"]}, '
                              f'Kurt: {kurtosis(data_dict["beta_record"][beta_idx]):.1f}')

    if bbar_manual:
        if bbar_manual_name:
            plt.scatter(x_vals, bbar_manual(x_vals), alpha = 0.6, s=0.5, label=bbar_manual_name)
        else:
            plt.scatter(x_vals, bbar_manual(x_vals), alpha = 0.6, s=0.5, label='Manual Function Estimate')

    if bbar_integral:
        # using the last entry in dict_list to see what the distribution was for generating the data
        # careful: the above comment (and code below) implies that this part of the function assumes that the
        # distribution used to generate each of the dictionaries in dict_list was the same pi distribution.
        if meta['pi'] == 'truncnorm':
            a_trunc_real, b_trunc_real = (meta['a_trunc'] - meta['loc']) / meta['scale'], (meta['b_trunc'] - meta['loc']) / meta['scale']
            pi = truncnorm(a_trunc_real, b_trunc_real, loc=meta['loc'], scale=meta['scale'])
        elif meta['pi'] == 'beta':
            pi = beta(meta['beta_distribution_a'], meta['beta_distribution_b'])
        else:
            raise AttributeError
        integral_estimated_bbar = b_bar_v(x=x_vals, pi=pi, b=dict_list[-1]['known_b'], integral_n=integral_n)
        plt.scatter(x_vals, integral_estimated_bbar, alpha = 0.6, s=0.5,
                    label=f"Integration Estimate of {dict_list[-1]['known_b'].__name__}")
    if y_lim:
        plt.ylim(y_lim)
    plt.legend()
    plt.show()


def metadata_table(dict_list):
    """
    Takes the values in dict list and outputs a table containing the metadata of each dict list, for easy viewing
    :param dict_list:
    :return: metadata_table (pd.DataFrame)
    """
    metadata = []
    # something wth pandas to initialize the dataframe
    for data_dict in dict_list:
        md = data_dict['meta_data']
        metadata.append(md)

    # Convert the list of metadata dictionaries into a DataFrame
    metadata_df = pd.DataFrame(metadata)
    return metadata_df


def calc_moments(mgf, eval_values, orders='fst'):
    """
    Evaluates the 1st, 2nd, and/or 3rd derivatives of the MGF provided, at the parameter values passed.
    :param mgf (sympy): the moment generating function sympy object; must be compatible sympy functions (not e.g. numpy)
    Also, must contain a variable t (the variable with which the mgf if differentiated with respect to,
    and at which the derivative(s) is/are evaluated at t=0)
        Example:
            mgf = sp.exp(mu * t + 1/2*sigma**2*t**2) * (Phi(beta - sigma * t ) - Phi(alpha - sigma * t))/(Phi(beta) - Phi(alpha))
    :param orders: which orders of the moment to calculate 'f': first, 's':second, 't':third.
    :param values (dict): value names and their values to sub into the sympy object, once the derivative is calculated
        Example:
            values = {sigma: scale, beta: b_trunc_real, alpha: a_trunc_real, mu: loc}
    :return: tuple of the moments of order specified by "orders" evaluated at the points in "values"
    """
    mgf_derivative = sp.diff(mgf, t)
    mgf_second_derivative = sp.diff(mgf_derivative, t)
    mgf_third_derivative = sp.diff(mgf_second_derivative, t)

    mgf_derivative_at_vals = mgf_derivative.subs(eval_values).subs(t, 0).evalf()
    mgf_second_derivative_at_vals = mgf_second_derivative.subs(eval_values).subs(t, 0).evalf()
    mgf_third_derivative_at_vals = mgf_third_derivative.subs(eval_values).subs(t, 0).evalf()

    return dict(f=mgf_derivative_at_vals if 'f' in orders else None,
                s=mgf_second_derivative_at_vals if 's' in orders else None,
                t=mgf_third_derivative_at_vals if 't' in orders else None)