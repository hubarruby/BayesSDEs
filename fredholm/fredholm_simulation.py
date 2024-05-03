import argparse
import pickle
import os

from helpers.utils import str2bool, construct_pi
from .fredholm_datagen_utils import *
from .fredholm_utils import FredholmGlobLoc


def parse_args():
    """
    Parse the command-line arguments for the Fredholm simulation.

    This function sets up and parses command-line arguments necessary for configuring the simulation.
    The results, including various parameters of the distribution and simulation settings,
    will be saved to a specified file path.

    Returns:
        argparse.Namespace: An object that contains the arguments and their values as attributes.
    """
    parser = argparse.ArgumentParser(description="Run the Fredholm simulation with configurable parameters.")
    parser.add_argument('--results_filepath', type=str, required=True,
                        help='filepath to save the results (should be a .pkl).')
    parser.add_argument('--pi_name', type=str, default='truncnorm -31.94 31.30 loc=1 scale=3.16',
                        help='The scipy.stats distribution name, and its parameters, all as a string separated '
                             'by spaces. Default is a truncated normal with peak at 1 and variance 10, truncated at '
                             '-100 and 100.')
    parser.add_argument('--diffusion', type=float, default=1, help='Diffusion coefficient.')
    parser.add_argument('--t_delta', type=float, default=0.05, help='Time delta.')
    parser.add_argument('--t_end', type=float, default=100, help='End time.')
    parser.add_argument('--start_val', type=float, default=0, help='Starting value for the simulation data generation.')
    parser.add_argument('--gibbs_iters', type=int, default=40, help='Number of Gibbs iterations.')
    parser.add_argument('--chunk_size', type=int, default=150, help='Chunk size for the B integral calculation.')
    parser.add_argument('--integral_n', type=int, default=5000, help='Number of integral points.')
    parser.add_argument('--y_domain', type=float, nargs=2, default=[-1000, 1000],
                        help='Domain for the Y variable as low, high.')
    parser.add_argument('--b_function', type=str, choices=['b1', 'b2', 'b_gamma1', 'b_gamma2'],
                        default='b1', help='The b function to use in the simulation (b1, b2, b_gamma1, b_gamma2).')
    parser.add_argument('--local_gig_a', type=float, default=2, help='"a" parameter for the local GIG distribution.')
    parser.add_argument('--local_gig_b', type=float, default=0, help='"b" parameter for the local GIG distribution.')
    parser.add_argument('--local_gig_p', type=float, default=1, help='"p" parameter for the local GIG distribution.')
    parser.add_argument('--global_gig_a', type=float, default=2, help='"a" parameter for the global GIG distribution.')
    parser.add_argument('--global_gig_b', type=float, default=0, help='"b" parameter for the global GIG distribution.')
    parser.add_argument('--global_gig_p', type=float, default=1, help='"p" parameter for the global GIG distribution.')
    parser.add_argument('--kernel_name', type=str, default='gauss', help='Name of the kernel function '
                        'used in calculations. Can be "gauss", "poly", "laplace, or "mult_exp"')
    parser.add_argument('--timer', type=str2bool, default=True,
                        help='Whether to display time estimates throughout the simulation process.')
    parser.add_argument('--matrix_calc', type=str, default='scipy',
                        help='Method for calculating the numeric integration of b. '
                             'Can be "chunked", "scipy", or "long".')
    parser.add_argument('--optimization_lambda', type=float, default=1,
                        help='lambda value for calculating optimal c')
    parser.add_argument('--run_gibbs', type=str2bool, default=True,
                        help='whether or not to run the gibbs_process or only manually calculate the optimal betas.')

    args = parser.parse_args()
    print("Parsed Arguments:", args)
    # Convert y_domain from string to tuple of integers
    args.y_domain = tuple(args.y_domain)
    return args


function_map = {
    'b1': b1,
    'b2': b2,
    'b_gamma1': b_gamma1,
    'b_gamma2': b_gamma2
}


class FredSimulation:
    """
    Class to create training data and run the gibbs learning process using the Fredholm approach.

    Attributes:
        results_filepath (str): Path where the simulation results will be saved.
        pi_name (str): The scipy.stats distribution name, and its parameters, all as a string separated by spaces.
        diffusion (float): Diffusion coefficient of the SDE.
        t_delta (float): Time delta for simulation steps.
        t_end (float): Ending time for the simulation.
        start_val (float): Initial value for the simulation.
        gibbs_iters (int): Number of Gibbs iterations to perform.
        chunk_size (int): Size of chunks for B integral calculation, prior to running the Gibbs sampling.
        integral_n (int): Number of points for numerical integration.
        y_domain (tuple): Domain of the Y variable as a tuple (low, high).
        b (callable): Function to be used in the simulation of data and the learning algorithm
        (b1, b2, b_gamma1, or b_gamma2).
        a_loc (float): 'a' parameter for local GIG distribution on the learned parameters.
        b_loc (float): 'b' parameter for local GIG distribution on the learned parameters.
        p_loc (float): 'p' parameter for local GIG distribution on the learned parameters.
        a_glob (float): 'a' parameter for global GIG distribution on the learned parameters.
        b_glob (float): 'b' parameter for global GIG distribution on the learned parameters.
        p_glob (float): 'p' parameter for global GIG distribution on the learned parameters.
        kernel_name (str): Name of the kernel function used.
        timer (bool): Whether to display time estimates throughout the simulation
        matrix_calc (str): method for calculation the numeric integration of b.
            Can be 'chunked', 'long', or 'scipy'.
            Use chunked for faster and less precise b matrix integration estimates,
            and scipy for more-precise and slower estimates.
            Using 'long' is not recommended (both imprecise and costly).

    Methods:
        run_simulation: Executes the simulation and saves the results.
    """
    def __init__(self, results_filepath, pi_name='truncnorm -31.94 31.30 loc=1 scale=3.16', diffusion=1, t_delta=0.05,
                 t_end=100, start_val=0, gibbs_iters=40, chunk_size=150, integral_n=5000, y_domain=(-100, 100), b=b1,
                 a_loc=2, b_loc=0, p_loc=1, a_glob=2, b_glob=0, p_glob=1, kernel_name='gauss', timer=True,
                 matrix_calc='scipy', optimization_lambda=1, run_gibbs=True):
        """
        Initialize a simulation instance with specified parameters for running Fredholm SDE learning simulations.

        Parameters:
            results_filepath (str): The file path where the simulation results will be stored.
            pi_name (str): The scipy.stats distribution name, and its parameters, all as a string separated by spaces.
            diffusion (float): The diffusion coefficient of the SDE data.
            t_delta (float): The time delta or step size for the SDE.
            t_end (float): The end time of the generated data.
            start_val (float): The starting value from which the generated data will begin.
            gibbs_iters (int): The number of Gibbs sampling iterations to perform during the learning process.
            chunk_size (int): The size of chunks used for calculating the B integral in parts, prior to the gibbs
            process.
            integral_n (int): The number of integral points used in numerical integration methods.
            y_domain (tuple of float): The range of the Y variable in the simulation, specified as (low, high).
            b (callable): The function defining the behavior over time, selected from predefined functions.
            a_loc (float): The 'a' parameter for the local Generalized Inverse Gaussian (GIG) distribution.
            b_loc (float): The 'b' parameter for the local GIG distribution.
            p_loc (float): The 'p' parameter for the local GIG distribution.
            a_glob (float): The 'a' parameter for the global GIG distribution.
            b_glob (float): The 'b' parameter for the global GIG distribution.
            p_glob (float): The 'p' parameter for the global GIG distribution.
            kernel_name (str): The name of the kernel function used in the simulation's calculations.
            timer (bool): Whether to display time estimates throughout the simulation.
            matrix_calc (str): Can be 'chunked', 'long', or 'scipy'.
                Use chunked for faster and less precise b matrix integration estimates,
                and scipy for more-precise and slower estimates.
                Using 'long' is not recommended (both imprecise and costly).

        Sets up the Fredholm learning simulation environment based on the provided parameters, preparing for a run that
        can be executed using the method `run_simulation`.
        """
        self.pi_name = pi_name
        self.diffusion = diffusion
        self.t_delta = t_delta
        self.t_end = t_end
        self.start_val = start_val
        self.gibbs_iters = gibbs_iters
        self.chunk_size = chunk_size
        self.integral_n = integral_n
        self.y_domain = y_domain
        self.results_filepath = results_filepath
        self.b = b
        self._local_gig_a = a_loc  # desired low shrinkage
        self._local_gig_b = b_loc
        self._local_gig_p = p_loc
        self._global_gig_a = a_glob  # desired high shrinkage
        self._global_gig_b = b_glob
        self._global_gig_p = p_glob
        self.kernel_name = kernel_name
        self.timer = timer
        self.matrix_calc = matrix_calc
        self.optimization_lambda = optimization_lambda
        self.run_gibbs = run_gibbs

    def run_simulation(self):
        """
        Run the Fredholm simulation using the initialized settings and parameters.

        This method generates the necessary data, performs the gibbs learning process through the
        FredholmGlobLoc object, and finally, saves the results to the specified file path.
        """
        np.random.seed(0)
        pi = construct_pi(self.pi_name)

        init_data = create_data(self.diffusion, b_bar, t_delta=self.t_delta, t_end=self.t_end, start_val=self.start_val,
                                pi=pi, b=self.b, integral_n=self.integral_n)

        fred_glob_loc_obj = FredholmGlobLoc(init_data=init_data,
                                            known_b=self.b,
                                            diffusion=self.diffusion,
                                            y_domain=self.y_domain,
                                            gibbs_iters=self.gibbs_iters,
                                            chunk_size=self.chunk_size,
                                            a_loc=self._local_gig_a,  # lambda
                                            b_loc=self._local_gig_b,  # lambda
                                            p_loc=self._local_gig_p,  # lambda
                                            a_glob=self._global_gig_a,  # tau
                                            b_glob=self._global_gig_b,  # tau
                                            p_glob=self._global_gig_p,  # tau
                                            kernel_name=self.kernel_name,
                                            timer=self.timer,
                                            matrix_calc_method=self.matrix_calc,
                                            optimization_lambda=self.optimization_lambda
                                            )

        if self.run_gibbs:
            fred_glob_loc_obj.run_gibbs()

        fred_results_dict = dict(
            meta_data=dict(pi_name=self.pi_name, diffusion=self.diffusion, t_delta=self.t_delta, t_end=self.t_end,
                           start_val=self.start_val, gibbs_iters=self.gibbs_iters, chunk_size=self.chunk_size,
                           integral_n=self.integral_n, y_domain=self.y_domain, results_filepath=self.results_filepath,
                           local_gig_a=self._local_gig_a, local_gig_b=self._local_gig_b, local_gig_p=self._local_gig_p,
                           global_gig_a=self._global_gig_a, global_gig_b=self._global_gig_b,
                           global_gig_p=self._global_gig_p, kernel_name=self.kernel_name, timer=self.timer,
                           matrix_calc_method=self.matrix_calc, optimization_lambda=self.optimization_lambda),
            beta_record=fred_glob_loc_obj.beta_record, y_domain=fred_glob_loc_obj.y_domain,
            init_data=fred_glob_loc_obj.init_data, known_b=fred_glob_loc_obj.known_b,
            diffusion=fred_glob_loc_obj.diffusion, b_mat=fred_glob_loc_obj.b_mat,
            lambda_record=fred_glob_loc_obj.lambda_mat_record, optimal_beta=fred_glob_loc_obj.optimal_beta)

        os.makedirs(os.path.dirname(self.results_filepath), exist_ok=True)

        # Save results
        with open(self.results_filepath, 'wb') as file:
            pickle.dump(fred_results_dict, file)


def main():
    args = parse_args()

    # Assuming function_map for b_function is handled outside main or earlier in the script
    b_func = function_map[args.b_function]

    simulation = FredSimulation(
        results_filepath=args.results_filepath,
        pi_name=args.pi_name,
        diffusion=args.diffusion,
        t_delta=args.t_delta,
        t_end=args.t_end,
        start_val=args.start_val,
        gibbs_iters=args.gibbs_iters,
        chunk_size=args.chunk_size,
        integral_n=args.integral_n,
        y_domain=args.y_domain,
        b=b_func,
        a_loc=args.local_gig_a,
        b_loc=args.local_gig_b,
        p_loc=args.local_gig_p,
        a_glob=args.global_gig_a,
        b_glob=args.global_gig_b,
        p_glob=args.global_gig_p,
        kernel_name=args.kernel_name,
        timer=args.timer,
        matrix_calc=args.matrix_calc,
        optimization_lambda=args.optimization_lambda,
        run_gibbs=args.run_gibbs
    )
    simulation.run_simulation()


if __name__ == '__main__':
    main()
# %%
