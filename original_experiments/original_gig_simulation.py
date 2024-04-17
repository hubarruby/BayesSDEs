import numpy as np
import argparse

import pickle
import os

from helpers.utils import create_data
from datagen import sin1, sin2, sin3, sin4, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, double_well
from helpers.prior_abstractions import Gig

def parse_args():
    parser = argparse.ArgumentParser(description="Run the simulation with configurable parameters.")
    parser.add_argument('--results_filepath', type=str, required=True,
                        help='filepath to save the results (should be a .pkl).')
    parser.add_argument('--diffusion', type=float, default=1, help='Diffusion coefficient.')
    parser.add_argument('--t_delta', type=float, default=0.05, help='Time delta.')
    parser.add_argument('--t_end', type=float, default=100, help='End time.')
    parser.add_argument('--start_val', type=float, default=0, help='Starting value for the simulation data generation.')
    parser.add_argument('--gibbs_iters', type=int, default=40, help='Number of Gibbs iterations.')
    parser.add_argument('--b_function', type=str, choices=['sin1', 'sin2', 'sin3', 'sin4', 'gamma1', 'gamma2', 'gamma3',
                                                           'gamma4', 'gamma5', 'gamma6', 'double_well'], default='double_well',
                        help='The b function to use in the data generation).')
    parser.add_argument('--gig_a', type=float, default=2, help='"a" parameter for the GIG distribution.')
    parser.add_argument('--gig_b', type=float, default=0, help='"b" parameter for the GIG distribution.')
    parser.add_argument('--gig_p', type=float, default=1, help='"p" parameter for the GIG distribution.')
    parser.add_argument('--kernel_name', type=str, default='gauss',
                        help='Name of the kernel function used in calculations.')

    args = parser.parse_args()
    print("Parsed Arguments:", args)
    return args


function_map = {
    'sin1': sin1,
    'sin2': sin2,
    'sin3': sin3,
    'sin4': sin4,
    'gamma1': gamma1,
    'gamma2': gamma2,
    'gamma3': gamma3,
    'gamma4': gamma4,
    'gamma5': gamma5,
    'gamma6': gamma6,
    'double_well': double_well
}


class GigSimulation:
    def __init__(self, results_filepath, diffusion=1, t_delta=0.05, t_end=100,
                 start_val=0, gibbs_iters=40, b_func=double_well, gig_a=2, gig_b=0,
                 gig_p=1, kernel_name='gauss'):
        self.diffusion = diffusion
        self.t_delta = t_delta
        self.t_end = t_end
        self.start_val = start_val
        self.gibbs_iters = gibbs_iters
        self.results_filepath = results_filepath
        self.b_func = b_func
        self.gig_a = gig_a
        self.gig_b = gig_b
        self.gig_p = gig_p
        self.kernel_name = kernel_name

    def run_simulation(self):
        np.random.seed(0)
        init_data = create_data(self.diffusion, b_func=self.b_func, t_delta=self.t_delta, t_end=self.t_end, start_val=self.start_val)

        gig_obj = Gig(init_data=init_data,
                    b_func=self.b_func,
                    diffusion=self.diffusion,
                    gibbs_iters=self.gibbs_iters,
                    gig_a=self.gig_a,  # lambda
                    gig_b=self.gig_b,  # lambda
                    gig_p=self.gig_p,  # lambda
                    kernel_name=self.kernel_name
                    )
        gig_obj.run_gibbs()

        fred_results_dict = dict(
            meta_data=dict(b_func=self.b_func, diffusion=self.diffusion, t_delta=self.t_delta, t_end=self.t_end,
                           start_val=self.start_val, gibbs_iters=self.gibbs_iters,
                           results_filepath=self.results_filepath, gig_a=self.gig_a, gig_b=self.gig_b,
                           gig_p=self.gig_p, kernel_name=self.kernel_name),
            beta_record=gig_obj.beta_record, init_data=gig_obj.init_data, diffusion=gig_obj.diffusion)

        os.makedirs(os.path.dirname(self.results_filepath), exist_ok=True)

        # Save results
        with open(self.results_filepath, 'wb') as file:
            pickle.dump(fred_results_dict, file)


def main():
    args = parse_args()

    # Assuming function_map for b_function is handled outside main or earlier in the script
    b_func = function_map[args.b_function]

    simulation = GigSimulation(
        results_filepath=args.results_filepath,
        diffusion=args.diffusion,
        t_delta=args.t_delta,
        t_end=args.t_end,
        start_val=args.start_val,
        gibbs_iters=args.gibbs_iters,
        b_func=b_func,
        gig_a=args.gig_a,
        gig_b=args.gig_b,
        gig_p=args.gig_p,
        kernel_name=args.kernel_name
    )
    simulation.run_simulation()


if __name__ == '__main__':
    main()