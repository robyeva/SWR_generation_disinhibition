import sys
import os

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

from helper_functions.simulate_noisy_rate import *
from helper_functions.utils_noisy_rate import *

if __name__ == '__main__':

    # Run short simulation for Fig. 13 rate model panels:
    run_fig_13()

    # Run short simulation for Fig. 15 rate model panels:
    run_fig_15()

    # Run long simulation for Fig. 12:
    # run_fig_12()
    # Plot Fig. 12:
    # plot_fig_12()
