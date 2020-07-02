'''
    Calls functions for all noisy rate model simulations and plots Fig. 12
'''

import sys
import os

# Directory to store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

from helper_functions.simulate_noisy_rate import *
from helper_functions.utils_noisy_rate import *

if __name__ == '__main__':

    # Run short simulations for Fig. 13E rate model panels:
    print('\n Running short simulations for Fig. 13E ...')
    run_short_noisy_rate_extra_dpr()

    # Run short simulation for Fig. 15D rate model panels:
    print('\n\n Running short simulation for Fig. 15D ...')
    run_short_noisy_rate_facil_only()

    # Run long simulation for Fig. 12:
    print('\n\n Running long simulations for Fig. 12 ...')
    run_long_noisy_rate_spont_and_evoked()

    # Plot Fig. 12:
    print('\n\n Creating Fig. 12 ...\n')
    plot_fig_12()
