__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains supporting functions needed to run the mean-field network and create all the plots"""


import sys
import os
sys.path.append(os.path.dirname( __file__ ) + '/../')

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'results'))


from run_spiking import filename_spiking

from helper_functions.simulate_and_plot_rate import *

# filename_spiking = 'swr_slice_network'
filename_rate = 'swr_slice_rate'


if __name__ == '__main__':

    # ================ create rate model from spiking model and save rate parameters
    simulate_rate_bistable_softplus_changing_Vx(filename_spiking)
    # finds optimal mean membrane potential values to define rate model parameters
    analyze_Vx_results(filename_spiking)

    # ================= simulations rate model
    # FIG 5, 10
    simulate_from_spiking(filename_spiking, filename_rate, simulate_SPW_like=True, simulate_bistable=True)
