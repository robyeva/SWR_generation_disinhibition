__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains supporting functions needed to run the mean-field network and create all the plots"""


from spiking_model.run_spiking import filename_spiking
from rate_model.simulate_and_plot_rate import *
import os

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), '..', 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), '..', 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results'))

filename_rate = 'swr_slice_rate'


if __name__ == '__main__':

    # ================ create rate model from spiking model and save rate parameters
    simulate_rate_bistable_softplus_changing_Vx(filename_spiking)
    # finds optimal mean membrane potential values to define rate model parameters
    analyze_Vx_results(filename_spiking)

    # ================= simulations rate model
    # FIG 2-2, 6-2
    simulate_from_spiking(filename_spiking, filename_rate, simulate_SPW_like=True, simulate_bistable=True)
