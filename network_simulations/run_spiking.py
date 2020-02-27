__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains all instruction to create spiking network and reproduce related figures in the manuscript.

Note: randomness is involved in network creation, so there might be some deviation from the values reported in the 
manuscript."""

import sys
import os
sys.path.append(os.path.dirname( __file__ ) + '/../')

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

from helper_functions.construct_spiking_network import *
from helper_functions.simulate_spiking import *
from helper_functions.figures_spiking import *

filename_spiking = 'swr_slice_network'


if __name__ == '__main__':

    seed(33)  # Brian seed

    # =============== 1 - Construct the network ========== #
    print('Constructing the spiking network..')
    step1_PA(filename_spiking, make_plots=False)

    step2_PB(filename_spiking, make_plots=False)

    step3_add_Wab(filename_spiking, make_plots=False)
    step4_merge_subnetworks(filename_spiking, make_plots=False)
    step5_current_injection(filename_spiking, curr_to_pop='A')    # -> this is more of a check up, but can be skipped
    step6_spontaneous(filename_spiking)

    # ================ 2 - Simulations and figures default spiking network ========== #
    
    # FIG 2
    print('Simulating network for Fig. 2')
    simulate_extended_syn_depression(filename_spiking, fraction_stim=1.)
    fig_bistability_manuscript(filename_spiking)

    # FIG 2-1 --> creates file _intermediate_fI.npz needed for the rate model definition
    print('Simulating network for Fig. 2-1')
    IF_curves_copied_neuron_ALLatonce(filename_spiking, excited_state=False)
    IF_curves_copied_neuron_ALLatonce(filename_spiking, excited_state=True)
    adds_on_fig_2(filename_spiking)

    # FIG 6
    print('Simulating network for Fig. 6')
    save_sim_all_current_for_fig(filename_spiking)
    figure_6(filename_spiking)

    # FIG 6-1
    print('Creating Fig. 6-1')
    # uses sims created in Fig. 6
    adds_on_fig_6(filename_spiking)

    # FIG 7
    print('Simulating network for Fig. 7')
    long_spontaneous_simulations(filename_spiking)
    simulate_multiple_evoked_SPW(filename_spiking)
    figure_7(filename_spiking)

    # FIG 8
    print('Simulating network for Fig. 8')
    long_spontaneous_simulations(filename_spiking, depressing_PB=True)
    # shorter sims are needed for plotting purposes - panel B with spikes
    long_spontaneous_simulations(filename_spiking, simtime_current=30*second, save_spikes=True)
    long_spontaneous_simulations(filename_spiking, simtime_current=30*second, save_spikes=True, depressing_PB=True)
    compare_default_to_other_plasticities(filename_spiking, depr_compare=True)
    
    # FIG 9
    print('Simulating network for Fig. 9')
    simulate_PtoA_facilitation_spontaneous(filename_spiking)
    # shorter sims are needed for plotting purposes - panel B with spikes
    simulate_PtoA_facilitation_spontaneous(filename_spiking, simtime_current=30*second, save_spikes=True)
    compare_default_to_other_plasticities(filename_spiking, depr_compare=False)

    # FIG 10
    print('Simulating network for Fig. 10')
    simulate_PtoA_facilitation_spontaneous(filename_spiking, BtoAdepression=False, with_norm=False, t_F=230,
                                           eta_F=0.32, gab_fac_only=4.5, gba_fac_only=5.5)
    plot_facilitationPtoA_effects(filename_spiking)






