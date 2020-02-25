__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains functions to construct the spiking network"""

from brian2 import *
import numpy as np
import os
import sys
sys.path.append(os.path.dirname( __file__ ) + '/../')

from helper_functions.utils_spiking import Connectivity, enough_presyn_input, \
    average_firing_rate, raster_plots, unit_firing_rate, \
    plot_average_pop_rate, calculation_CV, create_connections

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), '..', 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), '..', 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'results'))


def step1_PA(filename, make_plots=False, print_fr=False):
    """Simulate the P-A subnetwork using Brian.
    Connectivity parameters are chosen for the firing rates to be P ~ 1 Hz, A ~ 12 Hz, and network to be in an AI state

    :param filename: str
        Name of spiking filename
    :param make_plots: bool
        If True, makes summary plots (rasters, average population firing rates, CV of firing)
    :param print_fr: bool
        If True, plots firing rate info over the course of the simulation

    """
    # Speeds up Brian code using C++ standalone
    # set_device('cpp_standalone', clean=True)

    simtime_PA = 10 * second

    # ============= Defining network model parameters
    NP = 8200       # Number of pyramidal cells (slice-like)
    NB = 135        # Number of PV+BC cells (B)
    NA = 50         # Number of anti-SWR cells (A)

    tau_ampa = 2.0 * ms     # Glutamatergic synaptic time constant - decay
    tau_gabaA = 4.0 * ms    # GABAergic synaptic time constant (A) - decay

    # Neuron model
    g_leak = 10.0 * nsiemens    # Leak conductance
    v_rest = -60 * mV           # Resting potential
    rev_ampa = 0 * mV           # Excitatory reversal potential
    rev_gabaA = -70 * mV        # Inhibitory (A) reversal potential
    mem_cap = 200.0 * pfarad    # Membrane capacitance
    bg_curr_on = 200 * pA       # Background current
    vthr_all = -50. * mV        # Spiking threshold
    trefr_P = 1. * ms           # Refractory period after spike (P)
    trefr_A = 1. * ms           # Refractory period after spike (A)

    eqs_neurons_PA_2d = '''
    dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA))+bg_curr)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    vthr_2d: volt
    trefr : second
    bg_curr : amp
    '''

    # create dictionary to save values for further steps
    network_params = dict(NP=NP, NB=NB, NA=NA, tau_ampa=tau_ampa, tau_gabaA=tau_gabaA,
                          simtime_PA=simtime_PA, g_leak=g_leak, v_rest=v_rest, rev_ampa=rev_ampa, rev_gabaA=rev_gabaA,
                          mem_cap=mem_cap, bg_curr=bg_curr_on, eqs_neurons_PA_2d=eqs_neurons_PA_2d)

    # Initialize neuron group
    neurons_PA_2d = NeuronGroup(NP + NA, model=eqs_neurons_PA_2d, threshold='v > vthr_2d',
                                reset='v=v_rest', refractory='trefr', method='euler')
    pop_p_2d = neurons_PA_2d[:NP]
    pop_a_2d = neurons_PA_2d[NP:]

    neurons_PA_2d.bg_curr = bg_curr_on
    neurons_PA_2d.vthr_2d = vthr_all
    pop_p_2d.trefr = trefr_P
    pop_a_2d.trefr = trefr_A

    network_params['bg_curr'] = bg_curr_on
    network_params['vthr_P'] = vthr_all
    network_params['vthr_A'] = vthr_all
    network_params['trefr_P'] = trefr_P
    network_params['trefr_A'] = trefr_A

    # ============ Connecting the network
    # P ---> P
    pc_aux = 0.01
    if enough_presyn_input(pc_aux, NP):
        c_PP = Connectivity(0.2, pc_aux, NP, NP, 'c_PP', delay=1.*ms)
        c_PP.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in P->P')

    # P ---> A
    pc_aux = 0.01
    if enough_presyn_input(pc_aux, NP):
        c_AP = Connectivity(0.2, pc_aux, NP, NA, 'c_AP', delay=1.*ms)
        c_AP.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in P->A')

    # A ---> A
    pc_aux = 0.6
    if enough_presyn_input(pc_aux, NA):
        c_AA = Connectivity(4., pc_aux, NA, NA, 'c_AA', delay=1.*ms)
        c_AA.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in A->A')

    # A ---> P
    pc_aux = 0.6
    if enough_presyn_input(pc_aux, NA):
        c_PA = Connectivity(6., pc_aux, NA, NP, 'c_PA', delay=1.*ms)
        c_PA.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in A->P')

    # Add a constant amount every time there is a pre-spike.
    con_2d_PP = Synapses(pop_p_2d, pop_p_2d, 'g_pp: 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_2d_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_2d_PP.g_pp = c_PP.g_update

    con_2d_AP = Synapses(pop_p_2d, pop_a_2d, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_2d_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_2d_AP.g_ap = c_AP.g_update

    con_2d_AA = Synapses(pop_a_2d, pop_a_2d, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_2d_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_2d_AA.g_aa = c_AA.g_update

    con_2d_PA = Synapses(pop_a_2d, pop_p_2d, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_2d_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_2d_PA.g_pa = c_PA.g_update

    # ============== Set up monitors and initial conditions
    sm_2d_p = SpikeMonitor(pop_p_2d)
    sm_2d_a = SpikeMonitor(pop_a_2d)
    FRP = PopulationRateMonitor(pop_p_2d)
    FRA = PopulationRateMonitor(pop_a_2d)

    # Set initial values to start from non-SWR state
    pop_p_2d.v = v_rest
    pop_a_2d.v = v_rest + (pop_a_2d.vthr_2d - v_rest) * np.random.rand((NA))
    neurons_PA_2d.g_ampa = 0.1 * nS * np.random.rand((NP + NA))
    neurons_PA_2d.g_gabaA = 0.1 * nS * np.random.rand((NP + NA))

    # ======== Run!
    n_PA = Network(collect())
    n_PA.run(simtime_PA)

    # ======== Check firing rates and AI state (CV large, std of population firing small)
    fr_P_sim = average_firing_rate(sm_2d_p, simtime_PA - 1 * second, simtime_PA)
    fr_A_sim = average_firing_rate(sm_2d_a, simtime_PA - 1 * second, simtime_PA)
    print('------> Step 1 completed!')
    if print_fr:
        print('P firing rate', fr_P_sim)
        print('A firing rate', fr_A_sim)

    idx = int(2 * 1e4)  # remove initial part of simulation
    pop_FR_p = FRP.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
    pop_FR_a = FRA.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
    std_p = std(pop_FR_p)
    std_a = std(pop_FR_a)
    if print_fr:
        print('Std of P firing', std_p)
        print('Std of A firing', std_a)

    CV_p, _, CV_a = calculation_CV(sm_2d_p, None, sm_2d_a)
    if not make_plots:
        plt.close()
    if print_fr:
        print('Mean CV of P cells ', np.mean(CV_p))
        print('Mean CV of A cells ', np.mean(CV_a))

    p_unit_fr = unit_firing_rate(sm_2d_p, 2 * second, simtime_PA)
    a_unit_fr = unit_firing_rate(sm_2d_a, 2 * second, simtime_PA)
    if print_fr:
        print('Mean unit firing of P cells ', mean(p_unit_fr))
        print('Mean unit firing of A cells ', mean(a_unit_fr))

    if make_plots:
        plot_average_pop_rate(FRP, None, FRA)
        xlim1 = [(simtime_PA - 0.1 * second) / ms, (simtime_PA) / ms]
        raster_plots(sm_2d_p, None, sm_2d_a, xlim1, None, None, num_time_windows=1)

    # save network parameters in a file, to use them in following steps
    filename_full = os.path.join(path_folder,  filename + '_step1.npz')

    data_to_save_step_1 = {'network_params': network_params,
                           'dic_PP': c_PP.__dict__,
                           'dic_AP': c_AP.__dict__,
                           'dic_AA': c_AA.__dict__,
                           'dic_PA': c_PA.__dict__,
                           'P_sim_PA': fr_P_sim,
                           'A_sim_PA': fr_A_sim}

    np.savez_compressed(filename_full, **data_to_save_step_1)


def step2_PB(filename, make_plots=False, print_fr=False):
    """Simulate the P-B subnetwork using Brian.
    Connectivity parameters are chosen for the firing rates to be P ~ 40 Hz, B ~ 90 Hz

    :param filename: str
        Name of spiking filename
    :param make_plots: bool
        If True, makes summary plots (rasters, average population firing rates, CV of firing)
    :param print_fr: bool
        If True, plots firing rate info over the course of the simulation

        """
    # speed up using c++ standalone
    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', clean=True)

    simtime_PB = 10 * second

    filename_full = os.path.join(path_folder, filename + '_step1.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    # ============= Defining network model parameters
    rev_gabaB = -70 * mV        # Inhibitory (B) reversal potential
    tau_gabaB = 1.5 * ms        # GABAergic synaptic time constant (B) - decay
    vthr_B = -50 * mV           # Spiking threshold B cells
    trefr_B = 1. * ms           # Refractory period after spike (B)

    network_params['rev_gabaB'] = rev_gabaB
    network_params['tau_gabaB'] = tau_gabaB
    network_params['vthr_B'] = vthr_B
    network_params['trefr_B'] = trefr_B

    # create local variables for Brian code
    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    tau_ampa = network_params['tau_ampa']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']
    NB = network_params['NB']
    NP = network_params['NP']

    eqs_neurons_PB_2d = '''
        dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaB*(v-rev_gabaB))+bg_curr)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        vthr_2d: volt
        trefr : second
        '''
    network_params['eqs_neurons_PB_2d'] = eqs_neurons_PB_2d

    # Initialize neuron group
    neurons_PB_2d = NeuronGroup(NP + NB, model=eqs_neurons_PB_2d, threshold='v > vthr_2d',
                                reset='v=v_rest', refractory='trefr', method='euler')
    pop_p_2d = neurons_PB_2d[:NP]
    pop_b_2d = neurons_PB_2d[NP:]

    pop_p_2d.vthr_2d = network_params['vthr_P']
    pop_b_2d.vthr_2d = network_params['vthr_B']
    pop_p_2d.trefr = network_params['trefr_P']
    pop_b_2d.trefr = network_params['trefr_B']

    # ============ Connecting the network
    # Extract P ---> P connectivity from step 1
    dic = info_dictionary['dic_PP'].item()
    c_PP = Connectivity(dic['g_update'], dic['prob_of_connect'],
                         dic['size_pre'], dic['size_post'],
                         dic['name'], dic['delay'])
    c_PP.rows_connect_matrix = dic['rows_connect_matrix']
    c_PP.cols_connect_matrix = dic['cols_connect_matrix']

    # P ---> B
    pc_aux = 0.2
    if enough_presyn_input(pc_aux, network_params['NP']):
        c_BP = Connectivity(0.05, pc_aux, network_params['NP'], network_params['NB'], 'c_BP', delay=1.*ms)
        c_BP.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in P->B')

    # B ---> B
    pc_aux = 0.2
    if enough_presyn_input(pc_aux, network_params['NB']):
        c_BB = Connectivity(5., pc_aux, network_params['NB'], network_params['NB'], 'c_BB', delay=1.*ms)
        c_BB.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in B->B')

    # B ---> P
    pc_aux = 0.5
    if enough_presyn_input(pc_aux, network_params['NB']):
        c_PB = Connectivity(.7, pc_aux, network_params['NB'], network_params['NP'], 'c_PB', delay=1.*ms)
        c_PB.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in B->P')

    # create connections
    con_PP = Synapses(pop_p_2d, pop_p_2d, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_BP = Synapses(pop_p_2d, pop_b_2d, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b_2d, pop_b_2d, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b_2d, pop_p_2d, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    # ============== Set up monitors and initial conditions
    sm_2d_p = SpikeMonitor(pop_p_2d)
    sm_2d_b = SpikeMonitor(pop_b_2d)
    FRP = PopulationRateMonitor(pop_p_2d)
    FRB = PopulationRateMonitor(pop_b_2d)

    # Set initial values to start from SWR state
    pop_p_2d.v = v_rest + (-50 * mV - v_rest) * np.random.rand((NP))
    pop_b_2d.v = v_rest + (-50 * mV - v_rest) * np.random.rand((NB))
    neurons_PB_2d.g_ampa = 0.1 * nS * np.random.rand((NP + NB))
    neurons_PB_2d.g_gabaB = 0.1 * nS * np.random.rand((NP + NB))

    # ======== Run!
    n_PB = Network(collect())
    n_PB.run(simtime_PB)

    # ======== Check firing rates and AI state (CV large, std of population firing small)
    fr_P_sim = average_firing_rate(sm_2d_p, simtime_PB - 1 * second, simtime_PB)
    fr_B_sim = average_firing_rate(sm_2d_b, simtime_PB - 1 * second, simtime_PB)
    print('------> Step 2 completed!')
    if print_fr:
        print('P firing rate', fr_P_sim)
        print('B firing rate', fr_B_sim)

    idx = int(2 * 1e4)  # remove initial part of simulation
    pop_FR_p = FRP.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
    pop_FR_b = FRB.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
    std_p = std(pop_FR_p)
    std_b = std(pop_FR_b)
    if print_fr:
        print('std of P firing', std_p)
        print('std of B firing', std_b)

    CV_p, CV_b, _ = calculation_CV(sm_2d_p, sm_2d_b, None)
    if not make_plots:
        plt.close()
    if print_fr:
        print('Mean CV P cells ', np.mean(CV_p))
        print('Mean CV B cells ', np.mean(CV_b))

    p_unit_fr = unit_firing_rate(sm_2d_p, 2 * second, simtime_PB)
    b_unit_fr = unit_firing_rate(sm_2d_b, 2 * second, simtime_PB)
    if print_fr:
        print('Mean unit firing P cells ', mean(p_unit_fr))
        print('Mean unit firing B cells', mean(b_unit_fr))

    if make_plots:
        plot_average_pop_rate(FRP, FRB, None)
        xlim1 = [(simtime_PB - 0.1 * second) / ms, (simtime_PB) / ms]
        raster_plots(sm_2d_p, sm_2d_b, None, xlim1, None, None, num_time_windows=1)

    # save network parameters in a file, to use them in following steps
    filename_full = os.path.join(path_folder, filename + '_step2.npz')

    data_to_save_step_2 = info_dictionary.copy()
    del data_to_save_step_2['network_params']

    data_to_save_step_2['network_params'] = network_params
    data_to_save_step_2['dic_BB'] = c_BB.__dict__
    data_to_save_step_2['dic_BP'] = c_BP.__dict__
    data_to_save_step_2['dic_PB'] = c_PB.__dict__
    data_to_save_step_2['P_sim_PB'] = fr_P_sim
    data_to_save_step_2['B_sim_PB'] = fr_B_sim

    np.savez_compressed(filename_full, **data_to_save_step_2)


def step3_add_Wab(filename, tol_convergence_FR=3.*Hz, max_A=2.*Hz, make_plots=False, print_fr=False):
    """Simulate the P-B subnetwork, to which we add the A cells, the connections P->A and A->A from step1,
    and a new connection B -> A.
    Connectivity parameters of B -> A are chosen for the firing rates of A to be < 2 Hz.
    This state corresponds to the SWR state

    :param filename: str
        Name of spiking filename
    :param tol_convergence_FR: Brian Hz
        Maximal accepted deviation of firing rate of P and B cells from simulations in step 2
    :param max_A: Brian Hz
        Maximal accepted firing rate of A cells
    :param make_plots: bool
        If True, makes summary plots (rasters, average population firing rates, CV of firing)
    :param print_fr: bool
        If True, plots firing rate info over the course of the simulation

    """
    # speed up with c++ standalone
    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', clean=True)

    simtime_full = 6 * second

    filename_full = os.path.join(path_folder, filename + '_step2.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    # ============= Defining network model parameters
    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons = '''
        dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA)+g_gabaB*(v-rev_gabaB))+bg_curr)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        vthr_3d : volt
        trefr : second
        '''
    network_params['eqs_neurons'] = eqs_neurons

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'], model=eqs_neurons,
                          threshold='v > vthr_3d',
                          reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    # ============ Connecting the network
    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_AP', 'dic_AA', 'dic_PB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']

        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux

    # B ---> A
    pc_aux = 0.2
    if enough_presyn_input(pc_aux, network_params['NB']):
        c_AB = Connectivity(4., pc_aux, network_params['NB'], network_params['NA'], 'c_AB', delay=1.*ms)    # spont
        c_AB.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in B->A')

    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    # ============== Set up monitors and initial conditions
    sm_p = SpikeMonitor(pop_p)
    sm_b = SpikeMonitor(pop_b)
    sm_a = SpikeMonitor(pop_a)
    FRP = PopulationRateMonitor(pop_p)
    FRB = PopulationRateMonitor(pop_b)
    FRA = PopulationRateMonitor(pop_a)

    # Set initial values to start from SWR state
    pop_p.v = v_rest + (network_params['vthr_P']- v_rest) * np.random.rand((network_params['NP']))
    pop_b.v = v_rest + (network_params['vthr_P'] - v_rest) * np.random.rand((network_params['NB']))
    pop_a.v = v_rest
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))

    # ======== Run!
    n_step3 = Network(collect())
    n_step3.run(simtime_full)

    # ======== Check firing rates
    fr_P = average_firing_rate(sm_p, 2 * second, simtime_full)
    fr_B = average_firing_rate(sm_b, 2 * second, simtime_full)
    fr_A = average_firing_rate(sm_a, 2 * second, simtime_full)
    print('------> Step 3 completed!')
    if print_fr:
        print('P firing rate', fr_P)
        print('B firing rate', fr_B)
        print('A firing rate', fr_A)

    # ================ Check goodness of simulation
    target_P = info_dictionary['P_sim_PB'] * Hz
    target_B = info_dictionary['B_sim_PB'] * Hz
    if print_fr:
        print('target P FR: ', target_P)
        print('target B FR: ', target_B)
        print('max A FR: ', max_A)

    check_target_conditions = (np.abs(fr_P - target_P) < tol_convergence_FR) * \
                          (np.abs(fr_B - target_B) < tol_convergence_FR) * (fr_A < max_A)

    if not check_target_conditions:
        print('---> Failed: FR too different from target values in 2d subnetwork')

    else:
        print('---> Target conditions fulfilled')
        idx = int(2 * 1e4)      # remove initial part of simulation
        pop_FR_p = FRP.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
        pop_FR_b = FRB.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
        std_p = std(pop_FR_p)
        std_b = std(pop_FR_b)
        if print_fr:
            print('Std of P firing', std_p)
            print('Std of B firing', std_b)

        CV_p, CV_b, CV_a = calculation_CV(sm_p, sm_b, sm_a)
        if not make_plots:
            plt.close()
        if print_fr:
            print('Mean CV of P cells ', np.mean(CV_p))
            print('Mean CV of B cells ', np.mean(CV_b))
            print('Mean CV of A cells ', np.mean(CV_a))

        if make_plots:
            plot_average_pop_rate(FRP, FRB, FRA)
            xlim1 = [(simtime_full - 0.2 * second) / ms, (simtime_full) / ms]
            raster_plots(sm_p, sm_b, sm_a, xlim1, None, None, num_time_windows=1)

        filename_full = os.path.join(path_folder, filename + '_step3.npz')
        data_to_save_step_3 = info_dictionary.copy()
        del data_to_save_step_3['network_params']
        data_to_save_step_3['network_params'] = network_params
        data_to_save_step_3['dic_AB'] = c_AB.__dict__

        np.savez_compressed(filename_full, **data_to_save_step_3)


def step4_merge_subnetworks(filename, tol_convergence_FR=3.*Hz, max_B=2.*Hz, make_plots=False, print_fr=False):
    """Simulate the full network by using networks of previous steps and adding the connection B->A.
    Connectivity parameters of A -> B are chosen for the firing rates of B to be < 2 Hz.
    This state corresponds to the non-SWR state

    :param filename: str
        Name of spiking filename
    :param tol_convergence_FR: Brian Hz
        Maximal accepted deviation of firing rate of P and A cells from simulations in step 1
    :param max_B: Brian Hz
        Maximal accepted firing rate of B cells
    :param make_plots: bool
        If True, makes summary plots (rasters, average population firing rates, CV of firing)
    :param print_fr: bool
        If True, plots firing rate info over the course of the simulation

    """
    # speed up with c++ standalone
    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', clean=True)

    simtime_full = 6 * second

    filename_full = os.path.join(path_folder, filename + '_step3.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    # ============= Defining network model parameters
    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=network_params['eqs_neurons'], threshold='v > vthr_3d',
                          reset='v=v_rest', refractory='trefr', method='euler')
    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    # ============ Connecting the network
    for dic_name in ['dic_PP', 'dic_AP', 'dic_AA', 'dic_PA', 'dic_BB', 'dic_BP', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']

        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    # A ---> B
    pc_aux = 0.6
    if enough_presyn_input(pc_aux, network_params['NA']):
        c_BA = Connectivity(7., pc_aux, network_params['NA'], network_params['NB'], 'c_BA', delay=1.*ms) # spont
        c_BA.create_connectivity_matrix()
    else:
        raise ValueError('Too few presynaptic connections in A->B')

    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    # new connection
    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    # ============== Set up monitors and initial conditions
    sm_p = SpikeMonitor(pop_p)
    sm_b = SpikeMonitor(pop_b)
    sm_a = SpikeMonitor(pop_a)
    FRP = PopulationRateMonitor(pop_p)
    FRB = PopulationRateMonitor(pop_b)
    FRA = PopulationRateMonitor(pop_a)

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_P'] - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))

    # ======== Run!
    n_step4 = Network(collect())
    n_step4.run(simtime_full)

    # ======== Check firing rates
    fr_P = average_firing_rate(sm_p, 2 * second, simtime_full)
    fr_B = average_firing_rate(sm_b, 2 * second, simtime_full)
    fr_A = average_firing_rate(sm_a, 2 * second, simtime_full)
    print('------> Step 4 completed!')
    if print_fr:
        print('P firing rate', fr_P)
        print('B firing rate', fr_B)
        print('A firing rate', fr_A)

    # ================ Check goodness of simulation
    target_P = info_dictionary['P_sim_PA'] * Hz
    target_A = info_dictionary['A_sim_PA'] * Hz
    if print_fr:
        print('target P FR: ', target_P)
        print('target A FR: ', target_A)
        print('max B FR: ', max_B)

    check_target_conditions = (np.abs(fr_P - target_P) < tol_convergence_FR) * \
                          (np.abs(fr_A - target_A) < tol_convergence_FR) * (fr_B < max_B)

    if not check_target_conditions:
        print('---> Failed: FR too different from target values in 2d subnetwork')
    else:
        print('---> Target conditions fulfilled')
        idx = int(2 * 1e4)  # remove initial part of simulation
        pop_FR_p = FRP.smooth_rate('gaussian', width=3*ms)[idx:] / Hz
        pop_FR_a = FRA.smooth_rate('gaussian', width=3*ms)[idx:] / Hz

        std_p = std(pop_FR_p)
        std_a = std(pop_FR_a)
        if print_fr:
            print('Std of P firing', std_p)
            print('Std of A firing', std_a)

        CV_p, CV_b, CV_a = calculation_CV(sm_p, sm_b, sm_a)
        if not make_plots:
            plt.close()
        if print_fr:
            print('Mean CV of P cells ', np.mean(CV_p))
            print('Mean CV of B cells ', np.mean(CV_b))
            print('Mean CV of A cells ', np.mean(CV_a))

        if make_plots:
            plot_average_pop_rate(FRP, FRB, FRA)
            xlim1 = [(simtime_full - 0.5 * second) / ms, simtime_full / ms]
            raster_plots(sm_p, sm_b, sm_a, xlim1, None, None, num_time_windows=1)

        filename_full = os.path.join(path_folder, filename + '_step4.npz')
        data_to_save_step_4 = info_dictionary.copy()
        del data_to_save_step_4['network_params']
        data_to_save_step_4['network_params'] = network_params
        data_to_save_step_4['dic_BA'] = c_BA.__dict__

        np.savez_compressed(filename_full, **data_to_save_step_4)


def step5_current_injection(filename, sigma=500*pA, curr_to_pop='B', time_with_curr=0.1*second):
    """Control step: when current is injected to one of the populations (I > 0 to P or B, I < 0 to A), the network jumps
     from the non-SWR to SWR state

    :param filename: str
        Name of spiking filename
    :param sigma: Brian pA
        Maximal absolute value of current injected
    :param curr_to_pop: str
        'P', 'B', or 'A': decides to which population the current is injected
    :param time_with_curr: Brain second
        Length of current stimulation
     """

    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', build_on_run=False)

    simtime_current = 7 * second
    filename_full = os.path.join(path_folder, filename + '_step4.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']

        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_BA':
            c_BA = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
    dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA)+g_gabaB*(v-rev_gabaB))+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    vthr_3d: volt
    extracurrent: amp
    trefr : second
    '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'], model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    sm_p = SpikeMonitor(pop_p)
    sm_b = SpikeMonitor(pop_b)
    sm_a = SpikeMonitor(pop_a)

    FRP = PopulationRateMonitor(pop_p)
    FRB = PopulationRateMonitor(pop_b)
    FRA = PopulationRateMonitor(pop_a)

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A'] - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))

    warm_up_time = 3 * second
    n_step5 = Network(collect())
    n_step5.run(warm_up_time)

    # add heterogeneous current to a population
    if curr_to_pop == 'P':
        pop_p.extracurrent = sigma * np.random.rand(network_params['NP'])
    elif curr_to_pop == 'B':
        pop_b.extracurrent = sigma * np.random.rand(network_params['NB'])
    elif curr_to_pop == 'A':
        pop_a.extracurrent = -sigma * np.random.rand(network_params['NA'])

    current_on_time = time_with_curr
    n_step5.run(current_on_time)

    # switch off current
    if curr_to_pop == 'P':
        pop_p.extracurrent = 0 * pA
    elif curr_to_pop == 'B':
        pop_b.extracurrent = 0 * pA
    elif curr_to_pop == 'A':
        pop_a.extracurrent = 0 * pA

    check_behavior_time = simtime_current - warm_up_time - current_on_time
    n_step5.run(check_behavior_time)
    # need this for cpp_standalone with multiple run calls
    # device.build(directory='output', compile=True, run=True, debug=False)

    plot_average_pop_rate(FRP, FRB, FRA)

    xlim1 = [(warm_up_time - 0.2 * second) / ms, warm_up_time / ms]
    xlim2 = [(warm_up_time) / ms, (warm_up_time + 0.2 * second) / ms]
    xlim3 = [(simtime_current - 0.2 * second) / ms, (simtime_current) / ms]
    raster_plots(sm_p, sm_b, sm_a, xlim1, xlim2, xlim3)

    print('Step 5 completed!')


def step6_spontaneous(filename, value_eta=0.18, tau_depr_AB=250.*ms):
    """Add synaptic depression to network in step 4 to see spontaneous events.
    Note: syn depression acts on the connection B-> A (c_AB), and the system is bistable with depression = 0.5.
    To incorporate the dynamics of the depression, we double the value for g_ab

    :param filename: str
        Name of spiking filename
    :param value_eta: int
        Value of synaptic depression rate
    :param tau_depr_AB: Brian second
        Time constant of synaptic depression
    """

    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', build_on_run=False)

    simtime_current = 30 * second

    filename_full = os.path.join(path_folder, filename + '_step4.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
    dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA)+g_gabaB*(v-rev_gabaB))+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    vthr_3d: volt
    extracurrent: amp
    trefr : second
    '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA
    tau_depr_PB = 0. * ms

    tau_depr = tau_depr_AB

    # ========== Double the value of g_ab to account for depression present (bistable network for e = 0.5) =========== #
    info_dictionary['dic_AB'].item()['g_update'] = info_dictionary['dic_AB'].item()['g_update'] * 2.

    # create connections
    con_PP, con_PB, con_PA, con_BP, con_BB, con_BA, con_AP, con_AB, con_AA = create_connections(
        info_dictionary, pop_p, pop_b, pop_a, tau_depr, tau_depr_PB, use_dic_connect=False, depressing_PB=False)

    # ========================================= Simulation starting! ======================================== #
    FRP = PopulationRateMonitor(pop_p)
    FRA = PopulationRateMonitor(pop_a)
    FRB = PopulationRateMonitor(pop_b)

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (-50. * mV - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    eta_ab = 0.
    n_step6 = Network(collect())
    n_step6.run(1 * second)  # no synaptic depression yet

    eta_ab = value_eta

    n_step6.run(simtime_current - 1 * second)
    # need this for cpp_standalone with multiple run calls
    # device.build(directory='output', compile=True, run=True, debug=False)

    plot_average_pop_rate(FRP, FRB, FRA, my_width=3*ms)

    # Save doubled g_ab and depression variables in a file.
    # Use step 6 as a starting point for all following plots
    network_params['tau_depr_AB'] = tau_depr_AB
    network_params['eta_AB'] = value_eta
    filename_full = os.path.join(path_folder, filename + '_step6.npz')
    data_to_save_step_6 = info_dictionary.copy()
    del data_to_save_step_6['network_params']
    data_to_save_step_6['network_params'] = network_params
    data_to_save_step_6['dic_AB'] = info_dictionary['dic_AB'].item()

    np.savez_compressed(filename_full, **data_to_save_step_6)
    print('Step 6 completed! The network has been created')
