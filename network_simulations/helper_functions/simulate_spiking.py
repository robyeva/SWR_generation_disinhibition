__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains functions to simulate the spiking network"""

from brian2 import *
import numpy as np
import random
import os
import sys
import os
sys.path.append(os.path.dirname(__file__ ) + '/../')

from run_spiking import path_folder
from helper_functions.utils_spiking import create_connections, \
    average_firing_rate, \
    Connectivity, extract_conn_submatrix, \
    standard_neuronal_parameters_for_brian


def simulate_extended_syn_depression(filename, sigma_P=800*pA, time_with_curr=0.01*second, my_width=3*ms,
                                    compress_step=10, fraction_stim=0.6):
    """Save simulation where you combine the effect of synaptic depression and current injection to switch between SWR
    and non-SWR states

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    warm_up_time = 1.5 * second
    simtime_current = time_with_curr * 2. + warm_up_time + 4 * second

    filename_full = os.path.join(path_folder, filename + '_step6.npz')

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
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
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

    # Plastic B --> A
    # We use the clamp option to clamp x at intermediate value and see bistability in spontaneous network
    clamp_depr = 0.
    eqs_std_AB = '''
       g_ab : 1
       dx / dt = clamp_depr * (1. - x) / tau_depr_AB : 1 (clock-driven)
       '''

    con_AB = Synapses(pop_b, pop_a, model=eqs_std_AB,
                      on_pre='''g_gabaB += x*g_ab*nS
                            x = clip(x - x * eta_ab, 0, 1)
                             ''', delay=c_AB.delay, method='exact')
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update
    con_AB.x = 0.5
    tau_depr_AB = network_params['tau_depr_AB']

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A']- v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    FRP = PopulationRateMonitor(pop_p)
    FRB = PopulationRateMonitor(pop_b)
    FRA = PopulationRateMonitor(pop_a)

    # monitor synaptic depression
    monitor_depr = StateMonitor(con_AB, 'x', record=True)
    monitor_extracurr = StateMonitor(pop_p, 'extracurrent', record=True)

    curr_to_pop = 'P'
    eta_ab = 0.
    clamp_depr = 0.

    # - 1 - warm up, stays in non-SWR
    n_bist = Network(collect())
    n_bist.run(warm_up_time)

    # - 2 - inject positive current to P
    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = sigma_P * np.random.rand(idx_p)
    current_on_time = time_with_curr
    n_bist.run(current_on_time)

    # switch off current
    neurons.extracurrent = 0 * pA
    # - 3 - run in SWR state
    n_bist.run(1 * second)

    # - 4 - inject negative current to P
    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = -sigma_P * np.random.rand(idx_p)
    current_on_time = time_with_curr
    n_bist.run(current_on_time)
    # switch off current
    neurons.extracurrent = 0 * pA

    # - 5 - run in non-SWR state
    n_bist.run(1 * second)

    # - 6 - activate depression to high value
    con_AB.x = 0.8
    n_bist.run(1 * second)

    # - 7 - lower depression to go to outside
    con_AB.x = 0.2
    n_bist.run(1 * second)

    t_array = FRP.t[::compress_step] / ms
    p_array = np.array(FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
    b_array = np.array(FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
    a_array = np.array(FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
    mean_depr_array = np.mean(np.array(monitor_depr.x)[:, ::compress_step], axis=0)
    extracurr_array = np.mean(np.array(monitor_extracurr.extracurrent)[:, ::compress_step],
                              axis=0) * 1e12  # for units to be pA

    dic_to_save = {'t_array': t_array,
                   'p_array': p_array,
                   'b_array': b_array,
                   'a_array': a_array,
                   'my_width_smooth': my_width / ms,
                   'warm_up_time': warm_up_time / ms,
                   'simtime_current': simtime_current / ms,
                   'time_with_curr': time_with_curr / ms,
                   'sigma_P': sigma_P / pA,
                   'mean_depr_array': mean_depr_array,
                   'extracurr_array': extracurr_array,
                   'sim_dt_in_seconds': defaultclock.dt / second,
                   'compress_step_indices': compress_step,
                   }

    filename_to_save = os.path.join(path_folder, filename + 'curr_and_depr_bistability' + '.npz')
    np.savez_compressed(filename_to_save, **dic_to_save)


def save_sim_all_current_for_fig(filename, sigma_P=400*pA, sigma_B=200*pA, sigma_A=500*pA, time_with_curr=0.01 * second,
                                 my_width=3 * ms, compress_step=10, fraction_stim=.6):
    """Save simulations with spontaneous events and evoked SWRs by stimulation of P, B, or A cells.
    Needed to produce Fig. 9

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param sigma_B: Brian amp
        Maximal absolute value of current injected to B cells
    :param sigma_A: Brian amp
        Maximal absolute value of current injected to A cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    simtime_current = 1.5 * second

    filename_full = os.path.join(path_folder, filename + '_step6.npz')
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

    warm_up_time = 1. * second

    eqs_neurons_current = '''
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
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

    tau_depr_AB = network_params['tau_depr_AB']
    tau_depr_PB = 0 * ms

    # create connections
    con_PP, con_PB, con_PA, con_BP, con_BB, con_BA, con_AP, con_AB, con_AA = create_connections(
        info_dictionary, pop_p, pop_b, pop_a, tau_depr_AB, tau_depr_PB, use_dic_connect=False, depressing_PB=False)

    con_AB.x = 1.

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (pop_a.vthr_3d - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params['NA']))

    sm_p = SpikeMonitor(pop_p)
    sm_b = SpikeMonitor(pop_b)
    sm_a = SpikeMonitor(pop_a)

    FRP = PopulationRateMonitor(pop_p)
    FRB = PopulationRateMonitor(pop_b)
    FRA = PopulationRateMonitor(pop_a)

    # monitor synaptic depression
    monitor_depr = StateMonitor(con_AB, 'x', record=True)
    # monitor inhibitory input (to use as an approximation for LFP)
    B_inp_p = StateMonitor(pop_p, 'input_gabaB', record=True)
    # needed in Fig on LFP approximation
    A_inp_p = StateMonitor(pop_p, 'input_gabaA', record=True)
    P_inp_p = StateMonitor(pop_p, 'input_ampa', record=True)

    net_3 = Network(collect())
    net_3.store()

    # to store across simulations
    p_array = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    b_array = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    a_array = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    mean_depr_array = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    mean_inh_input_b = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    mean_inh_input_a = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    mean_exc_input_p = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))

    t_array = np.empty((3, int(simtime_current / (compress_step * defaultclock.dt))))
    spikes_dic = {}

    figure(figsize=[12, 8])

    for idx_c, curr_to_pop in enumerate(['P', 'B', 'A']):

        net_3.restore()

        eta_ab = network_params['eta_AB']
        net_3.run(warm_up_time)

        if curr_to_pop == 'P':
            idx_p = int(fraction_stim * network_params['NP'])
            pop_p[0:idx_p].extracurrent = sigma_P * np.random.rand(idx_p)
        elif curr_to_pop == 'B':
            idx_b = int(fraction_stim * network_params['NB'])
            pop_b[0:idx_b].extracurrent = sigma_B * np.random.rand(idx_b)
        elif curr_to_pop == 'A':
            idx_a = int(fraction_stim * network_params['NA'])
            pop_a[0:idx_a].extracurrent = -sigma_A * np.random.rand(idx_a)

        current_on_time = time_with_curr
        net_3.run(current_on_time)

        neurons.extracurrent = 0 * pA

        check_behavior_time = simtime_current - warm_up_time - current_on_time
        net_3.run(check_behavior_time)

        t_array[idx_c, :] = FRP.t[::compress_step] / ms
        p_array[idx_c, :] = np.array(FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
        b_array[idx_c, :] = np.array(FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
        a_array[idx_c, :] = np.array(FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz)
        spikes_dic['P_' + str(idx_c)] = array(sm_p.it)
        spikes_dic['B_' + str(idx_c)] = array(sm_b.it)
        spikes_dic['A_' + str(idx_c)] = array(sm_a.it)
        if compress_step > 1:
            spikes_dic['P_' + str(idx_c)][1, :] = np.around(spikes_dic['P_' + str(idx_c)][1, :], decimals=3)
            spikes_dic['B_' + str(idx_c)][1, :] = np.around(spikes_dic['B_' + str(idx_c)][1, :], decimals=3)
            spikes_dic['A_' + str(idx_c)][1, :] = np.around(spikes_dic['A_' + str(idx_c)][1, :], decimals=3)
        mean_depr_array[idx_c, :] = np.mean(np.array(monitor_depr.x), axis=0)[::compress_step]
        mean_inh_input_b[idx_c, :] = np.mean(np.array(B_inp_p.input_gabaB), axis=0)[::compress_step]
        mean_inh_input_a[idx_c, :] = np.mean(np.array(A_inp_p.input_gabaA), axis=0)[::compress_step]
        mean_exc_input_p[idx_c, :] = np.mean(np.array(P_inp_p.input_ampa), axis=0)[::compress_step]

        # Rough plot to get a feeling of the behavior (expected is a spontaneous event and an induced SWR in
        # correspondence of the yellow area)
        ax = subplot(4, 3, 1 + idx_c)
        plt.plot(t_array[idx_c, :], p_array[idx_c, :], '#ef3b53', label='P', lw=2.5)
        ix = np.linspace(warm_up_time / ms, (warm_up_time / ms + time_with_curr / ms))
        iy = np.linspace(10000, 10000)
        verts = [(warm_up_time / ms, 0)] + list(zip(ix, iy)) + [
            ((warm_up_time / ms + time_with_curr / ms), 0)]
        poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.3)
        ax.add_patch(poly)

        ax = subplot(4, 3, 4 + idx_c)
        plt.plot(t_array[idx_c, :], b_array[idx_c, :], '#3c3fef', label='B', lw=2.5)
        ix = np.linspace(warm_up_time / ms, (warm_up_time / ms + time_with_curr / ms))
        iy = np.linspace(10000, 10000)
        verts = [(warm_up_time / ms, 0)] + list(zip(ix, iy)) + [
            ((warm_up_time / ms + time_with_curr / ms), 0)]
        poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.3)
        ax.add_patch(poly)

        ax = subplot(4, 3, 7 + idx_c)
        plt.plot(t_array[idx_c, :], a_array[idx_c, :], '#0a9045', label='A', lw=2.5)
        ix = np.linspace(warm_up_time / ms, (warm_up_time / ms + time_with_curr / ms))
        iy = np.linspace(10000, 10000)
        verts = [(warm_up_time / ms, 0)] + list(zip(ix, iy)) + [
            ((warm_up_time / ms + time_with_curr / ms), 0)]
        poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.3)
        ax.add_patch(poly)

        ax = subplot(4, 3, 10 + idx_c)
        plot(t_array[idx_c, :], mean_depr_array[idx_c, :], '#e67e22', label='d', lw=2.5)
        ylim(np.min(mean_depr_array) - 0.05, 1.05)
        ix = np.linspace(warm_up_time / ms, (warm_up_time / ms + time_with_curr / ms))
        iy = np.linspace(10000, 10000)
        verts = [(warm_up_time / ms, 0)] + list(zip(ix, iy)) + [
            ((warm_up_time / ms + time_with_curr / ms), 0)]
        poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.3)
        ax.add_patch(poly)

    dic_to_save = {'t_array': t_array,  # we only take one, are all the same
                   'p_array': p_array,
                   'b_array': b_array,
                   'a_array': a_array,
                   'spikes_dic': spikes_dic,
                   'my_width_smooth': my_width / ms,
                   'mean_depr_array': mean_depr_array,
                   'warm_up_time': warm_up_time / ms,
                   'simtime_current': simtime_current / ms,
                   'time_with_curr': time_with_curr / ms,
                   'sim_dt_in_seconds': defaultclock.dt / second,
                   'compress_step_indices': compress_step,
                   'sigma_P': sigma_P / pA,
                   'sigma_B': sigma_B / pA,
                   'sigma_A': sigma_A / pA,
                   'mean_B_input_p': mean_inh_input_b / pA,
                   'mean_A_input_p': mean_inh_input_a / pA,
                   'mean_P_input_p': mean_exc_input_p / pA,
                   }
    filename_to_save = os.path.join(path_folder, filename + '_sim_fig9_fraction_' + str(fraction_stim) + '.npz')
    np.savez_compressed(filename_to_save, **dic_to_save)



def long_spontaneous_simulations(filename, simtime_current=10*60*second, my_width=3*ms, compress_step=10,
                                 plot_FR=False, save_input=True, save_spikes=False, depressing_PB=False,
                                 save_depression=False):
    """Save long spontaneous simulation, to study SWR dynamics. Needed to produce Fig. 11

    :param filename: str
        Name of spiking filename
    :param simtime_current: Brian second
        Length of simulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param plot_FR: bool
        If True, plot a summary of population FR
    :param save_input: bool
        If True, save the mean B input to P cells (approximated measure of the LFP)
    :param save_spikes: bool
        If True, save the single cell spikes (for raster plot)
    :param depressing_PB: bool
        If True, the B -> P connection is depressing
    :param save_depression: bool
        If True, save the mean value of the synaptic efficacy over time (for plotting purposes)
    """

    if not (save_input or save_spikes or save_depression):
        device.reinit()
        device.activate()
        set_device('cpp_standalone', build_on_run=False)
        defaultclock.dt = 0.1*ms

    filename_full = os.path.join(path_folder, filename + '_step6.npz')
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
    dv/dt=(-g_leak*(v-v_rest)-g_ampa*(v-rev_ampa)+b_input-g_gabaA*(v-rev_gabaA)+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    vthr_3d: volt
    extracurrent: amp
    trefr : second
    b_input = -g_gabaB*(v-rev_gabaB): amp
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

    tau_depr_AB = network_params['tau_depr_AB']
    if depressing_PB:
        tau_depr_PB = network_params['tau_depr_AB']
    else:
        tau_depr_PB = 0. * ms

    # create connections
    con_PP, con_PB, con_PA, con_BP, con_BB, con_BA, con_AP, con_AB, con_AA = create_connections(
        info_dictionary, pop_p, pop_b, pop_a, tau_depr_AB, tau_depr_PB, use_dic_connect=False,
        depressing_PB=depressing_PB)

    # ========================================= Simulation starting! ======================================== #
    warm_up_time = 3 * second
    check_behavior_time = simtime_current - warm_up_time

    FRP = PopulationRateMonitor(pop_p)
    FRA = PopulationRateMonitor(pop_a)
    FRB = PopulationRateMonitor(pop_b)
    if save_input:
        # only select 100 to avoid memory issues - or run on a powerful machine setting record=True
        b_input_to_p = StateMonitor(pop_p, 'b_input', record=np.random.randint(0, network_params['NP'],
                                                                               size=100))

    if save_spikes:
        sm_p = SpikeMonitor(pop_p)
        sm_b = SpikeMonitor(pop_b)
        sm_a = SpikeMonitor(pop_a)
        spikes_dic = {}

    if save_depression:
        monitor_depr = StateMonitor(con_AB, 'x', record=True)

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A'] - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    eta_ab = 0.
    if depressing_PB:
        eta_pb = 0.
    run(1 * second)

    eta_ab = network_params['eta_AB']
    if depressing_PB:
        eta_pb = eta_ab
    run(warm_up_time - 1 * second)

    # Simulation of spontaneous network
    start_spont = warm_up_time
    run(check_behavior_time, report='text')
    if not (save_input or save_spikes or save_depression):
        device.build(directory='output', compile=True, run=True, debug=False)

    if save_input:
        mean_b_input_to_p = np.mean(np.array(b_input_to_p.b_input)[:, ::compress_step], axis=0)
    if save_spikes:
        spikes_dic['P'] = array(sm_p.it)  # it's a list,[0] are neuron indices, [1] are spiking times
        spikes_dic['B'] = array(sm_b.it)
        spikes_dic['A'] = array(sm_a.it)
        # round spiking times to ms precision (instead of defaultclock.dt precision)
        if compress_step > 1:
            spikes_dic['P'][1, :] = np.around(spikes_dic['P'][1, :], decimals=3)
            spikes_dic['B'][1, :] = np.around(spikes_dic['B'][1, :], decimals=3)
            spikes_dic['A'][1, :] = np.around(spikes_dic['A'][1, :], decimals=3)
    if save_depression:
        mean_depr_array = np.mean(np.array(monitor_depr.x), axis=0)[::compress_step]

    dic_to_save = {'FRB_smooth': FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'time_array': np.asarray(FRB.t)[::compress_step] / second,
                   'FRP_smooth': FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'FRA_smooth': FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'start_spont': start_spont / second,
                   'simtime_current': simtime_current / second,
                   'sim_dt_in_seconds': defaultclock.dt / second,
                   'compress_step_indices': compress_step,
                   }
    if save_input:
        dic_to_save['mean_b_input_to_p'] = mean_b_input_to_p / pA
    if save_spikes:
        dic_to_save['spikes_dic'] = spikes_dic
    if save_depression:
        dic_to_save['mean_depr'] = mean_depr_array

    if depressing_PB:
        filename_to_save = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + \
                           '_deprPB.npz')
    else:
        filename_to_save = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + '.npz')

    np.savez_compressed(filename_to_save, **dic_to_save)

    if plot_FR:
        figure(figsize=[20, 10])
        for sub_n, my_data, my_label, my_col in zip([1, 2, 3], [FRP, FRB, FRA], ['P', 'B', 'A'], ['m', 'b', 'g']):

            subplot(3, 1, sub_n)
            if my_data is not None:
                plot(my_data.t / ms, my_data.smooth_rate('gaussian', width=my_width) / Hz, my_col, label=my_label,
                     lw=1.5)
                legend(loc='best')
                ylim(0, 100)
            if sub_n == 3:
                xlabel('Time [ms]')
                ylabel('Population rate [Hz]')
        suptitle('Smoothed instantaneous population rates')


def simulate_multiple_evoked_SPW(filename, simtime_current=10*60*second, interval_stim=2*second, my_width=3*ms,
                                 compress_step=10, plot_FR=False, save_input=True):
    """Simulation of spontaneous and evoked SWRs. Aimed at reproducing Kohus 2016, Fig 13C.
    :param filename: str
        Name of spiking filename
    :param simtime_current: Brian second
        Length of simulation
    :param interval_stim: Brian second
        Distance between two current injection (+- some jitter)
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param plot_FR: bool
        If True, plot a summary of population FR
    :param save_input: bool
        If True, save the mean B input to P cells (approximated measure of the LFP)
    """

    if not save_input:
        set_device('cpp_standalone', build_on_run=False)

    filename_full = os.path.join(path_folder, filename + '_step6.npz')
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

    # Eqs include a stimulus(t,i)
    eqs_neurons_current = '''
    dv/dt=(-g_leak*(v-v_rest)-g_ampa*(v-rev_ampa)+b_input-g_gabaA*(v-rev_gabaA)+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    vthr_3d: volt
    extracurrent = stimulus(t,i): amp
    trefr : second
    b_input = -g_gabaB*(v-rev_gabaB): amp
    '''

    num_repeat = int(simtime_current / interval_stim)
    time_with_stim = 0.01 * second
    print(num_repeat)
    Ib = 600 * pA
    fr_stim = 0.5
    idx_b = int(fr_stim * network_params['NB'])
    len_block = int(interval_stim / (time_with_stim / second))
    stim_matrix = np.zeros((len_block, network_params['NP'] + network_params['NB'] + network_params['NA'])) * pA
    stim_matrix[0, network_params['NP']:network_params['NP'] + idx_b] = Ib  # *np.random.rand(idx_b)
    full_stim = np.tile(stim_matrix, (num_repeat, 1))

    # add variability on when you stimulate
    for idx_row, my_row in enumerate(np.arange(num_repeat) * len_block):
        new_idx = my_row + np.random.randint(10)
        full_stim[[my_row, new_idx], :] = full_stim[[new_idx, my_row], :]  # swap rows to create some variability

    # find rows where at least one element is nonzero (e.g. rows of stimulations)
    stim_idx = np.where(np.any(full_stim != 0, axis=1))[0]
    for el in stim_idx:
        full_stim[el, network_params['NP']:network_params['NP'] + idx_b] = Ib * np.random.rand(idx_b)

    stim_times_array = stim_idx * time_with_stim  # times when stimulation happened
    stimulus = TimedArray(full_stim, dt=time_with_stim)

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


    tau_depr_AB = network_params['tau_depr_AB']
    tau_depr_PB = 0. * ms

    # create connections
    con_PP, con_PB, con_PA, con_BP, con_BB, con_BA, con_AP, con_AB, con_AA = create_connections(
        info_dictionary, pop_p, pop_b, pop_a, tau_depr_AB, tau_depr_PB, use_dic_connect=False, depressing_PB=False)

    # ========================================= Simulation starting! ======================================== #
    warm_up_time = 2. * second
    check_behavior_time = simtime_current - warm_up_time

    FRP = PopulationRateMonitor(pop_p)
    FRA = PopulationRateMonitor(pop_a)
    FRB = PopulationRateMonitor(pop_b)
    M = StateMonitor(pop_b, 'extracurrent', record=True)

    if save_input:
        # only select 100 to avoid memory issues, or run on powerful machine with record=True
        b_input_to_p = StateMonitor(pop_p, 'b_input', record=np.random.randint(0, network_params['NP'],
                                                                               size=100))

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A'] - v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    eta_ab = 0.
    run(1 * second)

    eta_ab = network_params['eta_AB']
    run(warm_up_time - 1 * second, report='text')

    start_spont = warm_up_time

    run(check_behavior_time, report='text')
    if not save_input:
        device.build(directory='output', compile=True, run=True, debug=False)

    if save_input:
        mean_b_input_to_p = np.mean(np.array(b_input_to_p.b_input)[:, ::compress_step], axis=0)

    dic_to_save = {'FRB_smooth': FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'time_array': FRB.t[::compress_step] / second,  # we only take one, are all the same
                   'FRP_smooth': FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'FRA_smooth': FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'start_spont': start_spont / second,
                   'simtime_current': simtime_current / second,
                   'stim_matrix': stim_matrix / pA,
                   'interval_stim': interval_stim / second,
                   'time_with_stim': time_with_stim / second,
                   'num_repeat': num_repeat,
                   'sim_dt_in_seconds': defaultclock.dt / second,
                   'compress_step_indices': compress_step,
                   'stim_times_array': stim_times_array / second
                   }
    if save_input:
        print('saving input')
        dic_to_save['mean_b_input_to_p'] = mean_b_input_to_p / pA

    filename_to_save = os.path.join(path_folder, filename + '_evoked_simtime_' + str(int(simtime_current / second)) + '.npz')
    np.savez_compressed(filename_to_save, **dic_to_save)

    if plot_FR:
        figure(figsize=[20, 10])
        for sub_n, my_data, my_label, my_col in zip([1, 2, 3], [FRP, FRB, FRA], ['P', 'B', 'A'], ['m', 'b', 'g']):

            ax = subplot(4, 1, sub_n)
            if my_data is not None:
                plot(my_data.t / ms, my_data.smooth_rate('gaussian', width=my_width) / Hz, my_col, label=my_label,
                     lw=1.5)
                legend(loc='best')
                ylim(0, 100)
            if sub_n == 3:
                xlabel('Time [ms]')
                ylabel('Population rate [Hz]')
            plot(M.t / ms, M.extracurrent[0] * 1e12)
        ax = subplot(414)
        plot(M.t / ms, M.extracurrent[0])
        plot(M.t / ms, M.extracurrent[1])
        suptitle('Smoothed instantaneous population rates')


def simulate_PtoA_facilitation_spontaneous(filename, simtime_current=10*60 * second, t_F=250, eta_F=0.15, max_z=1.,
                                           my_width=3*ms, gab_fac_only=8., gba_fac_only=7., compress_step=10,
                                           plot_FR=False, save_input=True, save_spikes=False, BtoAdepression=True,
                                           with_norm=True):
    """Simulate network with P -> A synaptic facilitation. Save simulations to plot Fig. 14 and 15
    :param filename: str
        Name of spiking filename
    :param simtime_current: Brian second
        Length of simulation
    :param t_F: int
        Time constant of facilitation
    :param eta_F: float
        Learning rate of facilitation
    :param max_z: float
        Upper bound for facilitation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param gab_fac_only: float
        Value of B-> A conductance update in case of BtoAdepression=False
    :param gba_fac_only: float
        Value of A-> B conductance update in case of BtoAdepression=False
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param plot_FR: bool
        If True, plot a summary of population FR
    :param save_input: bool
        If True, save the mean B input to P cells (approximated measure of the LFP)
    :param save_spikes: bool
        If True, save spikes for raster plot
    :param BtoAdepression: bool
        If True, simulate teh default network with P -> A facilitation in addition. Else, P-> A facilitation is the only
        plastic mechanism
    :param with_norm: bool
        If True, add normalization to the equation for P-> A faciliation. Mostly used in case with BtoAdepression=True
    """

    if not (save_input or save_spikes):
        set_device('cpp_standalone', build_on_run=False)

    filename_full = os.path.join(path_folder, filename + '_step6.npz')
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
    dv/dt=(-g_leak*(v-v_rest)-g_ampa*(v-rev_ampa)+b_input-g_gabaA*(v-rev_gabaA)+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    vthr_3d: volt
    extracurrent: amp
    trefr : second
    b_input = -g_gabaB*(v-rev_gabaB): amp
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

    tau_fac = t_F * ms
    tau_depr_PB = 0. * ms

    # create connections
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

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

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

    if BtoAdepression:
        # Plastic B-->A
        eqs_std_AB = '''
                  g_ab : 1
                  dx / dt = (1. - x) / tau_depr_AB : 1 (clock-driven)
                   '''
        con_AB = Synapses(pop_b, pop_a, model=eqs_std_AB,
                          on_pre='''g_gabaB += x*g_ab*nS
                               x = clip(x - x * eta_ab, 0, 1)
                                ''', delay=c_AB.delay, method='exact')
        con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
        con_AB.g_ab = c_AB.g_update
        con_AB.x = 1.
        tau_depr_AB = network_params['tau_depr_AB']
        eta_ab = network_params['eta_AB']

    else:
        con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
        con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
        con_AB.g_ab = c_AB.g_update

    # P --> A is plastic now - short term facilitation when spikes arrive from P to A
    eqs_std_AP = '''
    g_ap : 1
    dz / dt = (- z) / tau_fac : 1 (clock-driven)
    '''
    con_AP = Synapses(pop_p, pop_a, model=eqs_std_AP,
                      on_pre='''g_ampa += (1. + z)*g_ap*nS
                      z = z + (max_z - z) * eta_ap
                      ''', delay=c_AP.delay, method='exact')
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update
    con_AP.z = 0.  # full efficacy in the beginning of stimulation

    if with_norm:
        P0_val = 2*Hz       # network specific, is the value of P in non-SWR state
        con_AP.g_ap = c_AP.g_update / (
                1. + (tau_fac * eta_F * max_z * 2 * Hz / (1. + tau_fac * eta_F * P0_val)))
    warm_up_time = 3 * second
    check_behavior_time = simtime_current - warm_up_time

    FRP = PopulationRateMonitor(pop_p)
    FRA = PopulationRateMonitor(pop_a)
    FRB = PopulationRateMonitor(pop_b)
    if save_spikes:
        sm_p = SpikeMonitor(pop_p)
        sm_b = SpikeMonitor(pop_b)
        sm_a = SpikeMonitor(pop_a)
        spikes_dic = {}

    if save_input:
        # only select 100 to avoid memory issues or set record=True on more powerful machine
        b_input_to_p = StateMonitor(pop_p, 'b_input', record=np.random.randint(0, network_params['NP'],
                                                                               size=100))

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

    # ========================================= Simulation starting! ======================================== #
    eta_ap = 0.
    if BtoAdepression:
        eta_ab = 0.
    run(warm_up_time, report='text')  # no depression yet

    eta_ap = eta_F
    if BtoAdepression:
        eta_ab = network_params['eta_AB']
    else:
        # change values of reciprocal connections between interneurons in the P-> A facilitation ONLY case
        con_AB.g_ab = gab_fac_only
        con_BA.g_ba = gba_fac_only

    start_spont = warm_up_time
    run(check_behavior_time, report='text')
    if not (save_input or save_spikes):
        device.build(directory='output', compile=True, run=True, debug=False)

    if save_input:
        mean_b_input_to_p = np.mean(np.array(b_input_to_p.b_input)[:, ::compress_step], axis=0)
    if save_spikes:
        spikes_dic['P'] = np.array(sm_p.it)
        spikes_dic['B'] = np.array(sm_b.it)
        spikes_dic['A'] = np.array(sm_a.it)
        # round spiking times to ms precision (instead of defaultclock.dt precision)
        if compress_step > 1:
            spikes_dic['P'][1, :] = np.around(spikes_dic['P'][1, :], decimals=3)
            spikes_dic['B'][1, :] = np.around(spikes_dic['B'][1, :], decimals=3)
            spikes_dic['A'][1, :] = np.around(spikes_dic['A'][1, :], decimals=3)

    print('only saving is left to be done.. ')
    dic_to_save = {'FRB_smooth': FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'time_array': FRB.t[::compress_step] / second,
                   'FRP_smooth': FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'FRA_smooth': FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
                   'start_spont': start_spont / second,
                   'simtime_current': simtime_current / second,
                   'sim_dt_in_seconds': defaultclock.dt / second,
                   'compress_step_indices': compress_step,
                   }
    if save_input:
        dic_to_save['mean_b_input_to_p'] = mean_b_input_to_p / pA
    if save_spikes:
        dic_to_save['spikes_dic'] = spikes_dic

    if BtoAdepression:
        filename_to_save = os.path.join(path_folder, filename + '_simtime_' + str(int(simtime_current / second)) + '_tauF_' + \
                           str(int(t_F)) + '_etaF_' + str(eta_F) + '_maxz_' + str(max_z) + '_BtoAdepr.npz')

    else:
        filename_to_save = os.path.join(path_folder, filename + '_simtime_' + str(int(simtime_current / second)) + '_tauF_' + \
                           str(int(t_F)) + '_etaF_' + str(eta_F) + '_gabfac_' + str(gab_fac_only) + '_gbafac_' + str(
            gba_fac_only) + '_maxz_' + str(max_z) + '.npz')

    np.savez_compressed(filename_to_save, **dic_to_save)

    if plot_FR:
        figure(figsize=[20, 10])
        for sub_n, my_data, my_label, my_col in zip([1, 2, 3], [FRP, FRB, FRA], ['P', 'B', 'A'], ['m', 'b', 'g']):

            ax = subplot(4, 1, sub_n)
            if my_data is not None:
                plot(my_data.t / ms, my_data.smooth_rate('gaussian', width=my_width) / Hz, my_col, label=my_label,
                     lw=1.5)
                legend(loc='best')
                ylim(0, 100)
            if sub_n == 3:
                xlabel('Time [ms]')
                ylabel('Population rate [Hz]')
        suptitle('Smoothed instantaneous population rates')


def IF_curves_copied_neuron_ALLatonce(filename, excited_state=False, n_extra=50, use_dic_connect=False):
    """Compute If curves of neuron in pop by copying it so that it receives inputs from the network but its activity
    it's not fed back to the network"""

    simtime_current = 10 * second  # estimate FR with some extracurrent, to plot FR in I.f curve
    simtime_estimate_FR = 5 * second  # estimate FR before any current is injected

    if excited_state:
        jump_time = 3 * second
    else:
        jump_time = 0 * second

    # ============= 1: create default network
    filename_full = os.path.join(path_folder, filename + '_step6.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    if use_dic_connect:
        dic_list = ['W_pp', 'W_bp', 'W_bb', 'W_pa', 'W_ap', 'W_aa', 'W_ba', 'W_pb', 'W_ab']
        conn_dictionary = info_dictionary['dic_connect'].item()

    else:
        dic_list = ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']

    for dic_name in dic_list:

        if use_dic_connect:
            dic = conn_dictionary[dic_name]

            c_aux = Connectivity(dic.g_update / nS, dic.prob_of_connect,
                                 dic.size_pre, dic.size_post,
                                 dic.name, dic.delay)
            c_aux.rows_connect_matrix = dic.rows_connect_matrix
            c_aux.cols_connect_matrix = dic.cols_connect_matrix

            if dic_name == 'W_pp':
                c_PP = c_aux
            if dic_name == 'W_bp':
                c_BP = c_aux
            if dic_name == 'W_bb':
                c_BB = c_aux
            if dic_name == 'W_pa':
                c_PA = c_aux
            if dic_name == 'W_ap':
                c_AP = c_aux
            if dic_name == 'W_aa':
                c_AA = c_aux
            if dic_name == 'W_pb':
                c_PB = c_aux
            if dic_name == 'W_ba':
                c_BA = c_aux
            if dic_name == 'W_ab':
                c_AB = c_aux

        else:
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
    NP = network_params['NP']
    NB = network_params['NB']
    NA = network_params['NA']

    # net equations
    eqs_neurons_current = '''
    dv/dt = (-g_leak*(v-v_rest)+I_input+extracurrent)/mem_cap : volt (unless refractory)
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
    dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
    vthr_3d: volt
    extracurrent: amp
    trefr: second
    background_curr: amp
    I_input = -(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA)+g_gabaB*(v-rev_gabaB))+background_curr : amp
    '''

    # percentage option: note: need to be integer number resulting, or problems may arise
    # percent_to_save = 5    # decide for how many neurons the IF curves should be computed
    # num_additional_neurons = percent_to_save*(NP+NB+NA)
    n_to_save_each_pop = n_extra
    num_additional_neurons = n_to_save_each_pop * 3

    # Initialize neuron group
    neurons = NeuronGroup(NP + NB + NA + num_additional_neurons, model=eqs_neurons_current, threshold='v > vthr_3d',
                          reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:NP]
    pop_b = neurons[NP:NP + NB]
    pop_a = neurons[NP + NB:NP + NB + NA]
    # Copy of neurons. Their properties are defined later on
    extra_p_neurons = neurons[NP + NB + NA:NP + NB + NA + n_to_save_each_pop]
    extra_b_neurons = neurons[NP + NB + NA + n_to_save_each_pop:NP + NB + NA + n_to_save_each_pop * 2]
    extra_a_neurons = neurons[NP + NB + NA + 2 * n_to_save_each_pop:NP + NB + NA + 3 * n_to_save_each_pop]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    pop_p.background_curr = 200. * pA
    pop_b.background_curr = 200. * pA
    pop_a.background_curr = 200. * pA

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
    con_PA.g_pa = c_PA.g_update  # this is not a value, but an array

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update  # this is not a value, but an array

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update / 2.        # like it was clamped!

    # ======== 2 - connect new neurons to the network
    # select indices to copy neurons (randomly)
    indices_p = np.sort(random.sample(list(range(NP)), n_to_save_each_pop))
    indices_b = np.sort(random.sample(list(range(NB)), n_to_save_each_pop))
    indices_a = np.sort(random.sample(list(range(NA)), n_to_save_each_pop))

    # extra P neuron
    extra_p_neurons.vthr_3d = pop_p.vthr_3d[indices_p]
    extra_p_neurons.trefr = pop_p.trefr[indices_p]
    extra_p_neurons.background_curr = pop_p.background_curr[indices_p]

    con_PP_extra_p = Synapses(pop_p, extra_p_neurons, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    rows, cols, _ = extract_conn_submatrix(c_PP.rows_connect_matrix, c_PP.cols_connect_matrix, NP, NP, indices_p)
    con_PP_extra_p.connect(i=rows, j=cols)
    con_PP_extra_p.g_pp = c_PP.g_update

    con_PA_extra_p = Synapses(pop_a, extra_p_neurons, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    rows, cols, g_up_idx = extract_conn_submatrix(c_PA.rows_connect_matrix, c_PA.cols_connect_matrix, NA, NP, indices_p,
                                                  extract_g_update=True)
    con_PA_extra_p.connect(i=rows, j=cols)
    con_PA_extra_p.g_pa = c_PA.g_update

    con_PB_extra_p = Synapses(pop_b, extra_p_neurons, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    rows, cols, g_up_idx = extract_conn_submatrix(c_PB.rows_connect_matrix, c_PB.cols_connect_matrix, NB, NP, indices_p,
                                                  extract_g_update=True)
    con_PB_extra_p.connect(i=rows, j=cols)
    con_PB_extra_p.g_pb = c_PB.g_update

    # extra B neuron
    extra_b_neurons.vthr_3d = pop_b.vthr_3d[indices_b]
    extra_b_neurons.trefr = pop_b.trefr[indices_b]
    extra_b_neurons.background_curr = pop_b.background_curr[indices_b]

    con_BP_extra_b = Synapses(pop_p, extra_b_neurons, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    rows, cols, _ = extract_conn_submatrix(c_BP.rows_connect_matrix, c_BP.cols_connect_matrix, NP, NB, indices_b)
    con_BP_extra_b.connect(i=rows, j=cols)
    con_BP_extra_b.g_bp = c_BP.g_update

    con_BB_extra_b = Synapses(pop_b, extra_b_neurons, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    rows, cols, _ = extract_conn_submatrix(c_BB.rows_connect_matrix, c_BB.cols_connect_matrix, NB, NB, indices_b)
    con_BB_extra_b.connect(i=rows, j=cols)
    con_BB_extra_b.g_bb = c_BB.g_update

    con_BA_extra_b = Synapses(pop_a, extra_b_neurons, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    rows, cols, _ = extract_conn_submatrix(c_BA.rows_connect_matrix, c_BA.cols_connect_matrix, NA, NB, indices_b)
    con_BA_extra_b.connect(i=rows, j=cols)
    con_BA_extra_b.g_ba = c_BA.g_update

    # extra A neuron
    extra_a_neurons.vthr_3d = pop_a.vthr_3d[indices_a]
    extra_a_neurons.trefr = pop_a.trefr[indices_a]
    extra_a_neurons.background_curr = pop_a.background_curr[indices_a]

    con_AP_extra_a = Synapses(pop_p, extra_a_neurons, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    rows, cols, _ = extract_conn_submatrix(c_AP.rows_connect_matrix, c_AP.cols_connect_matrix, NP, NA, indices_a)
    con_AP_extra_a.connect(i=rows, j=cols)
    con_AP_extra_a.g_ap = c_AP.g_update

    con_AA_extra_a = Synapses(pop_a, extra_a_neurons, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    rows, cols, _ = extract_conn_submatrix(c_AA.rows_connect_matrix, c_AA.cols_connect_matrix, NA, NA, indices_a)
    con_AA_extra_a.connect(i=rows, j=cols)
    con_AA_extra_a.g_aa = c_AA.g_update

    con_AB_extra_a = Synapses(pop_b, extra_a_neurons, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    rows, cols, _ = extract_conn_submatrix(c_AB.rows_connect_matrix, c_AB.cols_connect_matrix, NB, NA, indices_a)
    con_AB_extra_a.connect(i=rows, j=cols)
    con_AB_extra_a.g_ab = c_AB.g_update / 2.        # like it was clamped

    # Set initial values to start from outside-SPW state
    pop_b.v = v_rest
    pop_p.v = v_rest
    extra_p_neurons.v = v_rest
    extra_b_neurons.v = v_rest
    pop_a.v = v_rest + (pop_a.vthr_3d - v_rest) * np.random.rand((NA))
    extra_a_neurons.v = v_rest + (extra_b_neurons.vthr_3d - v_rest) * np.random.rand((n_to_save_each_pop))
    # neurons.v = v_rest      # oscillation in the beginning
    neurons.g_ampa = 0.01 * nS * np.random.rand(
        (NP + NB + NA + num_additional_neurons))  # arbitrary choice of initial conductance values
    neurons.g_gabaA = 0.1 * nS * np.random.rand((NP + NB + NA + num_additional_neurons))
    neurons.g_gabaB = 0.1 * nS * np.random.rand((NP + NB + NA + num_additional_neurons))

    p_spikes = SpikeMonitor(pop_p)
    b_spikes = SpikeMonitor(pop_b)
    a_spikes = SpikeMonitor(pop_a)

    extra_spikes_p = SpikeMonitor(extra_p_neurons)
    extra_spikes_b = SpikeMonitor(extra_b_neurons)
    extra_spikes_a = SpikeMonitor(extra_a_neurons)

    # ======= 3 - run the network to estimate population FR
    warm_up_time = 1 * second  # run 1 second with defaults g-sb
    run(warm_up_time)

    # inject some current to one of the pop so that you jump to excited state and then compute the If curve from there
    if excited_state:
        run(warm_up_time)
        sigma_A = -650. * pA
        pop_a.extracurrent = sigma_A * np.random.rand(NA)

        current_on_time = 0.1 * second
        run(current_on_time)

        pop_a.extracurrent = 0 * pA

        run(jump_time - current_on_time - warm_up_time)

    print('----> Running network to estimate population firing rates... ')
    run(simtime_estimate_FR - 1 * second, report='text')

    pop_rate_P = average_firing_rate(p_spikes, jump_time + 2 * second, jump_time + simtime_estimate_FR)
    pop_rate_B = average_firing_rate(b_spikes, jump_time + 2 * second, jump_time + simtime_estimate_FR)
    pop_rate_A = average_firing_rate(a_spikes, jump_time + 2 * second, jump_time + simtime_estimate_FR)

    pop_rate_extra_p = average_firing_rate(extra_spikes_p, jump_time + 2 * second, jump_time + simtime_estimate_FR)
    pop_rate_extra_b = average_firing_rate(extra_spikes_b, jump_time + 2 * second, jump_time + simtime_estimate_FR)
    pop_rate_extra_a = average_firing_rate(extra_spikes_a, jump_time + 2 * second, jump_time + simtime_estimate_FR)

    print('==== State excited:', excited_state)
    print('Pop rates:  P: ', pop_rate_P)
    print('B: ', pop_rate_B)
    print('A: ', pop_rate_A)
    print('EXTRA: ')
    print('P: ', pop_rate_extra_p)
    print('B: ', pop_rate_extra_b)
    print('A: ', pop_rate_extra_a)

    # ======== 5 - inject current to extra neuron and count spikes
    print('----> Stimulating extra neurons to compute If curve')
    sm_extra_p = SpikeMonitor(extra_p_neurons)
    sm_extra_b = SpikeMonitor(extra_b_neurons)
    sm_extra_a = SpikeMonitor(extra_a_neurons)

    I_inp_mon_p = StateMonitor(extra_p_neurons, 'I_input', record=True)
    I_inp_mon_b = StateMonitor(extra_b_neurons, 'I_input', record=True)
    I_inp_mon_a = StateMonitor(extra_a_neurons, 'I_input', record=True)

    # Snapshot the state
    store()

    # extracurrent_array = np.arange(-100., 200., 5)*pA
    extracurrent_array = np.arange(-100., 200., 20) * pA

    # Run the trials
    spike_counts_p = np.empty((len(extracurrent_array), n_to_save_each_pop))
    spike_counts_b = np.empty((len(extracurrent_array), n_to_save_each_pop))
    spike_counts_a = np.empty((len(extracurrent_array), n_to_save_each_pop))

    mean_input_to_extra_p_neurons = np.empty(
        (len(extracurrent_array), n_to_save_each_pop))  # as many entries as neurons
    mean_input_to_extra_b_neurons = np.empty((len(extracurrent_array), n_to_save_each_pop))
    mean_input_to_extra_a_neurons = np.empty((len(extracurrent_array), n_to_save_each_pop))

    # std across time
    std_input_to_extra_p = np.empty((len(extracurrent_array), n_to_save_each_pop))
    std_input_to_extra_b = np.empty((len(extracurrent_array), n_to_save_each_pop))
    std_input_to_extra_a = np.empty((len(extracurrent_array), n_to_save_each_pop))

    for idx_now, extracurr_now in enumerate(extracurrent_array):
        print('Current: ', extracurr_now)
        # Restore the initial state
        restore()

        # add current simultaneously to all extra neurons
        extra_p_neurons.extracurrent = extracurr_now
        extra_b_neurons.extracurrent = extracurr_now
        extra_a_neurons.extracurrent = extracurr_now

        run(simtime_current)

        # store the results
        spike_counts_p[idx_now, :] = sm_extra_p.count
        spike_counts_b[idx_now, :] = sm_extra_b.count
        spike_counts_a[idx_now, :] = sm_extra_a.count

        mean_input_to_extra_p_neurons[idx_now, :] = mean(I_inp_mon_p.I_input, axis=1) / pA  # as many entries as neurons
        mean_input_to_extra_b_neurons[idx_now, :] = mean(I_inp_mon_b.I_input, axis=1) / pA
        mean_input_to_extra_a_neurons[idx_now, :] = mean(I_inp_mon_a.I_input, axis=1) / pA

        # std across time
        std_input_to_extra_p[idx_now, :] = std(I_inp_mon_p.I_input, axis=1) / pA
        std_input_to_extra_b[idx_now, :] = std(I_inp_mon_b.I_input, axis=1) / pA
        std_input_to_extra_a[idx_now, :] = std(I_inp_mon_a.I_input, axis=1) / pA

    fr_extra_p = spike_counts_p / simtime_current
    fr_extra_b = spike_counts_b / simtime_current
    fr_extra_a = spike_counts_a / simtime_current


    # save stimulation results, e.g. mean input and std, frequency of spiking
    if excited_state:
        filename_full = os.path.join(path_folder, filename + '_excited_IF_n' + str(n_to_save_each_pop) + '.npz')
    else:
        filename_full = os.path.join(path_folder, filename + '_resting_IF_n' + str(n_to_save_each_pop) + '.npz')

    # NOTE: things are saved without units
    data_to_save = {'mean_input_p': mean_input_to_extra_p_neurons,  # mean over time (1x each extra neuron)
                    'mean_input_b': mean_input_to_extra_b_neurons,
                    'mean_input_a': mean_input_to_extra_a_neurons,
                    'std_input_p': std_input_to_extra_p,
                    'std_input_b': std_input_to_extra_b,
                    'std_input_a': std_input_to_extra_a,
                    'extracurrent_array': extracurrent_array / pA,
                    'fr_p': fr_extra_p,
                    'fr_b': fr_extra_b,
                    'fr_a': fr_extra_a,
                    }

    np.savez_compressed(filename_full, **data_to_save)
    print('Data saved in file: ', filename_full)
