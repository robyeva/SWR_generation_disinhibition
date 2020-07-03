__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains functions to
- derive the rate network from the spiking model
- simulate and plot the rate model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from textwrap import wrap
from matplotlib.font_manager import FontProperties
from brian2 import mV, ms, second

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

from run_rate import path_folder
from helper_functions.utils_rate import *
from helper_functions.utils_spiking import Connectivity, add_sign_of_stimulation, adjust_xaxis, adjust_yaxis, AnchoredHScaleBar


def simulate_rate_bistable_softplus_changing_Vx(filename_spiking, step_v=0.5, plot_traces=False,
                                                save_results=True):
    """Simulate the rate model with all possible combinations of V_P,V_B, V_A

     :param filename_spiking: str
            Filename of spiking model
     :param step_v: float
            Decides step in mean membrane potential testing (V_x = np.range(-60, -50, step_v))
     :param plot_traces: bool
            If True, plots P,B,A traces for each V_P, V_B V_A combination
     :param save_results: bool
            If True, saves the firing rates for each V_P, V_B V_A combination
     """

    filename_spiking_full = os.path.join(path_folder, filename_spiking + '_step6.npz')
    data = np.load(filename_spiking_full, encoding='latin1', allow_pickle=True)
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

    rev_ampa = network_params['rev_ampa'] / mV
    rev_gabaB = network_params['rev_gabaB'] / mV
    rev_gabaA = network_params['rev_gabaA'] / mV
    tau_ampa = network_params['tau_ampa'] / ms
    tau_gabaB = network_params['tau_gabaB'] / ms
    tau_gabaA = network_params['tau_gabaA'] / ms
    NP = network_params['NP']
    NB = network_params['NB']
    NA = network_params['NA']

    volt_range_P = np.arange(-50., -60. - step_v, -step_v)
    volt_range_B = np.arange(-50., -60. - step_v, -step_v)
    volt_range_A = np.arange(-50., -60. - step_v, -step_v)

    a0array = np.zeros((len(volt_range_P), len(volt_range_B), len(volt_range_A), 3))
    a1array = np.zeros_like(a0array)
    p0array = np.zeros_like(a0array)
    p1array = np.zeros_like(a0array)
    b0array = np.zeros_like(a0array)
    b1array = np.zeros_like(a0array)

    mat_to_take = np.ones((len(volt_range_P), len(volt_range_B), len(volt_range_A)))
    tol_closeness = 0.1

    # also load data for I-f curves
    filename_IF = os.path.join(path_folder, filename_spiking + '_intermediate_fI.npz')
    data = np.load(filename_IF, encoding='latin1', allow_pickle=True)
    dic_data_IF = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    If_P_param = dic_data_IF['If_P_param']
    If_B_param = dic_data_IF['If_B_param']
    If_A_param = dic_data_IF['If_A_param']

    # 'biological' firing rates
    good_cond = np.array([[5, 5, 8], [8, 30, 5]])

    sim = SimParam()
    sim.t_max = 300 * 1e-3
    sim.pulse_lim = 30. * 1e-3
    sim.pulse_start = 0.05

    t_array = np.arange(0, sim.t_max, sim.dt)
    # 20 ms before current is injected: should be in outside state
    idx_before_current = (np.abs(t_array - (sim.pulse_start - 0.02))).argmin()
    # 100 ms before end of sim: should be in inside state
    idx_after_current = (np.abs(t_array - (sim.t_max - 0.1))).argmin()

    depr_clamp = DeprClamp()
    depr_clamp.t_second = np.infty
    depr_clamp.t_first = np.infty
    depr_clamp.first_change = 0.5

    # IC are now the non-SWR state
    p0 = 0.
    a0 = 12.
    b0 = 1.
    dIC = 0.5
    IC_resting = [p0, b0, a0, dIC]

    curr = ['I_p', 'I_b', 'I_a']
    num_pbs = [0, 1, 2]
    subplot_d = [4, 5, 6]

    for idx_p, mean_P_membpot in enumerate(volt_range_P):
        for idx_b, mean_B_membpot in enumerate(volt_range_B):
            for idx_a, mean_A_membpot in enumerate(volt_range_A):

                scaling_factor = 1e-3  # to make units be in pA*second

                W_pp = NP * c_PP.prob_of_connect * np.mean(c_PP.g_update) * tau_ampa * np.abs(
                    mean_P_membpot - rev_ampa) * scaling_factor
                W_pb = NB * c_PB.prob_of_connect * np.mean(c_PB.g_update) * tau_gabaB * (
                            mean_P_membpot - rev_gabaB) * scaling_factor
                W_pa = NA * c_PA.prob_of_connect * np.mean(c_PA.g_update) * tau_gabaA * (
                            mean_P_membpot - rev_gabaA) * scaling_factor

                W_bp = NP * c_BP.prob_of_connect * np.mean(c_BP.g_update) * tau_ampa * np.abs(
                    mean_B_membpot - rev_ampa) * scaling_factor
                W_bb = NB * c_BB.prob_of_connect * np.mean(c_BB.g_update) * tau_gabaB * (
                            mean_B_membpot - rev_gabaB) * scaling_factor
                W_ba = NA * c_BA.prob_of_connect * np.mean(c_BA.g_update) * tau_gabaA * (
                            mean_B_membpot - rev_gabaA) * scaling_factor

                W_ap = NP * c_AP.prob_of_connect * np.mean(c_AP.g_update) * tau_ampa * np.abs(
                    mean_A_membpot - rev_ampa) * scaling_factor
                W_ab = NB * c_AB.prob_of_connect * np.mean(c_AB.g_update) * tau_gabaB * (
                            mean_A_membpot - rev_gabaB) * scaling_factor
                W_aa = NA * c_AA.prob_of_connect * np.mean(c_AA.g_update) * tau_gabaA * (
                            mean_A_membpot - rev_gabaA) * scaling_factor

                dic = dict(tau_b=2.0 * 1e-3, tau_p=3.0 * 1e-3, tau_a=6.0 * 1e-3, W_pp=W_pp, W_pa=W_pa, W_pb=W_pb,
                           W_ap=W_ap, W_aa=W_aa, W_ab=W_ab, W_bp=W_bp, W_bb=W_bb, W_ba=W_ba)

                dic['eta'] = network_params['eta_AB']
                dic['tau_d'] = network_params['tau_depr_AB'] / second

                net = FullParamFromDict(dic)
                net.k_p = If_P_param[0]
                net.k_b = If_B_param[0]
                net.k_a = If_A_param[0]
                net.t_p = If_P_param[1] + 200
                net.t_b = If_B_param[1] + 200
                net.t_a = If_A_param[1] + 200

                sim_all_current = np.empty((np.shape(t_array)[0], 4, 3))
                fig = plt.figure(figsize=[11, 7])
                my_size = 12
                outer = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])
                gs2 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=outer[:, :], hspace=.08, wspace=0.08)

                x_lim_start = sim.pulse_start - 100 * 1e-3  # ms
                x_lim_end = sim.pulse_start + 200 * 1e-3  # ms
                xlim_stim = [x_lim_start, x_lim_end]

                for inj_curr, spbs, sd in zip(curr, num_pbs, subplot_d):
                    if inj_curr == 'I_p':
                        curr_value = 350.  # pA
                        dic['I_p'] = curr_value
                    elif inj_curr == 'I_b':
                        curr_value = 150.
                        dic['I_b'] = curr_value
                    if inj_curr == 'I_a':
                        curr_value = 200.  # negative is taken inside ODE
                        dic['I_a'] = curr_value

                    # ========== use If curves approximation using softplus curves
                    sim_all_current[:, :, spbs] = odeint(eq_clamp_depression, IC_resting, t_array, printmessg=False,
                                                         hmax=0.01, rtol=1e-6, atol=1e-8,
                                                         args=(
                                                         net, sim, depr_clamp, curr_value, False, inj_curr))

                    if np.any(is_nan(sim_all_current[idx_after_current, :, spbs])):
                        # print('Rerunning nan case')
                        # slightly change volt_value and retry
                        mean_P_membpot -= 0.1
                        mean_B_membpot += 0.01
                        mean_A_membpot += 0.01

                        # see 16.08 (1) for why abs in ampa things
                        W_pp = NP * c_PP.prob_of_connect * np.mean(c_PP.g_update) * tau_ampa * np.abs(
                            mean_P_membpot - rev_ampa) * scaling_factor
                        W_pb = NB * c_PB.prob_of_connect * np.mean(c_PB.g_update) * tau_gabaB * (
                                mean_P_membpot - rev_gabaB) * scaling_factor
                        W_pa = NA * c_PA.prob_of_connect * np.mean(c_PA.g_update) * tau_gabaA * (
                                mean_P_membpot - rev_gabaA) * scaling_factor

                        W_bp = NP * c_BP.prob_of_connect * np.mean(c_BP.g_update) * tau_ampa * np.abs(
                            mean_B_membpot - rev_ampa) * scaling_factor
                        W_bb = NB * c_BB.prob_of_connect * np.mean(c_BB.g_update) * tau_gabaB * (
                                mean_B_membpot - rev_gabaB) * scaling_factor
                        W_ba = NA * c_BA.prob_of_connect * np.mean(c_BA.g_update) * tau_gabaA * (
                                mean_B_membpot - rev_gabaA) * scaling_factor

                        W_ap = NP * c_AP.prob_of_connect * np.mean(c_AP.g_update) * tau_ampa * np.abs(
                            mean_A_membpot - rev_ampa) * scaling_factor
                        W_ab = NB * c_AB.prob_of_connect * np.mean(c_AB.g_update) * tau_gabaB * (
                                mean_A_membpot - rev_gabaB) * scaling_factor
                        W_aa = NA * c_AA.prob_of_connect * np.mean(c_AA.g_update) * tau_gabaA * (
                                mean_A_membpot - rev_gabaA) * scaling_factor

                        dic = dict(tau_b=2.0 * 1e-3, tau_p=3.0 * 1e-3, tau_a=6.0 * 1e-3, W_pp=W_pp, W_pa=W_pa,
                                   W_pb=W_pb,
                                   W_ap=W_ap, W_aa=W_aa, W_ab=W_ab, W_bp=W_bp, W_bb=W_bb, W_ba=W_ba)

                        dic['eta'] = network_params['eta_AB']
                        dic['tau_d'] = network_params['tau_depr_AB'] / second
                        net = FullParamFromDict(dic)

                        net.k_p = If_P_param[0]
                        net.k_b = If_B_param[0]
                        net.k_a = If_A_param[0]
                        net.t_p = If_P_param[1] + 200
                        net.t_b = If_B_param[1] + 200
                        net.t_a = If_A_param[1] + 200

                        sim_all_current[:, :, spbs] = odeint(eq_clamp_depression, IC_resting, t_array, printmessg=False,
                                                             hmax=10, rtol=1e-6, atol=1e-8,
                                                             args=(net, sim, depr_clamp, curr_value, False,
                                                                   inj_curr))
                        if np.any(is_nan(sim_all_current[idx_after_current, :, spbs])):
                            # print('Not solved', idx_p, idx_b, idx_a)
                            mat_to_take[idx_p, idx_b, idx_a] = 0

                    # Check which firing rates are biological
                    if np.any([sim_all_current[idx_before_current, 0, spbs] > good_cond[0, 0],
                               sim_all_current[idx_before_current, 1, spbs] > good_cond[0, 1],
                               sim_all_current[idx_before_current, 2, spbs] < good_cond[0, 2],
                               sim_all_current[idx_after_current, 0, spbs] < good_cond[1, 0],
                               sim_all_current[idx_after_current, 1, spbs] < good_cond[1, 1],
                               sim_all_current[idx_after_current, 2, spbs] > good_cond[1, 2]]):
                        mat_to_take[idx_p, idx_b, idx_a] = 0

                    else:
                        a0array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_before_current, 2, spbs]
                        a1array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_after_current, 2, spbs]
                        p0array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_before_current, 0, spbs]
                        p1array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_after_current, 0, spbs]
                        b0array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_before_current, 1, spbs]
                        b1array[idx_p, idx_b, idx_a, spbs] = sim_all_current[idx_after_current, 1, spbs]

                # mat_to_take contains intersection of conditions (as soon as one not valid, it is zero)
                # p0/bp/etc array tell us where the problem was, because they are zeros everywhere
                # also discard the cases where all I_x results have a FP, but FP are too different
                if mat_to_take[idx_p, idx_b, idx_a] == 1:
                    for my_data in [p0array, b0array, a0array, p1array, b1array, a1array]:
                        results_diff_I = my_data[idx_p, idx_b, idx_a, :]
                        if np.any(np.abs(results_diff_I - results_diff_I[0]) > tol_closeness):
                            mat_to_take[idx_p, idx_b, idx_a] = 0

                if plot_traces:

                    for curr_to_pop, spbs in zip(['P', 'B', 'A'], [0, 1, 2]):
                        # =================== P cells ================== #
                        ax = plt.subplot(gs2[0, spbs])
                        plt.plot(t_array, sim_all_current[:, 0, spbs], '#ef3b53', label='P', lw=2.5)
                        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
                        plt.scatter(t_array[idx_before_current], sim_all_current[idx_before_current, 0, spbs],
                                    marker='o', color='Yellow')
                        plt.scatter(t_array[idx_after_current], sim_all_current[idx_after_current, 0, spbs], marker='o',
                                    color='Blue')
                        plt.ylim([0, 100])

                        # =================== B cells ================== #
                        ax = plt.subplot(gs2[1, spbs])
                        plt.plot(t_array, sim_all_current[:, 1, spbs], '#3c3fef', label='B', lw=2.5)
                        plt.ylim([0, 200])
                        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
                        plt.scatter(t_array[idx_before_current], sim_all_current[idx_before_current, 1, spbs],
                                    marker='o', color='Yellow')
                        plt.scatter(t_array[idx_after_current], sim_all_current[idx_after_current, 1, spbs], marker='o',
                                    color='Blue')
                        # =================== A cells ================== #
                        ax = plt.subplot(gs2[2, spbs])
                        plt.plot(t_array, sim_all_current[:, 2, spbs], '#0a9045', label='A', lw=2.5)
                        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
                        plt.scatter(t_array[idx_before_current], sim_all_current[idx_before_current, 2, spbs],
                                    marker='o', color='Yellow')
                        plt.scatter(t_array[idx_after_current], sim_all_current[idx_after_current, 2, spbs], marker='o',
                                    color='Blue')
                        plt.ylim([0, 20])

                        # ================= syn. efficacy ================= #
                        ax = plt.subplot(gs2[3, spbs])
                        plt.plot(t_array, sim_all_current[:, 3, spbs], '#e67e22', label='d', lw=2.5)
                        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
                        if curr_to_pop == 'P':
                            plt.ylabel('Synaptic efficacy', fontsize=my_size)
                        plt.ylim([0, 1])
                        plt.scatter(t_array[idx_before_current], sim_all_current[idx_before_current, 3, spbs],
                                    marker='o', color='Yellow')
                        plt.scatter(t_array[idx_after_current], sim_all_current[idx_after_current, 3, spbs], marker='o',
                                    color='Blue')

                    plt.suptitle('Vx = %.1f, %.1f, %.1f, FR outside (P,B,A) %.2f, %.2f, %.2f Hz, FR inside: %.2f, %.2f, '
                             '%.2f Hz' % (mean_P_membpot, mean_B_membpot, mean_A_membpot,
                                sim_all_current[idx_before_current, 0, spbs],
                                sim_all_current[idx_before_current, 1, spbs],
                                sim_all_current[idx_before_current, 2, spbs],
                                sim_all_current[idx_after_current, 0, spbs],
                                sim_all_current[idx_after_current, 1, spbs],
                                sim_all_current[idx_after_current, 2, spbs]))

                    # add a common y label for population rate
                    fig.text(0.085, 0.6, 'Population rate [Hz]', va='center', ha='center', rotation='vertical',
                             fontsize=my_size)
                    plt.show()
                else:
                    plt.close()

    if save_results:
        # save arrays in a file to then plot, find best match, etc
        dic_sims = {'p0array': p0array,
                    'b0array': b0array,
                    'a0array': a0array,
                    'p1array': p1array,
                    'b1array': b1array,
                    'a1array': a1array,
                    'volt_range_P': volt_range_P,
                    'volt_range_B': volt_range_B,
                    'volt_range_A': volt_range_A,
                    'good_cond': good_cond,
                    'mat_to_take': mat_to_take,
                    }
        filename_sims = os.path.join(path_folder, filename_spiking + '_Vx_results_bistable_step_' + str(step_v) + '.npz')
        np.savez_compressed(filename_sims, **dic_sims)


def analyze_Vx_results(filename_spiking, step_v=0.5):
    """Finds optimal values for the mean membrane potential values (to go from spiking to rate model) based on
    simulations saved in 'simulate_rate_bistable_softplus_changing_Vx'

     :param filename_spiking: str
            Filename of spiking model
     :param step_v: float
            Step used in mean membrane potential testing (V_x = np.range(-60, -50, step_v))
     """

    filename_sims = os.path.join(path_folder, filename_spiking + '_Vx_results_bistable_step_' + str(step_v) + '.npz')
    data = np.load(filename_sims, encoding='latin1', allow_pickle = True)
    sim_dic = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    p0array = sim_dic['p0array']
    b0array = sim_dic['b0array']
    a0array = sim_dic['a0array']
    p1array = sim_dic['p1array']
    b1array = sim_dic['b1array']
    a1array = sim_dic['a1array']
    volt_range_P = sim_dic['volt_range_P']
    volt_range_B = sim_dic['volt_range_B']
    volt_range_A = sim_dic['volt_range_A']
    mat_to_take = sim_dic['mat_to_take']
    good_cond = sim_dic['good_cond']

    # of the 3 stimulations, (I_P, I_B, I_A) just take one (should all be closer than tol_closeness)
    p0array = np.squeeze(p0array[:, :, :, 0])
    b0array = np.squeeze(b0array[:, :, :, 0])
    a0array = np.squeeze(a0array[:, :, :, 0])
    p1array = np.squeeze(p1array[:, :, :, 0])
    b1array = np.squeeze(b1array[:, :, :, 0])
    a1array = np.squeeze(a1array[:, :, :, 0])
    k = np.squeeze(mat_to_take)

    # this is needed to correct for overflows in exp(ax+b)
    p0array = np.nan_to_num(p0array)
    b0array = np.nan_to_num(b0array)
    a0array = np.nan_to_num(a0array)
    p1array = np.nan_to_num(p1array)
    b1array = np.nan_to_num(b1array)
    a1array = np.nan_to_num(a1array)

    # plot 3d visualization of results
    fig = plt.figure(figsize=[16, 9])
    fig.subplots_adjust(wspace=0.5, hspace=0.7)
    cm = plt.cm.get_cmap('jet')
    my_size = 12
    my_steps = [-50, -53, -56, -59]

    scale_x = 1
    scale_y = 1
    scale_z = 1.8

    my_idx = [(np.abs(el - volt_range_A)).argmin() for el in my_steps]
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    for ax, data_array, my_title in zip([ax1, ax2, ax3, ax4, ax5, ax6],
                                        [p0array, b0array, a0array, p1array, b1array, a1array],
                                        ['P_0', 'B_0', 'A_0',
                                         'P_1', 'B_1', 'A_1']):
        x, y, z = data_array.nonzero()
        # make only one colorbar across all planes
        zs = data_array[x, y, z]
        min_, max_ = zs.min(), zs.max()

        for idx in range(len(my_steps)):
            right_idx = np.where(z == my_idx[idx])[0]
            xx, yy = np.meshgrid(list(range(len(volt_range_P))), list(range(len(volt_range_B))))
            zz = np.ones_like(yy) * my_idx[idx]
            ax.plot_surface(xx, yy, zz, alpha=0.1, zorder=-3, color='LightGray')
            sc = ax.scatter(x[right_idx], y[right_idx], z[right_idx],
                            c=data_array[x[right_idx], y[right_idx], z[right_idx]],
                            cmap=cm, vmin=min_, vmax=max_)

        ax.set_title(my_title, y=1.42, fontsize=my_size + 2)
        fancy_plotting_3d(ax, volt_range_P, volt_range_B, volt_range_A, my_size)
        axins = inset_axes(ax,
                           width="5%",  # width = 10% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='upper left',
                           bbox_to_anchor=(1.25, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
        cbar = plt.colorbar(sc, cax=axins)
        cbar.set_label('[1/s]', fontsize=my_size)
        ax.view_init(25, None)
        # orientation depends on figsize??!?!
        ax.text(4, -15, 0, 'P memb. pot. [mV]', 'x', fontsize=my_size)
        # ax.text(20, 10, 0, 'B memb. pot. [mV]', (0.90,0,1), fontsize=my_size)
        ax.text(20, 10, 0, 'B memb. pot. [mV]', 'y', fontsize=my_size)

    ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([scale_x, scale_y, scale_z, 1]))
    ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([scale_x, scale_y, scale_z, 1]))
    ax3.get_proj = lambda: np.dot(Axes3D.get_proj(ax3), np.diag([scale_x, scale_y, scale_z, 1]))
    ax4.get_proj = lambda: np.dot(Axes3D.get_proj(ax4), np.diag([scale_x, scale_y, scale_z, 1]))
    ax5.get_proj = lambda: np.dot(Axes3D.get_proj(ax5), np.diag([scale_x, scale_y, scale_z, 1]))
    ax6.get_proj = lambda: np.dot(Axes3D.get_proj(ax6), np.diag([scale_x, scale_y, scale_z, 1]))

    fig.text(0.085, 0.3, 'SWR state', va='center', ha='center', rotation='vertical', fontsize=my_size + 4)
    fig.text(0.085, 0.8, 'non-SWR state', va='center', ha='center', rotation='vertical', fontsize=my_size + 4)

    plt.savefig(os.path.join(path_folder, filename_spiking + '_find_good_membpot_FR_BISTABLE.pdf'),
            dpi=200, format='pdf')

    # ==================================== identify the best combination
    filename_full = os.path.join(path_folder, filename_spiking + '_target_FR.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    target_P_out = info_dictionary['target_P_nonSWR']
    target_B_out = info_dictionary['target_B_nonSWR'].copy()
    target_A_out = info_dictionary['target_A_nonSWR'].copy()
    target_P_in = info_dictionary['target_P_SWR'].copy()
    target_B_in = info_dictionary['target_B_SWR'].copy()
    target_A_in = info_dictionary['target_A_SWR'].copy()

    all_targets = [target_P_out, target_B_out, target_A_out, target_P_in, target_B_in, target_A_in]
    all_arrays = np.array(
        [p0array.flatten(), b0array.flatten(), a0array.flatten(), p1array.flatten(), b1array.flatten(),
         a1array.flatten()]).T
    idx_min = np.array([np.sqrt(x ** 2 + y ** 2 + z ** 2 + l ** 2 + m ** 2 + n ** 2) for (x, y, z, l, m, n) in
                        all_arrays - all_targets]).argmin()

    idx_min_tuple = np.unravel_index([idx_min], (len(volt_range_P), len(volt_range_B), len(volt_range_A)))

    print('Optimal membrane potential values: ', volt_range_P[idx_min_tuple[0]], volt_range_B[idx_min_tuple[1]], \
          volt_range_A[idx_min_tuple[2]])

    dic_opt = {'V_p': float(volt_range_P[idx_min_tuple[0]]),
               'V_b': float(volt_range_B[idx_min_tuple[1]]),
               'V_a': float(volt_range_A[idx_min_tuple[2]]),
               }
    filename_opt = os.path.join(path_folder, filename_spiking + '_optimal3d_alltargets_simulationbased.npz')
    print('Optimal values saved in: ', filename_opt)
    np.savez_compressed(filename_opt, **dic_opt)


def simulate_from_spiking(filename_spiking, filename_rate, simulate_bistable=False,
                          simulate_SPW_like=False, use_softplus=True):
    """Creates rate model using parameters derived from the spiking model and create plots 5 and 10

     :param filename_spiking: str
        Filename of spiking model
     :param filename_rate: str
        Filename of rate model
     :param simulate_bistable: bool
        If True, creates Fig. 2-2
     :param simulate_SPW_like: bool
        If True, creates Fig. 6-2
     :param use_softplus: bool
        If True (default), uses the softplus function to simulate the rate model. Alternatively, it uses its
        threshold-linear approximation

    """

    filename_spiking_full = os.path.join(path_folder, filename_spiking + '_step6.npz')
    data = np.load(filename_spiking_full, encoding='latin1',allow_pickle=True)
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

    rev_ampa = network_params['rev_ampa'] / mV
    rev_gabaB = network_params['rev_gabaB'] / mV
    rev_gabaA = network_params['rev_gabaA'] / mV
    tau_ampa = network_params['tau_ampa'] / ms
    tau_gabaB = network_params['tau_gabaB'] / ms
    tau_gabaA = network_params['tau_gabaA'] / ms
    NP = network_params['NP']
    NB = network_params['NB']
    NA = network_params['NA']

    filename_optimal = os.path.join(path_folder, filename_spiking + '_optimal3d_alltargets_simulationbased.npz')
    data = np.load(filename_optimal, encoding='latin1', allow_pickle=True)
    opt_dic = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    mean_P_membpot = opt_dic['V_p']
    mean_B_membpot = opt_dic['V_b']
    mean_A_membpot = opt_dic['V_a']
    print('Optimal values Memb Pot: ', mean_P_membpot, mean_B_membpot, mean_A_membpot)

    scaling_factor = 1e-3  # to make units be in pA*second

    W_pp = NP * c_PP.prob_of_connect * np.mean(c_PP.g_update) * tau_ampa * np.abs(
        mean_P_membpot - rev_ampa) * scaling_factor
    W_pb = NB * c_PB.prob_of_connect * np.mean(c_PB.g_update) * tau_gabaB * (
            mean_P_membpot - rev_gabaB) * scaling_factor
    W_pa = NA * c_PA.prob_of_connect * np.mean(c_PA.g_update) * tau_gabaA * (
            mean_P_membpot - rev_gabaA) * scaling_factor

    W_bp = NP * c_BP.prob_of_connect * np.mean(c_BP.g_update) * tau_ampa * np.abs(
        mean_B_membpot - rev_ampa) * scaling_factor
    W_bb = NB * c_BB.prob_of_connect * np.mean(c_BB.g_update) * tau_gabaB * (
            mean_B_membpot - rev_gabaB) * scaling_factor
    W_ba = NA * c_BA.prob_of_connect * np.mean(c_BA.g_update) * tau_gabaA * (
            mean_B_membpot - rev_gabaA) * scaling_factor

    W_ap = NP * c_AP.prob_of_connect * np.mean(c_AP.g_update) * tau_ampa * np.abs(
        mean_A_membpot - rev_ampa) * scaling_factor
    W_ab = NB * c_AB.prob_of_connect * np.mean(c_AB.g_update) * tau_gabaB * (
            mean_A_membpot - rev_gabaB) * scaling_factor
    W_aa = NA * c_AA.prob_of_connect * np.mean(c_AA.g_update) * tau_gabaA * (
            mean_A_membpot - rev_gabaA) * scaling_factor

    print('Values W')
    print('W_pp ', W_pp)
    print('W_pb ', W_pb)
    print('W_pa ', W_pa)

    print('W_bp ', W_bp)
    print('W_bb ', W_bb)
    print('W_ba ', W_ba)

    print('W_ap ', W_ap)
    print('W_ab ', W_ab)
    print('W_aa ', W_aa)

    dic = dict(tau_b=2.0 * 1e-3, tau_p=3.0 * 1e-3, tau_a=6.0 * 1e-3, W_pp=W_pp, W_pa=W_pa, W_pb=W_pb,
               W_ap=W_ap, W_aa=W_aa, W_ab=W_ab, W_bp=W_bp, W_bb=W_bb, W_ba=W_ba)

    dic['eta'] = network_params['eta_AB']
    dic['tau_d'] = network_params['tau_depr_AB'] / second

    # ============= also load data for f-I curves
    filename_fI = os.path.join(path_folder, filename_spiking + '_intermediate_fI.npz')
    data = np.load(filename_fI, encoding='latin1', allow_pickle=True)
    dic_data_fI = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    If_P_param = dic_data_fI['If_P_param']
    If_B_param = dic_data_fI['If_B_param']
    If_A_param = dic_data_fI['If_A_param']

    net = FullParamFromDict(dic)
    net.k_p = If_P_param[0]
    net.t_p = If_P_param[1] + 200
    net.k_b = If_B_param[0]
    net.t_b = If_B_param[1] + 200
    net.k_a = If_A_param[0]
    net.t_a = If_A_param[1] + 200

    print('Paramters k and t for P, B, and A: ')
    print(net.k_p, net.t_p)
    print(net.k_b, net.t_b)
    print(net.k_a, net.t_a)

    if simulate_bistable:
        # creates Fig. 5
        inside_check_bistability_singlecurr(net, filename_rate, use_softplus=use_softplus)

    if simulate_SPW_like:
        # creates Fig. 10
        inside_SPW_stimulation(net, filename_rate, use_softplus=use_softplus)


def inside_check_bistability_singlecurr(net, filename_rate, inj_curr='I_a', use_softplus=True):
    """Generates simulation to check for bistability in the model and creates Fig. 5

     :param net: class FullParamFromDict
        Contains parameters of the rate model
     :param filename_rate: str
        Filename of rate model
     :param inj_current: str
        To which cell type current is to be injected
     :param use_softplus: bool
        If True (default), uses the softplus function to simulate the rate model. Alternatively, it uses its
        threshold-linear approximation

    """
    sim = SimParam()
    sim.t_max = 6000 * 1e-3
    sim.pulse_lim = 10. * 1e-3
    sim.pulse_start = 1000. * 1e-3
    sim.second_pulse_start = 2000 * 1e-3
    sim.third_pulse_start = 4000 * 1e-3
    t = np.arange(0, sim.t_max, sim.dt)
    # IC are the outside SPW state
    p0 = 0.1
    a0 = 12.
    b0 = 3.
    dIC = 1. / (1. + net.eta * b0)
    IC_resting = [p0, b0, a0, dIC]

    curr_value = 300.  # pA

    depr_clamp = DeprClamp()
    depr_clamp.t_first = 3000 * 1e-3
    depr_clamp.t_second = 5000 * 1e-3
    depr_clamp.first_change = 0.5
    depr_clamp.second_change = 0.8
    depr_clamp.third_change = 0.2

    sim_all_current = odeint(eq_clamp_depression, IC_resting, t, printmessg=False,
                             # hmax=max integration step is 1   ms (to detect pulse)
                             hmax=0.01, rtol=1e-6, atol=1e-8,
                             # tcrit lists times where something important happens that the solver should not miss!
                             tcrit=[sim.pulse_start, sim.second_pulse_start, sim.third_pulse_start],
                             args=(net, sim, depr_clamp, curr_value, False, inj_curr, use_softplus))

    P = sim_all_current[:, 0]
    B = sim_all_current[:, 1]
    A = sim_all_current[:, 2]
    # d = sim_all_current[:, 3]         # this only contains copy of the initial condition for d,
    # acutal value for d is computed at each time step inside eq_clamp_depression
    d = depr_clamp.first_change * (t <= depr_clamp.t_first) \
        + depr_clamp.second_change * (t > depr_clamp.t_first) * (t <= depr_clamp.t_second) \
        + depr_clamp.third_change * (t > depr_clamp.t_second)

    I = curr_value * (sim.pulse_start <= t) * (t < sim.pulse_lim + sim.pulse_start) \
        + (-curr_value) * (sim.second_pulse_start <= t) * (t < sim.pulse_lim + sim.second_pulse_start) \
        + curr_value * (sim.third_pulse_start <= t) * (t < sim.pulse_lim + sim.third_pulse_start)

    fig = plt.figure(figsize=[9, 9])
    my_size = 13
    for my_data, subp, mycol, my_label in zip([P, B, A, d, I], [1, 2, 3, 5, 4],
                                              ['#ef3b53', '#3c3fef', '#0a9045', '#e67e22', '#d4b021'],
                                              ['P', 'B', 'A', None, None]):
        ax = plt.subplot(5, 1, subp)
        plt.plot(t, my_data, lw=3, c=mycol, label=my_label)
        adjust_yaxis(ax, my_size)
        adjust_xaxis(ax, [0, sim.t_max], my_size, show_bottom=False)
        # add_sign_of_stimulation(ax,True, sim.pulse_start, sim.pulse_lim)
        plt.legend(loc='best', prop={'size': my_size}, frameon=False)

        if subp == 5:
            # add scalebar
            ob = AnchoredHScaleBar(size=0.5, label="0.5 s", loc=3, frameon=False, extent=0,
                                   pad=1., sep=4, color="Black",
                                   borderpad=0.1,
                                   my_size=my_size)  # pad and borderpad can be used to modify the position of the bar a bit
            ax.add_artist(ob)
            plt.ylim(-0.05, 1.05)
            plt.yticks([0.2, 0.5, 0.8])
            plt.ylabel('\n'.join(wrap('Synaptic efficacy', 10)), fontsize=my_size)
        if subp == 4:
            plt.ylabel('\n'.join(wrap('Injected current [pA]', 14)), fontsize=my_size)
        if subp == 2:
            plt.ylim([-5., 152])
            plt.yticks([0, 100])
        if subp == 1:
            plt.ylim([-5., 100])
            plt.yticks([0, 50])
        if subp == 3:
            plt.ylim([-5., 25])
            plt.yticks([0, 20])

    fig.text(0.04, 0.65, 'Population rate [1/s]', va='center', ha='center', rotation='vertical', fontsize=my_size)
    plt.savefig(os.path.join(path_folder, filename_rate + '_bistableALL.pdf'), dpi=200,
                format='pdf', bbox_inches='tight')


def inside_SPW_stimulation(net_4d, filename_rate, use_softplus=True):
    """Creates simulation for rate model under current injection, to produce Fig. 10.

     :param net_4d: class FullParamFromDict
        Contains parameters of the rate model
     :param filename_rate: str
        Filename of rate model
     :param use_softplus: bool
        If True (default), uses the softplus function to simulate the rate model. Alternatively, it uses its
        threshold-linear approximation
    """

    sim = SimParam()
    sim.t_max = 300 * 1e-3
    sim.pulse_lim = 10. * 1e-3
    t_array = np.arange(0, sim.t_max, sim.dt)

    # Initial conditions: are important to start from non-SWR state
    p0 = 0.1
    a0 = 12.
    b0 = 3.

    dIC = 1. / (1. + net_4d.eta * b0)
    IC_resting = [p0, b0, a0, dIC]

    curr = ['I_p', 'I_b', 'I_a']
    num_pbs = [0, 1, 2]
    subplot_d = [4, 5, 6]
    sim_all_current = np.empty((np.shape(t_array)[0], 4, 3))

    for inj_curr, spbs, sd in zip(curr, num_pbs, subplot_d):
        if inj_curr == 'I_p':
            curr_value = 60.  # pA
        elif inj_curr == 'I_b':
            curr_value = 150.
        if inj_curr == 'I_a':
            curr_value = 200.

        sim_all_current[:, :, spbs] = odeint(eq_4d, IC_resting, t_array, printmessg=True,
                                             hmax=0.01, rtol=1e-6, atol=1e-8,
                                             args=(net_4d, sim, curr_value, False,
                                                   inj_curr, use_softplus))

    fig = plt.figure(figsize=[11, 7])
    my_size = 14
    outer = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])
    gs2 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=outer[:, :], hspace=.08, wspace=0.08)

    x_lim_start = sim.pulse_start - 100 * 1e-3  # ms
    x_lim_end = sim.pulse_start + 200 * 1e-3  # ms
    xlim_stim = [x_lim_start, x_lim_end]
    plt.rc('font', size=my_size)

    for curr_to_pop, idx_c in zip(['P', 'B', 'A'], [0, 1, 2]):

        # ============================================== #
        # =================== P cells ================== #
        # ============================================== #
        ax = plt.subplot(gs2[0, idx_c])
        plt.plot(t_array, sim_all_current[:, 0, idx_c], '#ef3b53', label='P', lw=2.5)
        plt.ylim(-5, np.max(sim_all_current[:, 0, :]) + 2)
        plt.yticks(np.arange(0, np.max(sim_all_current[:, 0, :]), 20))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
        if curr_to_pop == 'P':
            adjust_xaxis(ax, xlim_stim, my_size)
            adjust_yaxis(ax, my_size)
        else:
            plt.axis('off')
        # add a textbox with P,B,A activation / inactivation
        font0 = FontProperties()
        my_font = font0.copy()
        # my_font.set_weight('bold')
        if curr_to_pop in ['P', 'B']:
            textstr = curr_to_pop + ' activation'
        else:
            textstr = 'A inactivation'
        props = dict(facecolor='none', edgecolor='none')
        # place a text box in upper left in axes coords
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=my_size, fontproperties=my_font,
                verticalalignment='top', bbox=props)
        if idx_c == 2:
            plt.legend(loc='upper right', prop={'size': my_size}, frameon=False)

        # ============================================== #
        # =================== B cells ================== #
        # ============================================== #
        ax = plt.subplot(gs2[1, idx_c])
        plt.plot(t_array, sim_all_current[:, 1, idx_c], '#3c3fef', label='B', lw=2.5)
        plt.ylim(-5, np.max(sim_all_current[:, 1, :]) + 2)
        plt.yticks(np.arange(0, np.max(sim_all_current[:, 1, :]), 20))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
        if curr_to_pop == 'P':
            adjust_xaxis(ax, xlim_stim, my_size)
            adjust_yaxis(ax, my_size)
        else:
            plt.axis('off')
        if idx_c == 2:
            plt.legend(loc='upper right', prop={'size': my_size}, frameon=False)

        # ============================================== #
        # =================== A cells ================== #
        # ============================================== #
        ax = plt.subplot(gs2[2, idx_c])
        plt.plot(t_array, sim_all_current[:, 2, idx_c], '#0a9045', label='A', lw=2.5)
        plt.ylim(-5, np.max(sim_all_current[:, 2, :]) + 2)
        plt.yticks(np.arange(0, np.max(sim_all_current[:, 2, :]), 10))
        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
        if curr_to_pop == 'P':
            adjust_xaxis(ax, xlim_stim, my_size)
            adjust_yaxis(ax, my_size)
        else:
            plt.axis('off')
        if idx_c == 2:
            plt.legend(loc='lower right', prop={'size': my_size}, frameon=False)

        # ============================================== #
        # ================= depression ================= #
        # ============================================== #
        ax = plt.subplot(gs2[3, idx_c])
        plt.plot(t_array, sim_all_current[:, 3, idx_c], '#e67e22', label='d', lw=2.5)
        plt.ylim(np.min(sim_all_current[:, 3, :]) - 0.05, 1.05)
        plt.yticks(np.arange(1., np.min(sim_all_current[:, 3, :]), -0.2))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        add_sign_of_stimulation(ax, sim.pulse_start, sim.pulse_lim)
        if curr_to_pop == 'P':
            plt.ylabel('\n'.join(wrap('Synaptic efficacy', 10)), fontsize=my_size)
            adjust_xaxis(ax, xlim_stim, my_size)
            adjust_yaxis(ax, my_size)
        else:
            plt.axis('off')
        # add scalebar
        ob = AnchoredHScaleBar(size=50 * 1e-3, label="50 ms", loc=3, frameon=False, extent=0,
                               pad=1., sep=4, color="Black",
                               borderpad=0.1,
                               my_size=my_size)  # pad and borderpad can be used to modify the position of the bar a bit
        ax.add_artist(ob)

        # add arrow where stimulation is
        # one of the two x,y has to be inside axis lim, otherwise noting is shown!
        ax.annotate("", xy=(sim.pulse_start, 0.35), xytext=(sim.pulse_start, 0.15), xycoords='data',
                    arrowprops=dict(arrowstyle="->", lw=2.))

    # add a common y label for population rate
    fig.text(0.085, 0.6, 'Population rate [1/s]', va='center', ha='center', rotation='vertical', fontsize=my_size)
    plt.savefig(os.path.join(path_folder, filename_rate + '_stim_from_spiking.png'), dpi=300,
                format='png', bbox_inches='tight')
