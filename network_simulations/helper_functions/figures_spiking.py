__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains the code to reproduce Figures 2, 6-10, 2-1, 6-1 of the manuscript"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
from scipy.signal import filtfilt
from textwrap import wrap
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import sys
import os
sys.path.append(os.path.dirname( __file__ ) + '/../')

from run_spiking import path_folder
from helper_functions.utils_spiking import adjust_xaxis, adjust_yaxis, \
    add_sign_of_stimulation, create_butter_bandpass, \
    AnchoredHScaleBar, softplus_func, shaded_gradient, inside_analyze_spont, inside_analyze_evoked, fit_func, \
    define_Ifcurve_weighted_sum, adjust_axes_spont, create_connections, average_firing_rate
from helper_functions.detect_peaks import detect_peaks

# Import functions from bifurcation analysis code
sys.path.append(os.path.dirname( __file__ ) + '/../')
import bifurcation_analysis.figures_code.helper_functions.bifurcations as bif
import bifurcation_analysis.figures_code.helper_functions.nullclines as nc
import bifurcation_analysis.figures_code.helper_functions.model as model
import bifurcation_analysis.figures_code.helper_functions.params as params
import bifurcation_analysis.figures_code.helper_functions.aux_functions as aux

# FIG 2
def fig_bistability_manuscript(filename, save_targets=True):
    """Figure 2 of manuscript, showing bistability in the spiking network
    :param filename:
        Name of spiking filename
    :param save_targets: bool
        If True, saves the firing rates of population in non-SWR and SWR states with synaptic depression clamped at
        d=0.5
    """

    filename_to_save = os.path.join(path_folder, filename + 'curr_and_depr_bistability' + '.npz')
    data = np.load(filename_to_save, encoding='latin1', allow_pickle=True)  # all elements are unitless (/Hz and ms)
    data_dic = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    t_array = data_dic['t_array']
    p_array = data_dic['p_array']
    b_array = data_dic['b_array']
    a_array = data_dic['a_array']
    warm_up_time = data_dic['warm_up_time']
    simtime_current = data_dic['simtime_current']
    mean_depr_array = data_dic['mean_depr_array']
    extracurr_array = data_dic['extracurr_array']

    if save_targets:
        window_average = 300  # ms
        idx_nonSWR = np.argmin(np.abs(t_array - (warm_up_time - 500)))         # ms
        target_P_nonSWR = np.mean(p_array[idx_nonSWR - window_average:idx_nonSWR + window_average])
        target_B_nonSWR = np.mean(b_array[idx_nonSWR - window_average:idx_nonSWR + window_average])
        target_A_nonSWR = np.mean(a_array[idx_nonSWR - window_average:idx_nonSWR + window_average])
        idx_SWR = np.argmin(np.abs(t_array - (warm_up_time + 500)))         # ms
        target_P_SWR = np.mean(p_array[idx_SWR - window_average:idx_SWR + window_average])
        target_B_SWR = np.mean(b_array[idx_SWR - window_average:idx_SWR + window_average])
        target_A_SWR = np.mean(a_array[idx_SWR - window_average:idx_SWR + window_average])

        # we are using last values (for A injection) to save. No big differences across stimulation paradigms
        filename_full = os.path.join(path_folder, filename + '_target_FR.npz')
        data_to_save = {'target_P_nonSWR': target_P_nonSWR,
                        'target_B_nonSWR': target_B_nonSWR,
                        'target_A_nonSWR': target_A_nonSWR,
                        'target_P_SWR': target_P_SWR,
                        'target_B_SWR': target_B_SWR,
                        'target_A_SWR': target_A_SWR}

        np.savez_compressed(filename_full, **data_to_save)

    fig = plt.figure(figsize=[10, 9])
    my_size = 16
    plt.rc('text', usetex=True)
    rc('font', size=16)
    x_lim_start = warm_up_time - 1000  # ms
    x_lim_end = simtime_current  # ms

    xlim_bistable = [x_lim_start, x_lim_end]

    outer = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 1, 1, 1])
    outer.update(wspace=0.8, hspace=0.5)
    gs2 = gridspec.GridSpecFromSubplotSpec(5, 3, subplot_spec=outer[:, :], hspace=.08, wspace=0.08)

    # ============================================== #
    # =================== P cells ================== #
    # ============================================== #
    ax = subplot(gs2[0, 0:3])
    plt.plot(t_array, p_array, '#ef3b53', label='P', lw=2.5)
    ylim(-5, np.max(p_array) + 2)
    yticks(np.arange(0, np.max(p_array), 50))
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_bistable, my_size)

    # add a textbox with A, B
    font0 = FontProperties()
    my_font = font0.copy()
    textstr = r'\textbf{A}'
    props = dict(facecolor='none', edgecolor='none')
    ax.annotate(textstr, xy=(0.095, 1.05), xytext=(0.095, 1.15), xycoords='axes fraction',
                fontsize=my_size - 2, ha='center', va='bottom',
                bbox=dict(boxstyle='square', fc='white', ec='white'),
                arrowprops=dict(arrowstyle='-[, widthB=3.6, lengthB=.2', lw=2.0))

    textstr = r'\textbf{B}'
    props = dict(facecolor='none', edgecolor='none')
    ax.annotate(textstr, xy=(0.295, 1.05), xytext=(0.295, 1.15), xycoords='axes fraction',
                fontsize=my_size - 2, ha='center', va='bottom',
                bbox=dict(boxstyle='square', fc='white', ec='white'),
                arrowprops=dict(arrowstyle='-[, widthB=3.9, lengthB=.2', lw=2.0))

    # ============================================== #
    # =================== B cells ================== #
    # ============================================== #
    ax = subplot(gs2[1, 0:3])
    plot(t_array, b_array, '#3c3fef', label='B', lw=2.5)
    ylim(-5, np.max(b_array) + 2)
    yticks(np.arange(0, np.max(b_array), 50))
    plt.setp(ax.get_yticklabels()[1::2], visible=False)
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_bistable, my_size)

    # ============================================== #
    # =================== A cells ================== #
    # ============================================== #
    ax = subplot(gs2[2, 0:3])
    plot(t_array, a_array, '#0a9045', label='A', lw=2.5)
    ylim(-5, np.max(a_array) + 2)
    yticks(np.arange(0, np.max(a_array), 20))
    adjust_xaxis(ax, xlim_bistable, my_size)
    adjust_yaxis(ax, my_size)

    # ============================================== #
    # ============ injected current ================ #
    # ============================================== #
    ax = subplot(gs2[3, 0:3])
    plot(t_array, extracurr_array, '#d4b021', label='I', lw=2.5)
    ylim(np.min(extracurr_array) - 0.1, np.max(extracurr_array) + 0.1)
    adjust_xaxis(ax, xlim_bistable, my_size)
    ylabel('\n'.join(wrap('Mean injected current [pA]', 18)), fontsize=my_size)
    adjust_yaxis(ax, my_size)

    # ============================================== #
    # ================= depression ================= #
    # ============================================== #
    ax = subplot(gs2[4, 0:3])
    plot(t_array, mean_depr_array, '#e67e22', label='d', lw=2.5)
    ylim(np.min(mean_depr_array) - 0.05, 1.05)
    yticks(np.arange(1., np.min(mean_depr_array), -0.2))
    yticks([0.2, 0.5, 0.8])
    adjust_xaxis(ax, xlim_bistable, my_size)
    ylabel('\n'.join(wrap('Synaptic efficacy', 10)), fontsize=my_size)
    adjust_yaxis(ax, my_size)

    # add scalebar
    ob = AnchoredHScaleBar(size=500, label="0.5 s", loc=3, frameon=False, extent=0,
                           pad=1., sep=4, color="Black",
                           borderpad=-0.5,
                           my_size=my_size)  # pad and borderpad can be used to modify the position of the bar a bit
    ax.add_artist(ob)

    # add arrow where stimulation is
    ax.annotate("", xy=(warm_up_time, 0.0), xytext=(warm_up_time, -0.2), xycoords='data',
                arrowprops=dict(arrowstyle="->", lw=2.))

    # add a common y label for population rate
    fig.text(0.065, 0.7, 'Population rate [spikes/s]', va='center', ha='center', rotation='vertical', fontsize=my_size)

    savefig(os.path.join(path_folder, filename + '_bistability.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# FIG 6
def figure_6(filename, fraction_stim=0.6, for_defense=False):
    """Fig 6 manuscript. Includes raster plots and population activity, plus synaptic efficacy in the spontaneous
    scenario (panel A) and in case of activation of P cells (panel B), B cells (panel C), or inactivation of A cells
    (panel D).
    :param filename: str
        Name of spiking filename
    :param fraction_stim: float, [0,1]
        Fraction of cells stimulated (to plot the right file)
    """

    filename_saved = os.path.join(path_folder, filename +
                                  '_sim_fig6_fraction_' + str(fraction_stim) + '.npz')
    data = np.load(filename_saved, encoding='latin1', allow_pickle=True)  # all elements are unitless (/Hz and ms)
    info_dic = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    t_array = info_dic['t_array']
    p_array = info_dic['p_array']
    b_array = info_dic['b_array']
    a_array = info_dic['a_array']
    mean_depr_array = info_dic['mean_depr_array']
    warm_up_time = info_dic['warm_up_time']
    time_with_curr = info_dic['time_with_curr']
    sigma_P = info_dic['sigma_P']
    sigma_B = info_dic['sigma_B']
    sigma_A = info_dic['sigma_A']
    spikes_dic = info_dic['spikes_dic'].item()
    mean_B_input_p = info_dic['mean_B_input_p']
    compress_step_indices = int(info_dic['compress_step_indices'])

    fig = plt.figure(figsize=[14, 11])
    my_size = 16
    plt.rc('text', usetex=True)
    rc('font', size=16)
    # xlim used for stimulation case
    x_lim_start = warm_up_time - 100  # ms
    x_lim_end = warm_up_time + 200  # ms
    xlim_stim = [x_lim_start, x_lim_end]
    xlim_spont = [150, 450]     # NOTE: specific to the simulation with seed!!

    outer = gridspec.GridSpec(8, 4, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=outer[0:3, :], hspace=0.0, wspace=0.08)
    gs2 = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=outer[3:, :], hspace=.08, wspace=0.08)

    # first column is spontaneous - does not matter which idx_c you take, they are the same in spontaneous case
    idx_c = 0
    ax = subplot(gs1[0, idx_c])
    ip, tp = spikes_dic['P_' + str(idx_c)]
    plt.scatter(tp / ms, ip, marker='.', color='#ef3b53', s=0.45, rasterized=True)
    xlim(xlim_spont)
    plt.axis('off')
    ylim([0, 8200])

    font0 = FontProperties()
    my_font = font0.copy()
    textstr = 'Spontaneous'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=my_size, fontproperties=my_font,
            verticalalignment='top', bbox=props)

    textstr = r'\textbf{A}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(0.0, 1.45, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props)

    ax = subplot(gs1[1, idx_c])
    ib, tb = spikes_dic['B_' + str(idx_c)]
    plt.scatter(tb / ms, ib, marker='.', color='#3c3fef', s=0.75, rasterized=True)
    xlim(xlim_spont)
    ylim([0, 135])
    plt.axis('off')

    ax = subplot(gs1[2, idx_c])
    ia, ta = spikes_dic['A_' + str(idx_c)]
    plt.scatter(ta / ms, ia, marker='.', color='#0a9045', s=0.85, rasterized=True)
    xlim(xlim_spont)
    ylim([0, 50])
    plt.axis('off')

    # Population FR
    # P cells
    ax = subplot(gs2[0, idx_c])
    plt.plot(t_array[idx_c, :], p_array[idx_c, :], '#ef3b53', label='P', lw=2.5)
    ylim(-5, np.max(p_array) + 2)
    yticks(np.arange(0, np.max(p_array), 50))
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_spont, my_size)

    # B cells
    ax = subplot(gs2[1, idx_c])
    plot(t_array[idx_c, :], b_array[idx_c, :], '#3c3fef', label='B', lw=2.5)
    ylim(-5, np.max(b_array) + 2)
    yticks(np.arange(0, np.max(b_array), 50))
    plt.setp(ax.get_yticklabels()[1::2], visible=False)
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_spont, my_size)

    # A cells
    ax = subplot(gs2[2, idx_c])
    plot(t_array[idx_c, :], a_array[idx_c, :], '#0a9045', label='A', lw=2.5)
    ylim(-5, np.max(a_array) + 2)
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_spont, my_size)

    # depression
    ax = subplot(gs2[3, idx_c])
    plot(t_array[idx_c, :], mean_depr_array[idx_c, :], '#e67e22', label='d', lw=2.5)
    ylim(np.min(mean_depr_array) - 0.05, 1.05)
    yticks(np.arange(1., np.min(mean_depr_array), -0.2))
    plt.setp(ax.get_yticklabels()[1::2], visible=False)
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_spont, my_size)
    ylabel('\n'.join(wrap('Synaptic efficacy', 10)), fontsize=my_size)

    # ripple component
    ax = subplot(gs2[4, idx_c])
    # band pass filtered version of B input to P cells
    # Sample rate and desired cutoff frequencies (in Hz)
    fs = 1e4 / compress_step_indices
    lowcut = 90
    highcut = 180
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2)
    y_p = filtfilt(b, a, -mean_B_input_p[idx_c, :])
    plot(t_array[idx_c, :], y_p, '#3c3fef', lw=1.5)
    ylim([-120, 120])
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, xlim_spont, my_size)
    ylabel('\n'.join(wrap('Band-pass LFP [a.u.]', 11)), fontsize=my_size)

    # add scalebar
    ob = AnchoredHScaleBar(size=50, label="50 ms", loc=4, frameon=False, extent=0,
                           pad=.0, sep=4, color="Black",
                           borderpad=0.1,
                           my_size=my_size)  # pad and borderpad can be used to modify the position of the bar a bit
    ax.add_artist(ob)

    for curr_to_pop, idx_c, text_letter in zip(['P', 'B', 'A'], [0, 1, 2],
                                               [r'\textbf{B}', r'\textbf{C}', r'\textbf{D}']):
        # ============= raster ALL cells =============== #
        ax = subplot(gs1[0, idx_c + 1])
        ip, tp = spikes_dic['P_' + str(idx_c)]
        plt.scatter(tp / ms, ip, marker='.', color='#ef3b53', s=0.45, rasterized=True)
        xlim(xlim_stim)
        ylim([0, 8200])
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

        # add a textbox with P,B,A activation / inactivation
        font0 = FontProperties()
        my_font = font0.copy()
        if curr_to_pop in ['P', 'B']:
            textstr = curr_to_pop + ' activation'
        else:
            textstr = 'A inactivation'
        props = dict(facecolor='none', edgecolor='none')
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=my_size, fontproperties=my_font,
                verticalalignment='top', bbox=props)

        props = dict(facecolor='none', edgecolor='none')
        ax.text(0.0, 1.45, text_letter, transform=ax.transAxes, fontsize=my_size + 4,
                verticalalignment='top', bbox=props)

        ax = subplot(gs1[1, idx_c + 1])
        ib, tb = spikes_dic['B_' + str(idx_c)]
        plt.scatter(tb / ms, ib, marker='.', color='#3c3fef', s=0.75, rasterized=True)
        xlim(xlim_stim)
        ylim([0, 135])
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

        ax = subplot(gs1[2, idx_c + 1])
        ia, ta = spikes_dic['A_' + str(idx_c)]
        plt.scatter(ta / ms, ia, marker='.', color='#0a9045', s=0.85, rasterized=True)
        xlim(xlim_stim)
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

        # =================== P cells ================== #
        ax = subplot(gs2[0, idx_c + 1])
        plt.plot(t_array[idx_c, :], p_array[idx_c, :], '#ef3b53', label='P', lw=2.5)
        ylim(-5, np.max(p_array) + 2)
        yticks(np.arange(0, np.max(p_array), 50))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)
        adjust_xaxis(ax, xlim_stim, my_size)

        # =================== B cells ================== #
        ax = subplot(gs2[1, idx_c + 1])
        plot(t_array[idx_c, :], b_array[idx_c, :], '#3c3fef', label='B', lw=2.5)
        ylim(-5, np.max(b_array) + 2)
        yticks(np.arange(0, np.max(b_array), 50))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)
        adjust_xaxis(ax, xlim_stim, my_size)

        # =================== A cells ================== #
        ax = subplot(gs2[2, idx_c + 1])
        plot(t_array[idx_c, :], a_array[idx_c, :], '#0a9045', label='A', lw=2.5)
        ylim(-5, np.max(a_array) + 2)
        yticks(np.arange(0, np.max(a_array), 50))
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)
        adjust_xaxis(ax, xlim_stim, my_size)

        # ================= depression ================= #
        ax = subplot(gs2[3, idx_c + 1])
        plot(t_array[idx_c, :], mean_depr_array[idx_c, :], '#e67e22', label='d', lw=2.5)
        ylim(np.min(mean_depr_array) - 0.05, 1.05)
        yticks(np.arange(1., np.min(mean_depr_array), -0.2))
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)
        adjust_xaxis(ax, xlim_stim, my_size)

        # ============================================== #
        # ================= ripple component =========== #
        # ============================================== #
        ax = subplot(gs2[4, idx_c + 1])
        # band pass filtered version of inh input (high passed)
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 1e4 / compress_step_indices
        lowcut = 90
        highcut = 180
        b, a = create_butter_bandpass(lowcut, highcut, fs, order=2)
        y_p = filtfilt(b, a, -mean_B_input_p[idx_c, :])
        plot(t_array[idx_c, :], y_p, '#3c3fef', lw=1.5)
        ylim([-120, 120])
        plt.axis('off')
        add_sign_of_stimulation(ax, warm_up_time, time_with_curr)
        adjust_xaxis(ax, xlim_stim, my_size)

        ob = AnchoredHScaleBar(size=50, label="50 ms", loc=4, frameon=False, extent=0,
                               pad=0., sep=4, color="Black",
                               borderpad=0.1,
                               my_size=my_size)  # pad and borderpad can be used to modify the position of the bar a bit
        ax.add_artist(ob)

        # add arrow where stimulation is
        ax.annotate("", xy=(warm_up_time, -100.), xytext=(warm_up_time, -160.), xycoords='data',
                    arrowprops=dict(arrowstyle="->", lw=2.))

    fig.text(0.085, 0.44, 'Population rate [spikes/s]', va='center', ha='center', rotation='vertical', fontsize=my_size)

    if for_defense:
        fig.text(0.1, 0.86, 'P cells', va='center', ha='right', fontsize=my_size)
        fig.text(0.1, 0.75, 'B cells', va='center', ha='right', fontsize=my_size)
        fig.text(0.1, 0.65, 'A cells', va='center', ha='right', fontsize=my_size)

    else:
        fig.text(0.12, 0.86, '$N_P = 8200$', va='center', ha='right', fontsize=my_size)
        fig.text(0.12, 0.75, '$N_B = 135$', va='center', ha='right', fontsize=my_size)
        fig.text(0.12, 0.65, '$N_A = 50$', va='center', ha='right', fontsize=my_size)

    savefig(os.path.join(path_folder, filename + '_fig6' + '_fraction_' + str(
        fraction_stim) + '.png'),
            dpi=300, format='png', bbox_inches='tight')


# FIG 7
def figure_7(filename, simtime_current=10 * 60 * second):
    """Analyze spontaneous and evoked SWR simulations to study SWR dynamics (properties and correlation).
    Produces Fig. 7 of the manuscript

    :param filename: str
        Name of spiking filename
    :param simtime_current: Brian second
        Duration of simulation (to plot the right file)
    """

    # Spontaneous SWR simulation
    print('**Spontaneous')
    filename_full = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + '.npz')
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    IEI_end_start_FWHM, amp_peaks, durations_spont, t_spont, trace_spont, filt_trace = \
        inside_analyze_spont(info_dictionary, use_b_input=True)


    # Evoked (and spontaneous) SWR simulation
    print('**Evoked')
    filename_evoked_sim = os.path.join(path_folder, filename + '_evoked_simtime_' + str(int(simtime_current / second)) + '.npz')
    data = np.load(filename_evoked_sim, encoding='latin1', allow_pickle=True)
    info_dictionary_evoked = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    stim_times_array = info_dictionary_evoked['stim_times_array']

    IEI_end_start_FWHM_NEXT_evoked, IEI_end_start_FWHM_PREV_evoked, amp_evoked, durations_evoked, \
        t_evoked, trace_evoked, filt_trace_evoked = inside_analyze_evoked(info_dictionary_evoked, use_b_input=True)

    figure(figsize=[17.6/2.54, 0.5*17.6/2.54])
    my_size = 9
    gs1 = gridspec.GridSpec(3, 12)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rc('text', usetex=True)

    gs1.update(hspace =.6, wspace=7)

    # ============= ****Spontaneous SWRs
    ax = subplot(gs1[0, 0:6])
    t_plot_start = np.argmin(np.abs(t_spont - 24))
    t_plot_end = np.argmin(np.abs(t_spont - 27))
    plot(t_spont[t_plot_start:t_plot_end], trace_spont[t_plot_start:t_plot_end], 'k', lw=0.3, zorder=1)
    plot(t_spont[t_plot_start:t_plot_end], filt_trace[t_plot_start:t_plot_end], 'DodgerBlue', label='filtered',
         lw=1.5, zorder=2)
    ylim(-5, 100)
    ylabel('\n'.join(wrap('Filtered $B$ input to $P$ [pA]', 18)), fontsize=my_size)
    my_xlim = [t_spont[t_plot_start], t_spont[t_plot_end]]
    title('Spontaneous', fontsize=my_size)
    adjust_yaxis(ax, my_size)
    plt.setp(ax.get_yticklabels()[0::2], visible=False)
    adjust_xaxis(ax, my_xlim, my_size)

    ob = AnchoredHScaleBar(size=0.2, label="200 ms", loc=1, frameon=False, extent=0,
                           pad=0.5, sep=4., color="Black", borderpad=-1, my_size=my_size)
    ax.add_artist(ob)

    # ================ SWR properties
    for sub_n, my_data, my_label in zip([gs1[1, 0:2], gs1[1, 2:4], gs1[1, 4:6]],
                                        [IEI_end_start_FWHM, amp_peaks, durations_spont],
                                        ['IEI [sec]', 'Amplitude [pA]', 'FWHM [ms]']):
        ax = subplot(sub_n)
        plt.hist(my_data, bins=30, color='k', density=True)
        adjust_yaxis(ax, my_size)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        adjust_xaxis(ax, None, my_size, show_bottom=True)
        title(my_label, fontsize=my_size)
        if my_label == 'Amplitude [pA]':
            ax.set_yticks([0, 0.08, 0.16])
            xlim([0, 100])
            ax.set_xticks([0, 50, 100])
        elif my_label == 'FWHM [ms]':
            xlim([0, 135])
            ylim([0, 0.25])
            ax.set_xticks([0, 60, 120])
            ax.set_yticks([0, 0.1, 0.2])
        else:
            ylabel('\n'.join(wrap('Prob. density [a.u.]', 15)), fontsize=my_size)
            xlim([0, 2.5])
            ax.set_xticks([0, 1, 2])
            ylim([0, 2.5])
            ax.set_yticks([0, 2])

    # ================ Correlation spontaneous SWR events - IEI
    ax = subplot(gs1[2, 0:3])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[1:], facecolor='k',
                s=2)
    plt.axvline(x=0.188, linewidth=1, color='k', linestyle='--')
    xlabel('Previous IEI [sec]', fontsize=my_size)
    ylabel('\n'.join(wrap('Amplitude [pA]', 15)), fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[1:])
    adjust_axes_spont(ax, c, p, my_size)
    xlim([0, 2.2])
    ylim([40, 80])
    yticks([40, 70])
    xticks([0, 1, 2])

    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM, amp_peaks[1:], bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM), np.max(IEI_end_start_FWHM), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), 'r', lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant spont:', 1. / popt[1] * 1e3, ' ms')

    ax = subplot(gs1[2, 3:6])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[:-1], facecolor='k',
             edgecolor='k', s = 2)
    xlabel('Next IEI [sec]', fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[:-1])
    adjust_axes_spont(ax, c, p, my_size)
    ylim([40, 80])
    xlim([0, 2.2])
    yticks([40, 70])
    xticks([0, 1, 2])

    # ============= ****Evoked SWRs
    ax = subplot(gs1[0, 6:])
    curr_duration = info_dictionary_evoked['time_with_stim']
    # Use this code to choose a stretch to show!
    '''
    figure()
    plot(t_evoked, filt_trace_evoked)
    [axvline(_x, linewidth=1, color='y', zorder=2) for _x in stim_times_array]
    show()
    '''
    t_st = 51.5     # simulation specific
    t_end = 54.5    # simulation specific

    start_idx = (np.abs(t_evoked - t_st)).argmin()
    end_idx = (np.abs(t_evoked - t_end)).argmin()
    my_lim = [t_evoked[start_idx], t_evoked[end_idx]]
    plot(t_evoked[start_idx:end_idx], trace_evoked[start_idx:end_idx], 'k', lw=0.3, label='filtered', zorder=1)
    plot(t_evoked[start_idx:end_idx], filt_trace_evoked[start_idx:end_idx], 'DodgerBlue', label='filtered', lw=1.5,
         zorder=2)
    ylim(-5, 100)

    for el in stim_times_array:
        if (el > t_st) and (el < t_end):
            ix = np.linspace(el, (el + curr_duration))
            iy = np.linspace(500, 500)
            verts = [(el, -500)] + list(zip(ix, iy)) + [
                ((el + curr_duration), -500)]
            poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', zorder=-3)
            ax.add_patch(poly)
            ax.annotate("", xy=(el, 0.0), xytext=(el, -30.), xycoords='data',
                        arrowprops=dict(arrowstyle="->", lw=1.))

    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, my_lim, my_size, show_bottom=False)
    plt.setp(ax.get_yticklabels()[0::2], visible=False)
    ob = AnchoredHScaleBar(size=0.2, label="200 ms", loc=1, frameon=False, extent=0,
                           pad=0.5, sep=4., color="Black", borderpad=-1, my_size=my_size)

    ax.add_artist(ob)
    title('Evoked', fontsize=my_size)

    # ================ Properties of evoked SWRs
    for sub_n, my_data, my_label in zip([gs1[1, 6:8], gs1[1, 8:10], gs1[1, 10:]],
                                        [np.hstack((IEI_end_start_FWHM_NEXT_evoked, IEI_end_start_FWHM_PREV_evoked)), amp_evoked,
                                         durations_evoked],
                                        ['IEI [sec]', 'Amplitude [pA]', 'FWHM [ms]']):

        ax = subplot(sub_n)
        plt.hist(my_data, bins=30, color='k', density=True)
        adjust_yaxis(ax, my_size)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        adjust_xaxis(ax, None, my_size, show_bottom=True)
        title(my_label, fontsize=my_size)
        if my_label == 'Amplitude [pA]':
            xlim([0, 100])
            ax.set_yticks([0, 0.08])
            ax.set_xticks([0, 50, 100])
        elif my_label == 'FWHM [ms]':
            xlim([0, 135])
            ylim([0, 0.25])
            ax.set_xticks([0, 60, 120])
            ax.set_yticks([0, 0.1, 0.2])
        else:
            xlim([0, 2.5])
            ax.set_xticks([0, 1, 2])
            ylim([0, 2.5])
            ax.set_yticks([0, 2])

    # ================= Correlation of evoked SWR amplitude - IEI
    ax = subplot(gs1[2, 6:9])
    scatter(IEI_end_start_FWHM_NEXT_evoked, amp_evoked, facecolor='k',
            s=2)
    plt.axvline(x=0.082, linewidth=1, color='k', linestyle='--')
    xlabel('Previous IEI [sec]', fontsize=my_size)
    ylabel('\n'.join(wrap('Amplitude [pA]', 40)), fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM_NEXT_evoked, amp_evoked)
    adjust_axes_spont(ax, c, p, my_size)
    ylim([40, 80])
    xlim([0, 2.2])
    yticks([40, 70])
    xticks([0, 1, 2])

    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM_NEXT_evoked, amp_evoked, bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM_NEXT_evoked), np.max(IEI_end_start_FWHM_NEXT_evoked), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), 'r', lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant evoked:', 1. / popt[1] * 1e3, ' ms')

    ax = subplot(gs1[2, 9:])
    scatter(IEI_end_start_FWHM_PREV_evoked, amp_evoked, facecolor='k',
            edgecolor='k', s=2)
    xlabel('\n'.join(wrap('Next IEI [sec]', 16)), fontsize=my_size)
    c, p = pearsonr(amp_evoked, IEI_end_start_FWHM_PREV_evoked)
    adjust_axes_spont(ax, c, p, my_size)
    ylim([40, 85])
    xlim([0, 2.2])
    yticks([40, 70])
    xticks([0, 1, 2])

    savefig(filename_full[:-4] + '_figure7.png', dpi=300, format='png', bbox_inches='tight')


# FIG 8 and 9
def compare_default_to_other_plasticities(filename, simtime_current=10*60*second, t_F=250, eta_F=0.15, max_z=1.,
                                          depr_compare=True):
    """Compare the default case (with B -> A syn. depression) with the case in which either the B->P depression or the
    P->A facilitation is added

    :param filename: str
        Name of spiking filename
    :param simtime_current: Brian second
        Length of simulation (to choose which files to open)
    :param t_F: int
        Time constant of facilitation. Not used when comparing default to B->P depression case
    :param eta_F: float
        Learning rate of facilitation. Not used when comparing default to B->P depression case
    :param max_z: float
        Upper bound for facilitation. Not used when comparing default to B->P depression case
    :param depr_compare: bool
        If True, compare default to the case with additional B->P depression. When False, compare to the case with added
        P-> A facilitation
    """

    print('**Default (with B -> A depression)')
    filename_full = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + '.npz')
    print(filename_full)

    data = np.load(filename_full, encoding='latin1', allow_pickle=True)  # all elements are unitless (/Hz and second)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    IEI_end_start_FWHM, amp_peaks, durations_spont, t_spont, trace_spont, \
    filt_trace = inside_analyze_spont(info_dictionary, use_b_input=True)

    # =============== open file with shorter simulations to show trace and rasters
    T_short = 30  # second
    filename_short = os.path.join(path_folder, filename + '_spont_simtime_' + str(T_short) + '.npz')
    data = np.load(filename_short, encoding='latin1', allow_pickle=True)
    dic_short = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    FRP_short = dic_short['FRP_smooth']
    FRB_short = dic_short['FRB_smooth']
    FRA_short = dic_short['FRA_smooth']
    t_short = dic_short['time_array']
    spikes_short = dic_short['spikes_dic'].item()
    mean_btop_short = -dic_short['mean_b_input_to_p']

    # ======== B to P depression
    if depr_compare:
        # read info from spontaneous with depression B -> P
        filename_change = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + '_deprPB.npz')
        print(filename_change)

        # open file with shorter simulations to show trace and rasters
        filename_short_change = os.path.join(path_folder, filename + '_spont_simtime_' + str(T_short) + '_deprPB.npz')
        print(filename_short_change)

        string_print = '**With extra B->P depression'
        color_change = 'DeepPink'
        light_change = 'Pink'

    # ============ P to A facilitation
    else:
        filename_change = os.path.join(path_folder, filename + '_simtime_' + str(int(simtime_current / second)) + '_tauF_' + \
                          str(int(t_F)) + '_etaF_' + str(eta_F) + '_maxz_' + str(max_z) + '_BtoAdepr.npz')
        print(filename_change)

        # open file with shorter simulations to show trace and rasters
        filename_short_change = os.path.join(path_folder, filename + '_simtime_' + str(T_short) + '_tauF_' + \
                                str(int(t_F)) + '_etaF_' + str(eta_F) + '_maxz_' + str(max_z) + '_BtoAdepr.npz')
        print(filename_short_change)

        string_print = '**With extra P->A facilitation'
        color_change = 'SeaGreen'
        light_change = 'MediumSeaGreen'

    data = np.load(filename_change, encoding='latin1', allow_pickle=True)
    info_dictionary_change = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    print(string_print)

    IEI_end_start_FWHM_change, amp_peaks_change, durations_spont_change, t_spont_change, \
    trace_spont_change, filt_trace_change = inside_analyze_spont(info_dictionary_change, use_b_input=True)

    data = np.load(filename_short_change, encoding='latin1', allow_pickle=True)  # all elements are unitless (/Hz and second)
    dic_short_change = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    FRP_short_change = dic_short_change['FRP_smooth']
    FRB_short_change = dic_short_change['FRB_smooth']
    FRA_short_change = dic_short_change['FRA_smooth']
    t_short_change = dic_short_change['time_array']
    spikes_short_change = dic_short_change['spikes_dic'].item()
    mean_btop_short_change = -dic_short_change['mean_b_input_to_p']
    compress_step_indices = int(dic_short_change['compress_step_indices'])

    # Filter short traces here
    fs = 1e4 / compress_step_indices
    lowcut = -1.
    highcut = 5.
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2, btype='low')
    filt_short = filtfilt(b, a, mean_btop_short)
    filt_short_depr = filtfilt(b, a, mean_btop_short_change)

    # find peaks to align traces to a given peak
    idx_peaks_short = detect_peaks(filt_short, mph=30, mpd=int(1000 / compress_step_indices),
                                   show=False)
    idx_peaks_short = idx_peaks_short[1:-1]
    idx_peaks_short_change = detect_peaks(filt_short_depr, mph=30, mpd=int(1000 / compress_step_indices),
                                          show=False)

    idx_peaks_short_change = idx_peaks_short_change[1:-1]
    which_peak = 20     # arbitrary choice

    # figure()
    # plot(filt_short_depr)
    # show()

    my_size = 9
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rc('text', usetex=True)

    if depr_compare:
        fig = plt.figure(figsize=[17.6 / 2.54, 1.44 * 17.6 / 2.54])
        gs1 = gridspec.GridSpec(7, 6, height_ratios=[1, 1, 1, 1, 1, 0., 0.8])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs1[1:3, :], hspace=0.0, wspace=0.8)
        gs3 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs1[3:5, :], hspace=0.5, wspace=0.8)
        gs4 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs1[6, :], hspace=0.0, wspace=1.)
    else:
        fig = plt.figure(figsize=[17.6 / 2.54, 1.2 * 17.6 / 2.54])
        gs1 = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 1, 1, 1])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs1[1:3, :], hspace=0.0, wspace=0.8)
        gs3 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs1[3:, :], hspace=0.3, wspace=0.8)

    gs1.update(hspace=.3, wspace=.8)
    ax = subplot(gs1[0, 0:6])
    range_around_peak = int(30000 / compress_step_indices)
    peak_distance = t_short[idx_peaks_short[which_peak]] - t_short_change[idx_peaks_short_change[which_peak]]

    if depr_compare:
        my_lab_added = 'with $B$-to-$P$ depression added'
    else:
        my_lab_added = 'with $P$-to-$A$ facilitation added'
    plot(t_short_change[idx_peaks_short_change[which_peak] - range_around_peak:idx_peaks_short_change[which_peak] +
                                                                              range_around_peak] + peak_distance,
         filt_short_depr[idx_peaks_short_change[which_peak] - range_around_peak:idx_peaks_short_change[which_peak] +
                                                                               range_around_peak],
         color=color_change, label=my_lab_added, lw=1.5, zorder=2)
    plot(t_short[idx_peaks_short[which_peak] - range_around_peak:idx_peaks_short[which_peak] + range_around_peak],
         filt_short[idx_peaks_short[which_peak] - range_around_peak:idx_peaks_short[which_peak] + range_around_peak:],
         'k', label='default', lw=1.5, zorder=2)
    my_xlim = [t_short[idx_peaks_short[which_peak]] - 2, t_short[idx_peaks_short[which_peak]] + 2]
    ylim(-5, 120)
    ylabel('\n'.join(wrap('Filtered $B$ input to $P$ [pA]', 17)), fontsize=my_size)  # or should it be pA
    legend(loc='upper left', prop={'size': my_size}, frameon=False)
    plt.setp(ax.get_yticklabels()[::2], visible=False)

    # Create a Rectangle patch
    rect = patches.Rectangle((t_short[idx_peaks_short[which_peak]] - 0.15, -2), 0.3, 80, linewidth=1,
                             linestyle='--', edgecolor='DarkGray', facecolor='none')
    ax.add_patch(rect)

    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, my_xlim, my_size)

    ob = AnchoredHScaleBar(size=0.2, label="200 ms", loc=1, frameon=False, extent=0,
                           pad=0.5, sep=4., color="Black",
                           borderpad=-1,
                           my_size=my_size)
    ax.add_artist(ob)

    textstr = r'\textbf{A}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(-0.16, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props, size = my_size + 4)

    # =============== zoom into the traces for short sims
    event_range = int(2000 / compress_step_indices)  # take the 200ms before and after peak

    x1 = idx_peaks_short[which_peak] - event_range
    x2 = idx_peaks_short[which_peak] + event_range
    x3 = idx_peaks_short_change[which_peak] - event_range
    x4 = idx_peaks_short_change[which_peak] + event_range

    for fr_disp, fr_disp_depr, my_title, subn, max_rec in zip([FRP_short, FRB_short, FRA_short],
                                                              [FRP_short_change, FRB_short_change, FRA_short_change],
                                                              ['$P$ cells', '$B$ cells', '$A$ cells'], [0, 2, 4],
                                                              [130, 220, 28]):
        ax = subplot(gs2[0, subn:subn + 2])
        plot(fr_disp[x1:x2], 'k', lw=1.3)
        plot(fr_disp_depr[x3:x4], color=color_change, lw=1.)
        title(my_title, fontsize=my_size)
        adjust_yaxis(ax, my_size)
        if subn == 4:
            ylim([-0.2, max_rec + 1])
        else:
            ylim([-3, max_rec + 2])
        adjust_xaxis(ax, [0, 400], my_size)
        if subn == 0:
            ylabel('\n'.join(wrap('Pop. firing rate [spikes/s]', 20)), fontsize=my_size)
        ob = AnchoredHScaleBar(size=50, label="50 ms", loc=2, frameon=False, extent=0,
                               pad=0.5, sep=4., color="Black",
                               borderpad=0.5, my_size=my_size)
        ax.add_artist(ob)

        rect = patches.Rectangle((50, -0.2), 300, max_rec, linewidth=1, linestyle='--', edgecolor='DarkGray',
                                 facecolor='none')
        ax.add_patch(rect)

        if subn == 0:
            textstr = r'\textbf{B}'
            props = dict(facecolor='none', edgecolor='none')
            ax.text(-0.55, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
                    verticalalignment='top', bbox=props, size=my_size + 4)

    for subn, dic_key, pop_size, mark_size in zip([0, 2, 4], ['P', 'B', 'A'], [8200, 135, 50], [0.15, 0.2, 0.25]):
        ax = subplot(gs2[1, subn:subn + 2])
        ip, tp = spikes_short[dic_key]
        il = np.where(np.logical_and(tp > t_short[x1], tp < t_short[x2]))[0]
        ax.scatter((tp[il] - (t_short[idx_peaks_short[which_peak]] - event_range)), ip[il], marker='.', color='k',
                   s=mark_size, rasterized=True)
        ipd, tpd = spikes_short_change[dic_key]
        ilchange = np.where(np.logical_and(tpd > t_short_change[x3], tpd < t_short_change[x4]))[0]
        ax.scatter((tpd[ilchange] - (t_short_change[idx_peaks_short_change[which_peak]] - event_range)),
                   ipd[ilchange] + pop_size, marker='.', color=color_change, s=mark_size, rasterized=True)
        xrast_min = np.min(
            np.hstack((tpd[ilchange] - (t_short_change[idx_peaks_short_change[which_peak]] - event_range),
                       tp[il] - (t_short[idx_peaks_short[which_peak]] - event_range))))
        xrast_max = np.max(
            np.hstack((tpd[ilchange] - (t_short_change[idx_peaks_short_change[which_peak]] - event_range),
                       tp[il] - (t_short[idx_peaks_short[which_peak]] - event_range))))
        xlim([xrast_min, xrast_max])
        plt.axis('off')

    if depr_compare:
        fig.text(0.1, 0.597, '\n'.join(wrap('with $B$-to-$P$ depression added', 15)), va='center', ha='right', fontsize=my_size,
                 color=color_change)
        fig.text(0.1, 0.557, 'default', va='center', ha='right', fontsize=my_size)
    else:
        fig.text(0.1, 0.545, '\n'.join(wrap('with $P$-to-$A$ facilitation added', 15)), va='center', ha='right', fontsize=my_size,
                 color=color_change)
        fig.text(0.1, 0.49, 'default', va='center', ha='right', fontsize=my_size)

    # ================ SWR properties
    unit_amp = '[pA]'
    for sub_n, my_data_default, my_data_change, my_label in zip([gs3[0, 0:2], gs3[0, 2:4], gs3[0, 4:6]],
                                                                [IEI_end_start_FWHM, amp_peaks, durations_spont],
                                                                [IEI_end_start_FWHM_change, amp_peaks_change,
                                                                 durations_spont_change],
                                                                ['IEI [sec]', 'Amplitude ' + unit_amp, 'FWHM [ms]']):
        ax = subplot(sub_n)
        plt.hist([my_data_default, my_data_change], color=['k', color_change], bins=30, lw=0, density=True)
        adjust_yaxis(ax, my_size)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        adjust_xaxis(ax, None, my_size, show_bottom=True)
        title(my_label, fontsize=my_size)
        if my_label.find('Amplitude') != -1:
            if depr_compare:
                xlim([40, 80])
                ax.set_yticks([0, 0.08, 0.16])
            else:
                xlim([0, 100])
            ax.set_xticks([0,50,100])
        elif my_label.find('FWHM') != -1:
            ax.set_xticks([0, 60, 120])
            ax.set_yticks([0, 0.4, 0.8])
        else:
            if depr_compare:
                xlim([0, 2.5])
                ax.set_xticks([0, 1, 2])
            else:
                xlim([0, 4])
                ax.set_xticks([0, 1, 2, 3, 4])
                ax.set_yticks([0, 0.8, 1.6])
            ylabel('Prob. density [a. u.]', fontsize=my_size)

        if my_label == 'IEI [sec]':
            textstr = r'\textbf{C}'
            props = dict(facecolor='none', edgecolor='none')
            ax.text(-0.55, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
                    verticalalignment='top', bbox=props, size=my_size + 4)

    # ================ correlation spontaneous events
    ax = subplot(gs3[1, 0:3])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[1:], facecolor='k',
                edgecolor='k', s=2, rasterized=True)
    plt.scatter(IEI_end_start_FWHM_change, amp_peaks_change[1:], facecolor=color_change,
                edgecolor=color_change, s=2, rasterized=True)
    title('Previous IEI [sec]', fontsize=my_size)
    ylabel('\n'.join(wrap('Amplitude ' + unit_amp, 40)), fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[1:])  # of default
    adjust_axes_spont(ax, c, p, my_size)
    plt.axvline(x=0.188, linewidth=1., color='k', linestyle='-.')

    if depr_compare:
        plt.axvline(x=0.142, linewidth=1., color=color_change, linestyle='--')
        xlim([0, 2.5])
        xticks([0, 1, 2])
        ylim([20, 80])

    else:
        plt.axvline(x=0.209, linewidth=1., color=color_change, linestyle='--')
        xlim([0, 4])
        xticks([0, 2, 4])
        ylim([40, 80])
    yticks([40, 80])


    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM, amp_peaks[1:], bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM), np.max(IEI_end_start_FWHM), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), 'k', lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant spont:', 1. / popt[1] * 1e3, ' ms')

    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM_change, amp_peaks_change[1:], bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM_change), np.max(IEI_end_start_FWHM_change), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), color=light_change, lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant plastic:', 1. / popt[1] * 1e3, ' ms')

    textstr = r'\textbf{D}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(-0.33, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props, size=my_size + 4)

    ax = subplot(gs3[1, 3:6])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[:-1], facecolor='k',
                edgecolor='k', s = 2, rasterized=True)  # amplitude of previous event - both works
    plt.scatter(IEI_end_start_FWHM_change, amp_peaks_change[:-1], facecolor=color_change,
                edgecolor=color_change, s=2, rasterized=True)  # amplitude of previous event - both works
    ylabel('Amplitude ' + unit_amp, fontsize=my_size)
    title('Next IEI [sec]', fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[:-1])
    adjust_axes_spont(ax, c, p, my_size)
    if depr_compare:
        xlim([0, 2.5])
        xticks([0, 1, 2])
        ylim([20, 80])
    else:
        xlim([0, 4])
        xticks([0, 2, 4])
        ylim([40, 80])
    yticks([40, 80])

    # ================ Rate model bifurcation diagrams
    if depr_compare:
        home_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../'
        bs_def = bif.load_bifurcations(home_dir + 'bifurcation_analysis/bifurcation_diagrams/1param/','e',0,1)
        bs = bif.load_bifurcations(home_dir + 'bifurcation_analysis/bifurcation_diagrams/1param/','e_double',0,1)

        # x ticks:
        e_ticks=[0, 0.25, 0.5, 0.75, 1]
        e_ticklabels=[0,'',0.5,'',1]

        # y ticks:
        P_ticks=[0,50,100]
        B_ticks=[0,100,200]
        A_ticks=[0,5,10]

        # y range:
        pmax = 135
        bmax = 250
        amax = 15

        ax = subplot(gs4[0,0:2])
        textstr = r'\textbf{E}'
        props = dict(facecolor='none', edgecolor='none')
        ax.text(-0.54, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
                verticalalignment='top', bbox=props, size = my_size + 4)

        # Plot e-P bifurcation diagram:
        bif.plot_bifurcation(ax,aux,bs_def,'P',[0,1],pmax,'e',e_ticks,e_ticklabels,P_ticks,P_ticks,my_size,plot_color='black',line_width=1.5,inward_ticks=False)
        bif.plot_bifurcation(ax,aux,bs,'P',[0,1],pmax,'e',e_ticks,e_ticklabels,P_ticks,P_ticks,my_size,plot_color='DeepPink',line_width=1.5,inward_ticks=False)

        ax = subplot(gs4[0, 2:4])
        # Plot e-B bifurcation diagram:
        bif.plot_bifurcation(ax,aux,bs_def,'B',[0,1],bmax,'e',e_ticks,e_ticklabels,B_ticks,B_ticks,my_size,plot_color='black',line_width=1.5,inward_ticks=False)
        bif.plot_bifurcation(ax,aux,bs,'B',[0,1],bmax,'e',e_ticks,e_ticklabels,B_ticks,B_ticks,my_size,plot_color='DeepPink',line_width=1.5,inward_ticks=False)

        # Create grid spanning e and B space:
        E, B = np.meshgrid(np.arange(0, 1, .01), np.arange(-1, 250, .5))
        # Get e nullcline for values in grid:
        dE = model.de(E, B, params.tau_d, params.eta_d)
        # Plot e nullcline:
        nc.plot_nullcline(ax,E,B,dE,'e nullcline','upper right',(1.05,1.05),my_size)

        ax = subplot(gs4[0, 4:6])

        # Plot e-A bifurcation diagram:
        bif.plot_bifurcation(ax,aux,bs_def,'A',[0,1],amax,'e',e_ticks,e_ticklabels,A_ticks,A_ticks,my_size,plot_color='black',line_width=1.5,inward_ticks=False)
        bif.plot_bifurcation(ax,aux,bs,'A',[0,1],amax,'e',e_ticks,e_ticklabels,A_ticks,A_ticks,my_size,plot_color='DeepPink',line_width=1.5,inward_ticks=False)

        # Separating line:
        fig.text(0.5, 0.209, 'Rate model', va='center', ha='center', fontsize=my_size, bbox=dict(facecolor='white', edgecolor='white'))
        line = plt.Line2D([0.06,0.9],[0.21,0.21], linewidth=1, linestyle='--', transform=fig.transFigure, color='DarkGray')
        fig.add_artist(line)

    if depr_compare:
        savefig(
            os.path.join(path_folder, + filename + '_compare_with_BtoPdepression' + '_T_' + str(simtime_current / second) + '.png'),
            dpi=600, format='png', bbox_inches='tight')

    else:
        savefig(
            os.path.join(path_folder, filename + '_compare_with_PtoAfacilitation' + '_T_' + str(simtime_current / second) + '.png'),
            dpi=600, format='png', bbox_inches='tight')


# FIG 10
def plot_facilitationPtoA_effects(filename, simtime_current=10 * 60 * second, t_F=230., eta_F=0.32, max_z=1.,
                                  gab_fac_only=4.5, gba_fac_only=5.5):
    """Compare the default case (with B->A synaptic depression) with the case in which P->A facilitation is the only
    plastic mechanism.

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
    :param gab_fac_only: float
        Value of B-> A conductance update
    :param gba_fac_only: float
        Value of A-> B conductance update
    """

    print('**Default')
    filename_full = os.path.join(path_folder, filename + '_spont_simtime_' + str(int(simtime_current / second)) + '.npz')
    print(filename_full)
    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    FRB_display = info_dictionary['FRB_smooth']
    t_display = info_dictionary['time_array']

    IEI_end_start_FWHM, amp_peaks, durations_spont, t_spont, trace_spont, \
    filt_trace = inside_analyze_spont(info_dictionary, use_b_input=True, detection_thr=40, min_dist_idx=2000)

    print('**With facilitation B->A ONLY')
    filename_fac = os.path.join(path_folder, filename + '_simtime_' + str(int(simtime_current / second)) + '_tauF_' + \
                   str(int(t_F)) + '_etaF_' + str(eta_F) + '_gabfac_' + str(gab_fac_only) + '_gbafac_' + str(
        gba_fac_only) + '_maxz_' + str(max_z) + '.npz')
    print(filename_fac)
    data = np.load(filename_fac, encoding='latin1', allow_pickle=True)
    info_dictionary_fac = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    FRB_display_fac = info_dictionary_fac['FRB_smooth']
    t_display_fac = info_dictionary_fac['time_array']

    IEI_end_start_FWHM_fac, amp_peaks_fac, durations_spont_fac, t_spont_fac, trace_spont_fac, \
    filt_trace_fac = inside_analyze_spont(info_dictionary_fac, use_b_input=True, detection_thr=40, min_dist_idx=2000)

    # fig = plt.figure(figsize=[17.6 / 2.54, 0.9 * 17.6 / 2.54])
    fig = plt.figure(figsize=[17.6 / 2.54, 1.125 * 17.6 / 2.54])
    gs1 = gridspec.GridSpec(6, 6, height_ratios=[1, 1, 1, 1, 0., 0.8])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs1[2:4, :], hspace=0.5, wspace=0.8)
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs1[5, :], hspace=0., wspace=1.0)

    my_size = 9
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rc('text', usetex=True)

    gs1.update(hspace=.3, wspace=1.5)

    ax = subplot(gs1[0, 0:6])
    t_plot_start = np.argmin(np.abs(t_spont - 236))
    t_plot_end = np.argmin(np.abs(t_spont - 246))

    plot(info_dictionary['time_array'], -info_dictionary['mean_b_input_to_p'], 'Gray')
    plot(t_spont[t_plot_start:t_plot_end], filt_trace[t_plot_start:t_plot_end], 'k', label='filtered',
         lw=1.5, zorder=2)
    ylim(-5, 80)
    my_xlim = [t_spont[t_plot_start], t_spont[t_plot_end]]
    title('Default', fontsize=my_size)
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, my_xlim, my_size)

    ob = AnchoredHScaleBar(size=0.2, label="200 ms", loc=1, frameon=False, extent=0,
                           pad=0.5, sep=4., color="Black",
                           borderpad=-1,
                           my_size=my_size)
    ax.add_artist(ob)

    textstr = r'\textbf{A}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(-0.145, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props, size = my_size + 4)

    ax = subplot(gs1[1, 0:6])
    t_plot_start = np.argmin(np.abs(t_spont - 236))
    t_plot_end = np.argmin(np.abs(t_spont - 246))
    plot(info_dictionary_fac['time_array'], -info_dictionary_fac['mean_b_input_to_p'], 'Gray')
    plot(t_spont_fac[t_plot_start:t_plot_end], filt_trace_fac[t_plot_start:t_plot_end], 'DarkMagenta',
         label='filtered with fac only',
         lw=1.5, zorder=2)
    ylim(-5, 80)
    fig.text(0.075, 0.7, 'Filtered $B$ input to $P$ [pA]', va='center', ha='center', rotation='vertical',
        fontsize=my_size)
    my_xlim = [t_spont[t_plot_start], t_spont[t_plot_end]]
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, my_xlim, my_size)
    title('With $P$-to-$A$ facilitation only', fontsize=my_size)

    ob = AnchoredHScaleBar(size=0.2, label="200 ms", loc=1, frameon=False, extent=0,
                           pad=0.5, sep=4., color="Black",
                           borderpad=-1,
                           my_size=my_size)
    ax.add_artist(ob)

    # ================ SWR properties
    unit_amp = '[pA]'
    for sub_n, my_data_default, my_data_fac, my_label in zip(
            [gs2[0, 0:2], gs2[0, 2:4], gs2[0, 4:6]],
            [IEI_end_start_FWHM, amp_peaks, durations_spont],
            [IEI_end_start_FWHM_fac, amp_peaks_fac, durations_spont_fac],
            ['IEI [sec]', 'Amplitude ' + unit_amp, 'FWHM [ms]']):
        ax = subplot(sub_n)
        plt.hist([my_data_default, my_data_fac], bins=30, color=['k', 'DarkMagenta'], normed=True, lw=0)  # normalized (AUC=1)
        adjust_yaxis(ax, my_size)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        adjust_xaxis(ax, None, my_size, show_bottom=True)
        title(my_label, fontsize=my_size)
        if my_label.find('Amplitude') != -1:
            xlim([0, 100])
            ax.set_xticks([0, 50, 100])
            yticks([0, 0.08, 0.16])
        elif my_label.find('FWHM') != -1:
            xlim([50, 220])
            ax.set_xticks([50, 150])
            yticks([0, 0.08, 0.16])
        else:
            ylabel('Prob. density [a.u.]', fontsize=my_size)
            xlim([0, 2])
            ax.set_xticks([0, 1])
            yticks([0, 0.8, 1.6])

        if my_label == 'IEI [sec]':
            textstr = r'\textbf{B}'
            props = dict(facecolor='none', edgecolor='none')
            ax.text(-0.55, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
                    verticalalignment='top', bbox=props, size=my_size + 4)

    # ================ correlation
    ax = subplot(gs2[1, 0:3])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[1:], facecolor='k',
                edgecolor='face', s=2, rasterized=True)
    plt.scatter(IEI_end_start_FWHM_fac, amp_peaks_fac[1:], facecolor='DarkMagenta',
                edgecolor='face', s=2,  rasterized=True)
    plt.axvline(x=0.17, linewidth=1., color='k', linestyle='-.')
    plt.axvline(x=0.019, linewidth=1., color='DarkMagenta', linestyle='--')

    title('Previous IEI [sec]', fontsize=my_size)
    ylabel('\n'.join(wrap('Amplitude ' + unit_amp, 40)), fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[1:])
    adjust_axes_spont(ax, c, p, my_size)
    xlim([-0.1, 2.7])
    ylim([30, 100])
    yticks([40, 100])
    xticks([0, 1, 2])

    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM, amp_peaks[1:], bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM), np.max(IEI_end_start_FWHM), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), 'k', lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant default:', 1. / popt[1] * 1e3, ' ms')

    popt, pcov = curve_fit(fit_func, IEI_end_start_FWHM_fac, amp_peaks_fac[1:], bounds=(0, [100, 100, 100]))
    aux_xaxis = np.arange(np.min(IEI_end_start_FWHM_fac), np.max(IEI_end_start_FWHM_fac), 0.05)
    plt.plot(aux_xaxis, fit_func(aux_xaxis, *popt), 'Orchid', lw=1.5,
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    print('Fitted time constant plastic:', 1. / popt[1] * 1e3, ' ms')

    textstr = r'\textbf{C}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(-0.32, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props, size=my_size + 4)

    ax = subplot(gs2[1, 3:6])
    plt.scatter(IEI_end_start_FWHM, amp_peaks[:-1], facecolor='k',
                edgecolor='k', s=2, rasterized=True)
    plt.scatter(IEI_end_start_FWHM_fac, amp_peaks_fac[:-1], facecolor='DarkMagenta',
                edgecolor='DarkMagenta', s=2, rasterized=True)
    ylabel('Amplitude ' + unit_amp, fontsize=my_size)
    title('Next IEI [sec]', fontsize=my_size)
    c, p = pearsonr(IEI_end_start_FWHM, amp_peaks[:-1])
    adjust_axes_spont(ax, c, p, my_size)
    xlim([-0.1, 2.7])
    yticks([40, 100])
    xticks([0, 1, 2])
    ylim([30, 100])

    # ================ Rate model bifurcation diagram
    home_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../'
    bs = bif.load_bifurcations(home_dir + 'bifurcation_analysis/bifurcation_diagrams/1param/','z',0,1)

    # x ticks:
    z_ticks=[0, 0.25, 0.5, 0.75, 1]
    z_ticklabels=[0,'',0.5,'',1]

    # y ticks:
    P_ticks=[0,20,40]
    B_ticks=[0,50,100]
    A_ticks=[0,5,10]

    # y range:
    pmax = 50
    bmax = 120
    amax = 15

    ax = subplot(gs3[0,0:2])
    textstr = r'\textbf{D}'
    props = dict(facecolor='none', edgecolor='none')
    ax.text(-0.54, 1.1, textstr, transform=ax.transAxes, fontsize=my_size + 4,
            verticalalignment='top', bbox=props, size = my_size + 4)

    # Plot z-P bifurcation diagram:
    bif.plot_bifurcation(ax,aux,bs,'P',[0,1],pmax,'z',z_ticks,z_ticklabels,P_ticks,P_ticks,my_size,plot_color='DarkMagenta',line_width=1.5,inward_ticks=False)

    # Create grid spanning z and P space:
    Z, P = np.meshgrid(np.arange(0, 1, .01), np.arange(-1, 120, .5))
    # Get z nullcline for values in grid:
    dZ = model.dz(Z, P, params.tau_f, params.eta_f, params.z_max)
    # Plot z nullcline:
    nc.plot_nullcline(ax,Z,P,dZ,'z nullcline','lower right',(1.05,0.0),my_size)

    ax = subplot(gs3[0, 2:4])
    # Plot z-B bifurcation diagram:
    bif.plot_bifurcation(ax,aux,bs,'B',[0,1],bmax,'z',z_ticks,z_ticklabels,B_ticks,B_ticks,my_size,plot_color='DarkMagenta',line_width=1.5,inward_ticks=False)

    ax = subplot(gs3[0, 4:6])
    # Plot z-A bifurcation diagram:
    bif.plot_bifurcation(ax,aux,bs,'A',[0,1],amax,'z',z_ticks,z_ticklabels,A_ticks,A_ticks,my_size,plot_color='DarkMagenta',line_width=1.5,inward_ticks=False)

    # Separating line:
    fig.text(0.5, 0.229, 'Rate model', va='center', ha='center', fontsize=my_size, bbox=dict(facecolor='white', edgecolor='white'))
    line = plt.Line2D([0.06,0.9],[0.23,0.23], linewidth=1, linestyle='--', transform=fig.transFigure, color='DarkGray')
    fig.add_artist(line)

    savefig(os.path.join(path_folder, filename + '_compare_with_PtoAfac_only.png'),
            dpi=200, format='png', bbox_inches='tight')


# FIG 6-1
def adds_on_fig_6(filename, fraction_stim=.6):
    """Additional plots to explain LFP definition, producing Suppl. Figure 6-1 """

    filename_saved = os.path.join(path_folder, + filename + '_sim_all_current_fig6_fraction_' + str(fraction_stim) + '.npz')

    data = np.load(filename_saved, encoding='latin1', allow_pickle=True)  # all elements are unitless (/Hz
    info_dic = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    t_array = info_dic['t_array']
    p_array = info_dic['p_array']
    b_array = info_dic['b_array']
    a_array = info_dic['a_array']
    my_width_smooth = info_dic['my_width_smooth']
    mean_depr_array = info_dic['mean_depr_array']
    warm_up_time = info_dic['warm_up_time']
    simtime_current = info_dic['simtime_current']
    time_with_curr = info_dic['time_with_curr']
    sigma_P = info_dic['sigma_P']
    sigma_B = info_dic['sigma_B']
    sigma_A = info_dic['sigma_A']
    spikes_dic = info_dic['spikes_dic'].item()
    mean_P_input_p = info_dic['mean_P_input_p']
    mean_B_input_p = info_dic['mean_B_input_p']
    mean_A_input_p = info_dic['mean_A_input_p']
    compress_step_indices = int(info_dic['compress_step_indices'])

    fig = plt.figure(figsize=[12, 9])
    my_size = 11
    x_lim_start = warm_up_time - 700  # ms
    x_lim_end = warm_up_time + 250  # ms
    xlim1 = [x_lim_start, x_lim_end]

    curr_to_pop = 'B'
    idx_c = 1
    # ============================================== #
    # =================== P cells ================== #
    # ============================================== #
    ax = subplot(511)
    ip, tp = spikes_dic['P_' + str(idx_c)]
    plt.scatter(tp / ms, ip, marker='.', color='#ef3b53', s=0.85, rasterized=True)
    # ylim([0, 50])
    xlim(xlim1)
    plt.axis('off')
    add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

    ax = subplot(512)
    plt.plot(t_array[idx_c, :], p_array[idx_c, :], '#ef3b53', lw=2., label='P')
    ylim(-5, np.max(p_array) + 2)
    yticks([0, 40, 80])
    adjust_yaxis(ax, my_size)
    adjust_xaxis(ax, None, my_size, show_bottom=False)
    ylabel('\n'.join(wrap('P pop. rate [spikes/s]', 12)), fontsize=my_size)

    xlim(xlim1)
    ob = AnchoredHScaleBar(size=50, label="50 ms", loc=3, frameon=False, extent=0,
                           pad=1., sep=4, color="Black",
                           borderpad=0.1,
                           my_size=my_size)   # pad and borderpad can be used to modify the position of the bar a bit
    ax.add_artist(ob)
    add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

    ax = subplot(513)
    # select part of array that is plotted
    idx_xlim_start = (np.abs(t_array - x_lim_start)).argmin()
    idx_xlim_end = (np.abs(t_array - x_lim_end)).argmin()

    plot(t_array[idx_c, idx_xlim_start:idx_xlim_end],
         mean_P_input_p[idx_c, idx_xlim_start:idx_xlim_end],
         '#ef3b53', lw=1.5, label='P input to P cells')

    plot(t_array[idx_c, idx_xlim_start:idx_xlim_end],
         mean_A_input_p[idx_c, idx_xlim_start:idx_xlim_end],
         '#0a9045', lw=1.5, label='A input to P cells')

    plot(t_array[idx_c, idx_xlim_start:idx_xlim_end],
         mean_B_input_p[idx_c, idx_xlim_start:idx_xlim_end],
         '#3c3fef', lw=1.5, label='B input to P cells')
    legend(loc='lower center', prop={'size': my_size}, framealpha=1.)

    xlim(xlim1)
    ylabel('\n'.join(wrap('Input current to P [pA]', 14)), fontsize=my_size)
    adjust_xaxis(ax, None, my_size, show_bottom=False)
    adjust_yaxis(ax, my_size)
    add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

    ax = subplot(514)
    # use Butterworth low pass
    fs = 1e4 / compress_step_indices
    lowcut = -1.
    highcut = 5.
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2, btype='low')
    filt_trace = filtfilt(b, a, -mean_B_input_p[idx_c, :])
    plot(t_array[idx_c, idx_xlim_start:idx_xlim_end], filt_trace[idx_xlim_start:idx_xlim_end], '#3c3fef', lw=2.,
         label='low pass of B input to P')
    xlim(xlim1)
    ylabel('\n'.join(wrap('- Low-pass filtered B input to P [pA]', 16)), fontsize=my_size)
    adjust_xaxis(ax, None, my_size, show_bottom=False)
    adjust_yaxis(ax, my_size)
    add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

    ax = subplot(515)
    # band pass filtered version of inh input (high passed)
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 1e4 / compress_step_indices
    lowcut = 70
    highcut = 180
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2)
    y_p = filtfilt(b, a, -mean_B_input_p[idx_c, :])
    plot(t_array[idx_c, idx_xlim_start:idx_xlim_end], y_p[idx_xlim_start:idx_xlim_end], '#3c3fef', lw=1.5,
         label='band pass of B input to P')
    xlim(xlim1)
    ylabel('\n'.join(wrap('- Band-pass filtered B input to P [pA]', 16)), fontsize=my_size)
    adjust_xaxis(ax, None, my_size, show_bottom=False)
    adjust_yaxis(ax, my_size)
    add_sign_of_stimulation(ax, warm_up_time, time_with_curr)

    savefig(os.path.join(path_folder, filename + '_fig6_addson_SPW_like_mod_' + curr_to_pop + '.png'), dpi=300,
            format='png', bbox_inches='tight')



# FIG 2-1
def adds_on_fig_2(filename, save_intermediate=True):
    """Show If curves and their approximation"""
    fit_boundary_IF_low = [0., -100]  # for a and b respectively
    fit_boundary_IF_up = [2., 0.]  # for a and b respectively

    fig = plt.figure(figsize=[12, 8])
    my_size = 14
    # matplotlib.rc('font', family='serif', serif='cm10')
    plt.rc('text', usetex=True)
    rc('font', size=16)
    outer = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])
    outer.update(wspace=0.5, hspace=0.5)
    gs3 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=outer[:, :], hspace=.9, wspace=0.2)
    # I-f curves
    n = 50  # number of cell for mean I-f curve
    # resting state
    filename_resting = os.path.join(path_folder, filename + '_resting_IF_n' + str(n) + '.npz')
    data = np.load(filename_resting, encoding='latin1', allow_pickle=True)
    dic_resting = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    extra_curr_rest = dic_resting['extracurrent_array']
    shifted_input_p_rest = [extra_curr_rest + dic_resting['mean_input_p'][:, i] for i in range(n)]
    shifted_input_b_rest = [extra_curr_rest + dic_resting['mean_input_b'][:, i] for i in range(n)]
    shifted_input_a_rest = [extra_curr_rest + dic_resting['mean_input_a'][:, i] for i in range(n)]

    # excited case
    filename_excited = os.path.join(path_folder, filename + '_excited_IF_n' + str(n) + '.npz')
    data = np.load(filename_excited, encoding='latin1', allow_pickle=True)
    dic_excited = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))

    extra_curr_exc = dic_excited['extracurrent_array']  # num data points
    shifted_input_p_exc = [extra_curr_exc + dic_excited['mean_input_p'][:, i] for i in range(n)]
    shifted_input_b_exc = [extra_curr_exc + dic_excited['mean_input_b'][:, i] for i in range(n)]
    shifted_input_a_exc = [extra_curr_exc + dic_excited['mean_input_a'][:, i] for i in range(n)]

    # NOTE; if interp axis too large, need to take care of the bad definition of weighted curve of P,A for large I
    # (possibly make a variable interp_axis)
    interp_axis = np.arange(0, 300, 1.)

    # !!! to plot mean input, take input across neurons when extracurrent is zero
    idx_no_extracurr_rest = np.where(extra_curr_rest == 0.)[0]
    idx_no_extracurr_exc = np.where(extra_curr_exc == 0.)[0]

    xlim_if = [0, 350]
    ylim_if = [0, 120]

    ax1 = subplot(gs3[0:2, 0])  # P rest
    font0 = FontProperties()
    my_font = font0.copy()
    textstr = r'\textbf{A}'
    props = dict(facecolor='none', edgecolor='none')
    # place a text box in upper left in axes coords
    ax1.text(-.3, 1.1, textstr, transform=ax1.transAxes, fontsize=my_size, fontproperties=my_font,
             verticalalignment='top', bbox=props)
    ax2 = subplot(gs3[0:2, 1])  # B rest
    ax3 = subplot(gs3[0:2, 2])  # A rest
    ax4 = subplot(gs3[2:, 0])  # P exc
    textstr = r'\textbf{B}'
    props = dict(facecolor='none', edgecolor='none')
    # place a text box in upper left in axes coords
    ax4.text(-.3, 1.1, textstr, transform=ax4.transAxes, fontsize=my_size, fontproperties=my_font,
             verticalalignment='top', bbox=props)
    ax5 = subplot(gs3[2:, 1])  # B exc
    ax6 = subplot(gs3[2:, 2])  # A exc

    # ======================================== #
    # ============== P rest, ax1
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_p_rest[i], dic_resting['fr_p'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax1.plot(interp_axis, f_interp[:, i], c='Gray')

    # mean_fP_rest = mean(f_interp, axis=1)
    mean_fP_rest = np.nanmean(f_interp, axis=1)
    ax1.plot(interp_axis, mean_fP_rest, c='#ef3b53', lw=2, label='mean P rest')
    shaded_gradient(ax1, mean(dic_resting['mean_input_p'][idx_no_extracurr_rest, :]),
                    mean(dic_resting['std_input_p'][idx_no_extracurr_rest, :]), base_color='r')
    # ======================================== #
    # =============== P exc, ax4
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_p_exc[i], dic_excited['fr_p'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax4.plot(interp_axis, f_interp[:, i], c='Gray')

    # mean_fP_exc = mean(f_interp, axis=1)
    mean_fP_exc = np.nanmean(f_interp, axis=1)
    ax4.plot(interp_axis, mean_fP_exc, c='#ef3b53', lw=2, label='mean P exc')
    shaded_gradient(ax4, mean(dic_excited['mean_input_p'][idx_no_extracurr_exc, :]),
                    mean(dic_excited['std_input_p'][idx_no_extracurr_exc, :]), base_color='r')

    # ================== add intermediate curve P
    x1 = mean(dic_resting['mean_input_p'][idx_no_extracurr_rest, :]) - mean(
        dic_resting['std_input_p'][idx_no_extracurr_rest, :])
    x2 = mean(dic_resting['mean_input_p'][idx_no_extracurr_rest, :]) + mean(
        dic_resting['std_input_p'][idx_no_extracurr_rest, :])
    y1 = mean(dic_excited['mean_input_p'][idx_no_extracurr_exc, :]) - mean(
        dic_excited['std_input_p'][idx_no_extracurr_exc, :])
    y2 = mean(dic_excited['mean_input_p'][idx_no_extracurr_exc, :]) + mean(
        dic_excited['std_input_p'][idx_no_extracurr_exc, :])
    P_interm = define_Ifcurve_weighted_sum(interp_axis, y1, mean_fP_rest, mean_fP_exc, celltype='P')

    ax1.plot(interp_axis, P_interm, lw=2, color='Orange')
    ax4.plot(interp_axis, P_interm, lw=2, color='Orange')

    # add curve fit to softplus
    # poor performance if you set no bounds...
    popt_P, pcov = curve_fit(softplus_func, interp_axis, P_interm, bounds=(fit_boundary_IF_low, fit_boundary_IF_up))
    If_P_softplus = softplus_func(interp_axis, *popt_P)
    I_P_softplus = interp_axis
    If_P_param = popt_P

    ax1.plot(interp_axis, softplus_func(interp_axis, *popt_P), 'k', lw=2.,)
    ax4.plot(interp_axis, softplus_func(interp_axis, *popt_P), 'k', lw=2.,)

    k, s = popt_P
    # ax1.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')
    # ax4.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')

    # ======================================== #
    # ============= B rest, ax 2
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_b_rest[i], dic_resting['fr_b'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax2.plot(interp_axis, f_interp[:, i], c='Gray')

    # mean_fB_rest = mean(f_interp, axis=1)
    mean_fB_rest = np.nanmean(f_interp, axis=1)
    ax2.plot(interp_axis, mean_fB_rest, c='#3c3fef', lw=2, label='mean B rest')
    shaded_gradient(ax2, mean(mean(dic_resting['mean_input_b'][idx_no_extracurr_rest, :])),
                    mean(dic_resting['std_input_b'][idx_no_extracurr_rest, :]), base_color='b')

    # =================== B exc, ax 5
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_b_exc[i], dic_excited['fr_b'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax5.plot(interp_axis, f_interp[:, i], c='Gray')

    mean_fB_exc = np.nanmean(f_interp, axis=1)
    # mean_fB_exc = np.mean(f_interp, axis=1)
    ax5.plot(interp_axis, mean_fB_exc, c='#3c3fef', lw=2, label='mean B exc')
    shaded_gradient(ax5, mean(dic_excited['mean_input_b'][idx_no_extracurr_exc, :]),
                    mean(dic_excited['std_input_b'][idx_no_extracurr_exc, :]), base_color='b')

    # ================== add intermediate curve B
    x1 = mean(dic_resting['mean_input_b'][idx_no_extracurr_rest, :]) - mean(
        dic_resting['std_input_b'][idx_no_extracurr_rest, :])
    x2 = mean(dic_resting['mean_input_b'][idx_no_extracurr_rest, :]) + mean(
        dic_resting['std_input_b'][idx_no_extracurr_rest, :])
    y1 = mean(dic_excited['mean_input_b'][idx_no_extracurr_exc, :]) - mean(
        dic_excited['std_input_b'][idx_no_extracurr_exc, :])
    y2 = mean(dic_excited['mean_input_b'][idx_no_extracurr_exc, :]) + mean(
        dic_excited['std_input_b'][idx_no_extracurr_exc, :])
    B_interm = define_Ifcurve_weighted_sum(interp_axis, y1, mean_fB_rest, mean_fB_exc, celltype='B')

    ax2.plot(interp_axis, B_interm, lw=2, color='Orange')
    ax5.plot(interp_axis, B_interm, lw=2, color='Orange')

    # add curve fit to softplus
    # poor performance if you set no bounds...
    popt_B, pcov = curve_fit(softplus_func, interp_axis, B_interm, bounds=(fit_boundary_IF_low, fit_boundary_IF_up))
    If_B_softplus = softplus_func(interp_axis, *popt_B)
    I_B_softplus = interp_axis
    If_B_param = popt_B

    ax2.plot(interp_axis, softplus_func(interp_axis, *popt_B), 'k', lw=2.)
    ax5.plot(interp_axis, softplus_func(interp_axis, *popt_B), 'k', lw=2.)

    k, s = popt_B
    # ax2.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')
    # ax5.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')

    # ======================================== #
    # ============= A rest
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_a_rest[i], dic_resting['fr_a'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax3.plot(interp_axis, f_interp[:, i], c='Gray')

    # mean_fA_rest = mean(f_interp, axis=1)
    mean_fA_rest = np.nanmean(f_interp, axis=1)
    ax3.plot(interp_axis, mean_fA_rest, c='#0a9045', lw=2, label='mean A rest')
    shaded_gradient(ax3, mean(mean(dic_resting['mean_input_a'][idx_no_extracurr_rest, :])),
                    mean(dic_resting['std_input_a'][idx_no_extracurr_rest, :]), base_color='g')

    # =============== A exc, ax 6
    f_interp = np.empty((len(interp_axis), n))
    for i in range(n):
        # interpolate curves
        f = interp1d(shifted_input_a_exc[i], dic_excited['fr_a'][:, i], kind='linear', bounds_error=False)
        f_interp[:, i] = f(interp_axis)
        ax6.plot(interp_axis, f_interp[:, i], c='Gray')

    # mean_fA_exc = mean(f_interp, axis=1)
    mean_fA_exc = np.nanmean(f_interp, axis=1)
    ax6.plot(interp_axis, mean_fA_exc, c='#0a9045', lw=2, label='mean A exc')
    shaded_gradient(ax6, mean(dic_excited['mean_input_a'][idx_no_extracurr_exc, :]),
                    mean(dic_excited['std_input_a'][idx_no_extracurr_exc, :]), base_color='g')

    # ================== add intermediate curve
    x1 = mean(dic_resting['mean_input_a'][idx_no_extracurr_rest, :]) - mean(
        dic_resting['std_input_a'][idx_no_extracurr_rest, :])
    x2 = mean(dic_resting['mean_input_a'][idx_no_extracurr_rest, :]) + mean(
        dic_resting['std_input_a'][idx_no_extracurr_rest, :])
    y1 = mean(dic_excited['mean_input_a'][idx_no_extracurr_exc, :]) - mean(
        dic_excited['std_input_a'][idx_no_extracurr_exc, :])
    y2 = mean(dic_excited['mean_input_a'][idx_no_extracurr_exc, :]) + mean(
        dic_excited['std_input_a'][idx_no_extracurr_exc, :])
    # A_interm = define_intermediate_activation_function(interp_axis, x1, x2, y1, y2, mean_fA_rest, mean_fA_exc,
    #                                                    celltype='A')
    A_interm = define_Ifcurve_weighted_sum(interp_axis, x1, mean_fA_rest, mean_fA_exc, celltype='A')

    ax3.plot(interp_axis, A_interm, lw=2, color='Orange')
    ax6.plot(interp_axis, A_interm, lw=2, color='Orange')

    # add curve fit to softplus
    # poor performance if you set no bounds...
    popt_A, pcov = curve_fit(softplus_func, interp_axis, A_interm, bounds=(fit_boundary_IF_low, fit_boundary_IF_up))
    If_A_softplus = softplus_func(interp_axis, *popt_A)
    I_A_softplus = interp_axis
    If_A_param = popt_A
    ax3.plot(interp_axis, softplus_func(interp_axis, *popt_A), 'k', lw=2.)
    ax6.plot(interp_axis, softplus_func(interp_axis, *popt_A), 'k', lw=2.)
    k, s = popt_A
    # ax3.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')
    # ax6.plot(interp_axis, (k*(interp_axis + s))*(interp_axis >= -s) + 0.*(interp_axis < -s), lw = 2., color='Yellow')

    for my_ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        adjust_xaxis(my_ax, xlim_if, my_size, show_bottom=True)
        adjust_yaxis(my_ax, my_size)
        my_ax.set_yticks([0, 50, 100])
        my_ax.set_xticks([0, 150, 300])
        my_ax.set_ylim(ylim_if)
        my_ax.set_xlim(xlim_if)

    ax1.set_ylabel('Firing rate [spikes/s]', fontsize=my_size)
    ax1.set_title('f-I curves P population', fontsize=my_size)
    ax2.set_title('f-I curves B population', fontsize=my_size)
    ax3.set_title('f-I curves A population', fontsize=my_size)
    ax4.set_xlabel('I [pA]', fontsize=my_size)
    ax4.set_ylabel('Firing rate [spikes/s]', fontsize=my_size)
    ax5.set_xlabel('I [pA]', fontsize=my_size)
    ax6.set_xlabel('I [pA]', fontsize=my_size)

    savefig(os.path.join(path_folder, filename + '_fig2addson_SPW_like' + '.png'))

    if save_intermediate:
        # save intermediate curves to test behavior of mean-field
        filename_full = os.path.join(path_folder, filename + '_intermediate_fI.npz')

        data_to_save_fI = {'interp_axis': interp_axis,
                           'P_interm': P_interm,
                           'B_interm': B_interm,
                           'A_interm': A_interm,
                           'mean_fP_rest': mean_fP_rest,
                           'mean_fB_rest': mean_fB_rest,
                           'mean_fA_rest': mean_fA_rest,
                           'mean_fP_exc': mean_fP_exc,
                           'mean_fB_exc': mean_fB_exc,
                           'mean_fA_exc': mean_fA_exc,
                           'I_P_softplus': I_P_softplus,
                           'I_B_softplus': I_B_softplus,
                           'I_A_softplus': I_A_softplus,
                           'If_P_softplus': If_P_softplus,
                           'If_B_softplus': If_B_softplus,
                           'If_A_softplus': If_A_softplus,
                           'If_P_param': If_P_param,
                           'If_B_param': If_B_param,
                           'If_A_param': If_A_param
                           }

        np.savez_compressed(filename_full, **data_to_save_fI)
        print('data saved in ', filename_full)
