__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains supporting functions needed to run the spiking network"""

import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt

import sys
import os
sys.path.append(os.path.dirname( __file__ ) + '/../')

from helper_functions.detect_peaks import detect_peaks  # works when called from run rate, not from spiking


class Connectivity():
    """ Param class contains all the variables relative to connectivities """

    def __init__(self, g_update, prob_of_connect, size_pre, size_post, name, delay=1 * ms):
        """ Create a new parameter set """

        self.g_update = g_update
        self.prob_of_connect = prob_of_connect
        self.size_pre = size_pre
        self.size_post = size_post
        self.delay = delay
        self.name = name

    def create_connectivity_matrix(self):
        conn_matrix = np.random.binomial(1, self.prob_of_connect, size=(self.size_pre, self.size_post))
        rows, cols = nonzero(conn_matrix)

        self.rows_connect_matrix = rows
        self.cols_connect_matrix = cols


class SpikingParamFromDict():
    """ Param class contains all the variables"""

    def __init__(self, dic, dic_idx=-1):
        """ Create a new parameter set """

        for k, v in dic.items():
            setattr(self, k, v)


def average_firing_rate(sp_monitor, time_start, time_end):
    """Computes the average of the neuron firing rate over time. time_start tells us from which point in the simulation
    we should be looking
    """
    # num_spikes = sp_monitor.count[0]
    # len(spiketrain) is the number of neurons in the population
    spiketrain = sp_monitor.spike_trains()
    # noinspection PyTypeChecker
    fr_neurons = [float(sum((time_start < spiketrain[i]) * (spiketrain[i] < time_end))) for i in range(len(spiketrain))]
    average_fr = mean(fr_neurons) / (time_end - time_start)
    return average_fr


def unit_firing_rate(sp_monitor, time_start, time_end):
    """Computes the neuron firing rate over time. time_start tells us from which point in the simulation
    we should be looking
    """
    spiketrain = sp_monitor.spike_trains()
    fr_neurons = [float(sum((time_start < spiketrain[i]) * (spiketrain[i] < time_end))) for i in range(len(spiketrain))]
    average_fr = fr_neurons / (time_end - time_start)  # array of values
    return average_fr


def plot_average_pop_rate(FRP, FRB, FRA, my_width=3 * ms):
    """Plot smoothed population rate"""
    figure(figsize=[13, 10])
    if FRP is not None:
        subplot(311)
        plot(FRP.t / ms, FRP.smooth_rate('gaussian', width=my_width) / Hz, '#ef3b53', label='P', lw=1.5)
        legend(loc='best')
        ylabel('Population rate [spikes/s]')
        ylim(0, 100)

    if FRB is not None:
        subplot(312)
        plot(FRB.t / ms, FRB.smooth_rate('gaussian', width=my_width) / Hz, '#3c3fef', label='B', lw=1.5)
        legend(loc='best')
        ylabel('Population rate [spikes/s]')
        ylim(0, 150)

    if FRA is not None:
        subplot(313)
        plot(FRA.t / ms, FRA.smooth_rate('gaussian', width=my_width) / Hz, '#0a9045', label='A', lw=1.5)
        ylim(0, 60)
        legend(loc='best')
        xlabel('Time [ms]')
        ylabel('Population rate [spikes/s]')


def raster_plots(smp, smb, sma, xlim1, xlim2, xlim3, num_time_windows=3):
    """Plot raster plots of simulations. Could be one population or more, in one time window or more.
    smp, smb, sma are brian objects (SpikeMonitor)"""

    num_col = 3
    if num_time_windows == 3:

        figure(figsize=[20, 16])
        num_row = 3
        array_subplots_P = [1, 4, 7]
        array_subplots_B = [2, 5, 8]
        array_subplots_A = [3, 6, 9]

        array_xlim = [xlim1, xlim2, xlim3]
        array_title_str_P = ["P pop, before current", "P pop, during current", "P pop, after current"]
        array_title_str_B = ["B pop, before current", "B pop, during current", "B pop, after current"]
        array_title_str_A = ["A pop, before current", "A pop, during current", "A pop, after current"]

    elif num_time_windows == 2:

        figure(figsize=[16, 10])
        num_row = 2
        array_subplots_P = [1, 4]
        array_subplots_B = [2, 5]
        array_subplots_A = [3, 6]

        array_xlim = [xlim1, xlim2]
        array_title_str_P = ["P pop, before plasticity", "P pop, after plasticity"]
        array_title_str_B = ["B pop, before plasticity", "B pop, after plasticity"]
        array_title_str_A = ["A pop, before plasticity", "A pop, after plasticity"]

    elif num_time_windows == 1:
        num_row = 1
        figure(figsize=[15, 5])
        array_subplots_P = [1]
        array_subplots_B = [2]
        array_subplots_A = [3]

        array_xlim = [xlim1]
        array_title_str_P = ["P pop"]
        array_title_str_B = ["B pop"]
        array_title_str_A = ["A pop"]

    else:
        raise ValueError('Wrong number of time windows')

    # Plot P raster plots
    if smp is not None:
        i, t = smp.it
        for curr_subplot, curr_xlim, title_str in zip(array_subplots_P, array_xlim, array_title_str_P):
            subplot(num_row, num_col, curr_subplot)
            plot(t / ms, i, '.', color='#ef3b53', ms=0.45)
            title(title_str)
            xlabel("time [ms]")
            # yticks([])
            ylabel('Neuron ID')
            xlim(curr_xlim)

    if smb is not None:
        i, t = smb.it
        for curr_subplot, curr_xlim, title_str in zip(array_subplots_B, array_xlim, array_title_str_B):
            subplot(num_row, num_col, curr_subplot)
            plot(t / ms, i, '.', color='#3c3fef', ms=0.45)
            title(title_str)
            xlabel("time [ms]")
            xlim(curr_xlim)

    if sma is not None:
        i, t = sma.it
        for curr_subplot, curr_xlim, title_str in zip(array_subplots_A, array_xlim, array_title_str_A):
            subplot(num_row, num_col, curr_subplot)
            plot(t / ms, i, '.', color='#0a9045', ms=0.45)
            title(title_str)
            xlabel("time [ms]")
            xlim(curr_xlim)


def extract_conn_submatrix(mat_rows, mat_cols, size_pre, size_post, indices_to_select_cols, extract_g_update=False):
    """given arrays of nonzero synaptic pairs, extract the submatrix of connections from network
    to extra neurons"""
    auxmat = np.zeros((size_pre, size_post))
    auxmat[mat_rows, mat_cols] = True
    submat = auxmat[:, indices_to_select_cols]
    rows, cols = nonzero(submat)

    if extract_g_update:
        sliced_dummy_mat = np.arange(size_pre * size_post).reshape(size_pre, size_post)[:, indices_to_select_cols]
        raveled = np.ravel_multi_index((mat_rows, mat_cols), (size_pre, size_post))
        idx_to_take_g = np.in1d(raveled, sliced_dummy_mat.flatten())
        return rows, cols, idx_to_take_g
    else:
        return rows, cols, None


def enough_presyn_input(conn_prob, N_pre, thr_presyn_input=5):
    """calculate std on number of presynaptic inputs to a given neuron.
    Only if large enough continue with Brian network creation """
    return np.sqrt(conn_prob * N_pre) > thr_presyn_input


def create_connections(info_dictionary, pop_p, pop_b, pop_a, tau_depr, tau_depr_PB, use_dic_connect=False, depressing_PB=False):
    """Initialize connections in the network. Defult is fully connected network with plastic AB, and possibly PB.
    To be used by all functions dealing with spiking network after step4"""

    if use_dic_connect:
        conn_dictionary = info_dictionary['dic_connect'].copy()

    if use_dic_connect:
        dic_list = ['W_pp', 'W_bp', 'W_bb', 'W_pa', 'W_ap', 'W_aa', 'W_ba', 'W_pb', 'W_ab']
    else:
        dic_list = ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']

    for dic_name in dic_list:

        if use_dic_connect:
            c_aux = Connectivity(conn_dictionary[dic_name].g_update / nS, conn_dictionary[dic_name].prob_of_connect,
                                 conn_dictionary[dic_name].size_pre, conn_dictionary[dic_name].size_post,
                                 conn_dictionary[dic_name].name, conn_dictionary[dic_name].delay)
            c_aux.rows_connect_matrix = conn_dictionary[dic_name].rows_connect_matrix
            c_aux.cols_connect_matrix = conn_dictionary[dic_name].cols_connect_matrix

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

    # try out what happens when B->P is depressing
    if depressing_PB:
        eqs_std_PB = '''
               g_pb : 1
               dy / dt = (1. - y) / tau_depr_PB : 1 (clock-driven)
               '''
        con_PB = Synapses(pop_b, pop_p, model=eqs_std_PB,
                          on_pre='''g_gabaB += y*g_pb*nS
                                y = clip(y - y * eta_pb, 0, 1)
                                 ''', delay=c_PB.delay, method='exact')
        con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
        con_PB.g_pb = c_PB.g_update
        con_PB.y = 1.
    else:
        con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
        con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
        con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

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

    return con_PP, con_PB, con_PA, con_BP, con_BB, con_BA, con_AP, con_AB, con_AA


def standard_neuronal_parameters_for_brian():
    """Need to have them as single variables or Brian won't understand (e.g. if they are in a class)"""
    NP = 8200  # num pyramidal cells (P)
    NB = 135  # num PV+BC cells (B)
    NA = 50  # num SOM+ cells (A)

    tau_ampa = 2.0 * ms  # Glutamatergic synaptic time constant - decay
    tau_gabaA = 4.0 * ms  # GABAergic synaptic time constant - decay
    tau_gabaB = 1.5 * ms  # GABAergic synaptic time constant - decay B

    # Neuron model
    g_leak = 10.0 * nsiemens  # Leak conductance
    v_rest = -60 * mV  # Resting potential
    rev_ampa = 0 * mV  # Exc reversal potential
    rev_gabaB = -70 * mV  # Inhibitory reversal potential
    rev_gabaA = -70 * mV  # Inhibitory reversal potential
    mem_cap = 200.0 * pfarad  # Membrane capacitance
    vthr_all = -50. * mV
    vthr_P = -50. * mV
    vthr_B = -50. * mV
    vthr_A = -50. * mV
    bg_curr = 200 * pA  # Background current
    extra_curr = 0. * pA
    trefr_P = 1 * ms
    trefr_B = 1 * ms
    trefr_A = 1 * ms
    tau_syn_del = 1. * ms
    tau_depr_AB = 250. * ms
    eta_ab = 0.35  # 0.18 as we are interested mostly in the bistable case, we make it large so that it is more
    # likely that the state switches back to outside

    eqs_neurons = '''
           dv/dt=(-g_leak*(v-v_rest)-(g_ampa*(v-rev_ampa)+g_gabaA*(v-rev_gabaA))+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
           dg_ampa/dt = -g_ampa/tau_ampa : siemens
           dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
           dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
           input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
           vthr: volt
           extracurrent: amp
           trefr : second
           '''

    return NP, NB, NA, tau_ampa, tau_gabaB, tau_gabaA, g_leak, v_rest, rev_ampa, rev_gabaB, rev_gabaA, mem_cap, \
           vthr_all, vthr_P, vthr_B, vthr_A, bg_curr, extra_curr, trefr_P, trefr_B, trefr_A, tau_syn_del, tau_depr_AB, \
           eta_ab, eqs_neurons


def fit_func(x, a, b, c):
    """Use exponential function to fit data. More realistic to assume there is some biologically motivated upper bound
    to values of amplitude of events"""
    return a * (1. - np.exp(-b * x)) + c


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """
    To create scale bars (horizontal)
    from: https://stackoverflow.com/questions/43258638/is-there-a-convenient-way-to-add-a-scale-indicator-to-a-plot-in-matplotlib
    size: length of bar in data units
    extent : height of bar ends in axes units """

    def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, horiz=True, my_size=12, **kwargs):
        if not ax:
            ax = plt.gca()
        if horiz:
            trans = ax.get_xaxis_transform()
        else:
            trans = ax.get_yaxis_transform()

        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        if horiz:
            line = Line2D([0, size], [0, 0], **kwargs)
            vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **kwargs)
            vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **kwargs)
        else:
            line = Line2D([0, 0], [0, size], **kwargs)
            vline1 = Line2D([-extent / 2., extent / 2.], [0, 0], **kwargs)
            vline2 = Line2D([-extent / 2., extent / 2.], [size, size], **kwargs)

        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops={'size': my_size})

        if horiz:
            self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                     align="center", pad=ppad, sep=sep)
        else:
            self.vpac = matplotlib.offsetbox.HPacker(children=[size_bar, txt],
                                                     align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                        borderpad=borderpad, child=self.vpac, prop=prop,
                                                        frameon=frameon)



def add_sign_of_stimulation(ax, warm_up_time, time_with_curr):
    """Create shaded area when stimulation is injected, or a bar"""
    # Shaded region where current is injected
    ix = np.linspace(warm_up_time, (warm_up_time + time_with_curr))
    iy = np.linspace(10000, 10000)
    verts = [(warm_up_time, -1000)] + list(zip(ix, iy)) + [
        ((warm_up_time + time_with_curr), -1000)]
    poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.3)
    ax.add_patch(poly)


def adjust_yaxis(ax, my_size):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelsize=my_size)


def adjust_xaxis(ax, x_lim, my_size, show_bottom=False):
    if show_bottom:
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,
            top=False,  # ticks along the top edge are off
            labelbottom=True,
            labelsize=my_size)
    else:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
    xlim(x_lim)


def create_butter_bandpass(lowcut, highcut, fs, order=2, btype='band'):
    """create a butterworth digital filter with given order"""
    # normalize freq with Nyquist freq (to be in [0,1])
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if btype == 'band':
        b, a = butter(order, [low, high], btype=btype, analog=False)
    elif btype == 'low':
        b, a = butter(order, high, btype=btype, analog=False)
    else:
        b = 0
        a = 0
    return b, a


def adjust_axes_spont(ax, c, p, my_size):
    # textstr = 'Correlation: %.2f, p: %.4f' % (c, p)
    textstr = 'Correlation: %.2f' % (c)
    props = dict(lw=0, facecolor='white')
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelsize=my_size)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        labelsize=my_size)


def define_intermediate_activation_function(x, x1, x2, y1, y2, f1, f2, celltype='P'):
    """x1, x2 = mean -+1std in outside state, y1, y2 = mean -+1std in inside state.
     NOTE: all the nan_to_num are needed because mean_Fx_rest/exc contain nans and nans+0 = nan -> this messes up the
     function definition as sum of portions depending on x values"""
    if celltype in ['P', 'B']:
        if x2 < y1:
            return np.nan_to_num(f1 * (x <= x2)) + np.nan_to_num((f1 + f2) / 2. * (x2 < x) * (x <= y1)) \
                   + np.nan_to_num(f2 * (x > y1))
        else:
            return np.nan_to_num(f1 * (x <= y1)) + np.nan_to_num((f1 + f2) / 2. * (y1 < x) * (x <= x2)) \
                   + np.nan_to_num(f2 * (x > x2))

    if celltype in ['A']:
        if x2 < y2:
            return np.nan_to_num(f2 * (x <= x1)) + np.nan_to_num((f1 + f2) / 2. * (x1 < x) * (x <= y2)) \
                   + np.nan_to_num(f1 * (x > y2))
    else:
        return 0.


def define_Ifcurve_weighted_sum(x, thr, f_out, f_in, celltype='P'):
    """Thr = mean -1std in active state (inside for P, B, outside for A)
    NOTE: all the nan_to_num are needed because mean_Fx_rest/exc contain nans and nans+0 = nan -> this messes up the
    function definition as sum of portions depending on x values"""
    if celltype in ['P', 'B']:
        return np.nan_to_num(f_out * (x <= thr)) + np.nan_to_num(f_in * (x > thr))

    elif celltype == 'A':
        return np.nan_to_num(f_in * (x <= thr)) + np.nan_to_num(f_out * (x > thr))


def shaded_gradient(ax, mymean, mystd, base_color='r'):
    """Custom made fucntion to plot a gradient in the shaded area around the I-f curves, where most of the input arrives.
    Only works with base color red, blue and green"""
    N = 200

    my_data = np.linspace(mymean, mymean + mystd, N)
    my_to_x = my_data + 0.05

    custom_span = np.linspace(0.5, 1., N)
    for i, el in enumerate(custom_span):
        ch = float(el)

        if base_color == 'b':
            ax.axvspan(my_data[i], my_to_x[i], color=(ch, ch, 1.))
        elif base_color == 'g':
            ax.axvspan(my_data[i], my_to_x[i], color=(ch, 1., ch))
        elif base_color == 'r':
            ax.axvspan(my_data[i], my_to_x[i], color=(1., ch, ch))

    from_xright = np.linspace(mymean, mymean - mystd, N)
    to_xright = from_xright + 0.05
    for i, el in enumerate(custom_span):
        ch = float(el)

        if base_color == 'b':
            ax.axvspan(from_xright[i], to_xright[i], color=(ch, ch, 1.))
        elif base_color == 'g':
            ax.axvspan(from_xright[i], to_xright[i], color=(ch, 1., ch))
        elif base_color == 'r':
            ax.axvspan(from_xright[i], to_xright[i], color=(1., ch, ch))



def softplus_func(x, k, s):
    """Possible apporixmation of the weigther If curves"""
    return np.log(1. + np.exp(k * (x + s)))


def calculation_CV(sm_p, sm_b, sm_a):
    """Compute CV given the SpikeMonitor brian object"""

    spike_trains_p = sm_p.spike_trains()
    # remove spike_trains of cell that do not fire
    true_spike_trains_p = [spike_trains_p[i] for i in range(len(spike_trains_p)) if len(spike_trains_p[i]) >= 2]
    # take only spikes in last portion (t>= time_to_take)
    time_to_take = 1 * second
    portion_sp_p = {}

    for i in range(len(true_spike_trains_p)):
        current_sp_p = spike_trains_p[i]
        current_sp_p = current_sp_p[current_sp_p >= time_to_take]
        portion_sp_p[i] = current_sp_p
    true_portion_sp_p = [portion_sp_p[i] for i in range(len(portion_sp_p)) if len(portion_sp_p[i]) >= 2]
    ISI_p = [np.diff(true_portion_sp_p[i]) for i in range(len(true_portion_sp_p))]
    CV_p = [std(ISI_p[i]) / mean(ISI_p[i]) for i in range(len(ISI_p))]
    N_spiked_p = len(CV_p)

    if sm_b is not None:
        spike_trains_b = sm_b.spike_trains()
        # remove spike_trains of cell that do not fire
        true_spike_trains_b = [spike_trains_b[i] for i in range(len(spike_trains_b)) if len(spike_trains_b[i]) >= 2]
        portion_sp_b = {}

        for i in range(len(true_spike_trains_b)):
            current_sp_b = spike_trains_b[i]
            current_sp_b = current_sp_b[current_sp_b >= time_to_take]
            portion_sp_b[i] = current_sp_b
        true_portion_sp_b = [portion_sp_b[i] for i in range(len(portion_sp_b)) if len(portion_sp_b[i]) >= 2]
        ISI_b = [np.diff(true_portion_sp_b[i]) for i in range(len(true_portion_sp_b))]
        CV_b = [std(ISI_b[i]) / mean(ISI_b[i]) for i in range(len(ISI_b))]
        N_spiked_b = len(CV_b)
    else:
        CV_b = None

    if sm_a is not None:
        spike_trains_a = sm_a.spike_trains()
        # remove spike_trains of cell that do not fire
        true_spike_trains_a = [spike_trains_a[i] for i in range(len(spike_trains_a)) if len(spike_trains_a[i]) >= 2]
        portion_sp_a = {}

        for i in range(len(true_spike_trains_a)):
            current_sp_a = spike_trains_a[i]
            current_sp_a = current_sp_a[current_sp_a >= time_to_take]
            portion_sp_a[i] = current_sp_a
        true_portion_sp_a = [portion_sp_a[i] for i in range(len(portion_sp_a)) if len(portion_sp_a[i]) >= 2]
        ISI_a = [np.diff(true_portion_sp_a[i]) for i in range(len(true_portion_sp_a))]
        CV_a = [std(ISI_a[i]) / mean(ISI_a[i]) for i in range(len(ISI_a))]
        N_spiked_a = len(CV_a)
    else:
        CV_a = None

    figure(figsize=[14, 6])
    subplot(131)
    plt.hist(CV_p, 20, color='r', alpha=0.8)
    xlabel('CV P pop')
    title('N cell that spiked: %d / %d ' % (N_spiked_p, len(spike_trains_p)))

    if sm_b is not None:
        subplot(132)
        try:
            plt.hist(CV_b, 20, color='b', alpha=0.8)
        except:
            print('not enough bins for B')
        xlabel('CV B pop')
        title('N cell that spiked:  %d / %d ' % (N_spiked_b, len(spike_trains_b)))

    if sm_a is not None:
        subplot(133)
        try:
            plt.hist(CV_a, 20, color='g', alpha=0.8)
        except:
            print('not enough bins for A')
        xlabel('CV A pop')
        title('N cell that spiked:  %d / %d ' % (N_spiked_a, len(spike_trains_a)))

    return CV_p, CV_b, CV_a


def inside_analyze_spont(info_dictionary, use_b_input=False, detection_thr=30, min_dist_idx=1000):
    """Procedure to analyze spontaneous SWR events
    :param info_dictionary: dic
        Dictionary where simulation results are stored
    :param use_b_input: bool
        If True, use LFP approximated signal for the analysis, Otherwise, use B average FR
    :param detection_thr: int
        Minimal height of peak to detect
    :param min_dist_idx: int
        Minimal distance (in array length) between two successively detected peaks
    :return:

        IEI_end_start_FWHM: numpy.ndarray
            Array of inter-event-intervals
        amp_peaks:  numpy.ndarray
            Amplitude of SWR events
        durations_spont: numpy.ndarray
            Duration of SWR events (FWHM)
        t_spont: numpy.ndarray
            Time array of simulation when spontaneous SWR have started (after warm-up)
        trace_spont: numpy.ndarray
            Simulation trace (LFP or B FR) when spontaneous SWR have started (after warm-up)
        filt_trace: numpy.ndarray
            Low-pass filtered version of trace_spont (for plotting purposes)
    """

    if use_b_input:
        mean_b_input_to_p = info_dictionary['mean_b_input_to_p']
        trace_fr = -mean_b_input_to_p
    else:
        FRB_smooth = info_dictionary['FRB_smooth']
        trace_fr = FRB_smooth

    t_fr = info_dictionary['time_array']
    start_spont = info_dictionary['start_spont']

    idx_start_spont = np.where(t_fr >= start_spont)[0][0]
    trace_spont = trace_fr[idx_start_spont:]
    t_spont = t_fr[idx_start_spont:]
    compress_step_indices = int(info_dictionary['compress_step_indices'])

    # filter trace with Butterworth filter
    fs = 1e4 / compress_step_indices
    lowcut = -1.
    highcut = 5.
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2, btype='low')
    filt_trace = filtfilt(b, a, trace_spont)

    if use_b_input:
        idx_peaks_spont = detect_peaks(filt_trace, mph=detection_thr, mpd=int(min_dist_idx / compress_step_indices),
                                       show=False)
    else:
        idx_peaks_spont = detect_peaks(filt_trace, mph=detection_thr, mpd=int(min_dist_idx / compress_step_indices),
                                       show=False)
    # leave out first and last event as they might not have enough baseline on the sides
    idx_peaks_spont = idx_peaks_spont[1:-1]

    # Amplitude of events
    amp_peaks = filt_trace[idx_peaks_spont]
    # Define baseline firing
    event_range = int(2000 / compress_step_indices)  # take the 200ms before and after peak
    mat_to_take = np.array(
        [list(range(my_peak - event_range, my_peak + event_range + 1)) for my_peak in idx_peaks_spont])
    sweeps = filt_trace[mat_to_take]  # num_peaks * 2*event_range + 1
    baseline = np.mean(sweeps[:, :int(event_range / 2)])

    half_max = (amp_peaks - baseline) / 2. + baseline
    # Times of SWR events
    time_peaks = t_spont[idx_peaks_spont]
    # find Full Width at Half Maximum
    nearest_start_idx = np.empty_like(time_peaks)
    nearest_end_idx = np.empty_like(time_peaks)
    my_wind = int(1000 / compress_step_indices)  # look back and forward until +- 100 ms from peak
    for n_idx, el in enumerate(idx_peaks_spont):
        # this in the idx in the cut out portion
        aux_idx = (
            np.abs(filt_trace[idx_peaks_spont[n_idx] - my_wind:idx_peaks_spont[n_idx]] - half_max[n_idx])).argmin()
        # we need this to recover the idx in the full trace
        nearest_start_idx[n_idx] = aux_idx + idx_peaks_spont[n_idx] - my_wind
        aux_idx = (
            np.abs(filt_trace[idx_peaks_spont[n_idx]: idx_peaks_spont[n_idx] + my_wind] - half_max[n_idx])).argmin()
        nearest_end_idx[n_idx] = aux_idx + idx_peaks_spont[n_idx]

    nearest_start_idx = nearest_start_idx.astype(int)
    nearest_end_idx = nearest_end_idx.astype(int)
    FWHM = t_spont[nearest_end_idx] - t_spont[nearest_start_idx]

    # IEI: distance between end of one event and start of the next one
    IEI_end_start_FWHM = t_spont[nearest_start_idx[1:]] - t_spont[nearest_end_idx[:-1]]

    # use FWHM as a proxy of SWR duration
    durations_spont = FWHM * 1e3

    return IEI_end_start_FWHM, amp_peaks, durations_spont, t_spont, trace_spont, filt_trace


def inside_analyze_evoked(info_dictionary, use_b_input=False):
    """Procedure to analyze evoked SWR events. The spontaneous events present in the trace are NOT used in the analysis.

    :param info_dictionary: dic
        Dictionary where simulation results are stored
    :param use_b_input: bool
        If True, use LFP approximated signal for the analysis, Otherwise, use B average FR

    :return:

        IEI_end_start_FWHM_NEXT_evoked: numpy.ndarray
            Array of inter-event-intervals from spontaneous to NEXT evoked event
        IEI_end_start_FWHM_PREV_evoked: numpy.ndarray
            Array of inter-event-intervals from evoked event to following spontaneous event
        amp_evoked:  numpy.ndarray
            Amplitude of evoked SWR events
        durations_evoked: numpy.ndarray
            Duration of evoked SWR events (FWHM)
        t_spont: numpy.ndarray
            Time array of simulation when spontaneous SWR have started (after warm-up)
        trace_spont: numpy.ndarray
            Simulation trace (LFP or B FR) when spontaneous SWR have started (after warm-up)
        filt_trace: numpy.ndarray
            Low-pass filtered version of trace_spont (for plotting purposes)
    """

    if use_b_input:
        mean_b_input_to_p = info_dictionary['mean_b_input_to_p']
        trace_fr = -mean_b_input_to_p
    else:
        FRB_smooth = info_dictionary['FRB_smooth']
        trace_fr = FRB_smooth

    t_fr = info_dictionary['time_array']
    start_spont = info_dictionary['start_spont']
    stim_times_array = info_dictionary['stim_times_array']

    idx_start_spont = np.where(t_fr >= start_spont)[0][0]
    trace_spont = trace_fr[idx_start_spont:]
    t_spont = t_fr[idx_start_spont:]
    compress_step_indices = int(info_dictionary['compress_step_indices'])

    # filter trace with Butterworth filter
    fs = 1e4 / compress_step_indices
    lowcut = -1.
    highcut = 5.
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=2, btype='low')
    filt_trace = filtfilt(b, a, trace_spont)

    # detect peaks of filtered trace
    idx_peaks_spont = detect_peaks(filt_trace, mph=40, mpd=int(1000 / compress_step_indices), show=False)
    # leave out first and last event as they might not have enough baseline on the sides
    idx_peaks_spont = idx_peaks_spont[1:-1]

    time_peaks = t_spont[idx_peaks_spont]
    # ======== Select only evoked events!
    time_of_stim = stim_times_array
    evoked_idx = np.zeros_like(time_of_stim)
    previous_idx = np.zeros_like(time_of_stim)
    following_idx = np.zeros_like(time_of_stim)

    for idx_time, my_time in enumerate(time_of_stim[:-1]):

        aux_idx = (np.abs(time_peaks - my_time)).argmin()
        # has to come _after the stimulation
        if (0 < time_peaks[aux_idx] - my_time) and (time_peaks[aux_idx] - my_time <= 0.05):  # unit is second
            evoked_idx[idx_time] = np.abs(t_spont - time_peaks[aux_idx]).argmin()
            previous_idx[idx_time] = np.abs(t_spont - time_peaks[aux_idx - 1]).argmin()
            following_idx[idx_time] = np.abs(t_spont - time_peaks[aux_idx + 1]).argmin()

    previous_idx = previous_idx[evoked_idx > 0]
    following_idx = following_idx[evoked_idx > 0]
    evoked_idx = evoked_idx[evoked_idx > 0]
    # remove first element of array as there might be no previous one for the first evoked peak
    if previous_idx[0] > evoked_idx[0]:
        evoked_idx = evoked_idx[1:]
        previous_idx = previous_idx[1:]
        following_idx = following_idx[1:]
    if following_idx[-1] < evoked_idx[-1]:
        evoked_idx = evoked_idx[:-1]
        previous_idx = previous_idx[:-1]
        following_idx = following_idx[:-1]

    evoked_idx = evoked_idx.astype(int)
    previous_idx = previous_idx.astype(int)
    following_idx = following_idx.astype(int)

    # Amplitude of events
    amp_evoked = filt_trace[evoked_idx]
    amp_previous = filt_trace[previous_idx]
    amp_following = filt_trace[following_idx]
    # put together all 'sweeps' (evoked + previous ones) to take first and last portion to define baseline firing
    event_range = int(2000 / compress_step_indices)  # take the 200ms before and after peak

    sweeps = np.empty((3 * len(evoked_idx), event_range * 2))
    for n_peak, my_peak_idx in enumerate(evoked_idx):
        sweeps[n_peak, :] = filt_trace[my_peak_idx - event_range: my_peak_idx + event_range]
        sweeps[n_peak + len(evoked_idx), :] = filt_trace[
                                              previous_idx[n_peak] - event_range: previous_idx[n_peak] + event_range]
        sweeps[n_peak + 2 * len(evoked_idx), :] = filt_trace[following_idx[n_peak] - event_range: following_idx[
                                                                                                             n_peak] + event_range]
    # baseline in average across sweeps and across first 100 ms of them
    baseline = np.mean(sweeps[:, :int(event_range / 2)])  # use initial portion

    half_max_evoked = (amp_evoked - baseline) / 2. + baseline
    half_max_previous = (amp_previous - baseline) / 2. + baseline
    half_max_following = (amp_following - baseline) / 2. + baseline

    # find Full Width at Half Maximum
    # ======= For evoked events
    nearest_start_idx_evoked = np.empty_like(amp_evoked)
    nearest_end_idx_evoked = np.empty_like(amp_evoked)
    my_wind = int(1000 / compress_step_indices)  # look back and forward until +- 100 ms from peak
    for n_idx, el in enumerate(evoked_idx):
        # this in the idx in the cut out portion
        aux_idx = (np.abs(filt_trace[el - my_wind:el] - half_max_evoked[n_idx])).argmin()
        # we need this to recover the idx in the full trace
        nearest_start_idx_evoked[n_idx] = aux_idx + el - my_wind
        aux_idx = (np.abs(filt_trace[el: el + my_wind] - half_max_evoked[n_idx])).argmin()
        nearest_end_idx_evoked[n_idx] = aux_idx + el

    # ======= For previous events
    nearest_start_idx_previous = np.empty_like(amp_previous)
    nearest_end_idx_previous = np.empty_like(amp_previous)
    for n_idx, el in enumerate(previous_idx):
        # this in the idx in the cut out portion
        aux_idx = (np.abs(filt_trace[el - my_wind:el] - half_max_previous[n_idx])).argmin()
        # we need this to recover the idx in the full trace
        nearest_start_idx_previous[n_idx] = aux_idx + el - my_wind
        aux_idx = (np.abs(filt_trace[el: el + my_wind] - half_max_previous[n_idx])).argmin()
        nearest_end_idx_previous[n_idx] = aux_idx + el

    # ========= For following events
    nearest_start_idx_following = np.empty_like(amp_following)
    nearest_end_idx_following = np.empty_like(amp_following)
    for n_idx, el in enumerate(following_idx):
        # this in the idx in the cut out portion
        aux_idx = (np.abs(filt_trace[el - my_wind:el] - half_max_following[n_idx])).argmin()
        # we need this to recover the idx in the full trace
        nearest_start_idx_following[n_idx] = aux_idx + el - my_wind
        aux_idx = (np.abs(filt_trace[el: el + my_wind] - half_max_following[n_idx])).argmin()
        nearest_end_idx_following[n_idx] = aux_idx + el

    nearest_start_idx_evoked = nearest_start_idx_evoked.astype(int)
    nearest_end_idx_evoked = nearest_end_idx_evoked.astype(int)
    FWHM_evoked = t_spont[nearest_end_idx_evoked] - t_spont[nearest_start_idx_evoked]

    nearest_end_idx_previous = nearest_end_idx_previous.astype(int)
    nearest_start_idx_following = nearest_start_idx_following.astype(int)

    # IEI - distance between end of *previous* event and start of the *evoked* SWR
    IEI_end_start_FWHM_NEXT_evoked = t_spont[nearest_start_idx_evoked] - t_spont[nearest_end_idx_previous]

    # for opposite correlation, need IEI between evoked and *next* spontaneous SWR
    IEI_end_start_FWHM_PREV_evoked = t_spont[nearest_start_idx_following] - t_spont[nearest_end_idx_evoked]

    # use FWHM as the duration
    durations_evoked = FWHM_evoked * 1e3  # in ms

    return IEI_end_start_FWHM_NEXT_evoked, IEI_end_start_FWHM_PREV_evoked, amp_evoked, durations_evoked, \
           t_spont, trace_spont, filt_trace
