__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains supporting functions needed to create the rate model from the spiking network and run the network
simulations shown in Fig. 2-2, 6-2"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class FullParamFromDict(object):
    def __init__(self, my_dic, dic_idx=-1):
        for key in my_dic:
            setattr(self, key, my_dic[key])

        self.dic = my_dic
        self.dic_idx = dic_idx

    def print_elements(self):
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))


class DeprClamp:
    """ Param class contains the variables of the simulation """

    def __init__(self):
        self.dt = 0.1 * 1e-3
        self.t_max = 3000 * 1e-3  # s
        # clamp values
        self.first_change = 0.5  # d value
        self.second_change = 0.2
        self.third_change = 0.8
        # clamp times
        self.t_first = 1500 * 1e-3  # time of change
        self.t_second = 2000 * 1e-3


class SimParam:
    """ Param class contains the variables of the rate-model simulation """

    def __init__(self):
        self.dt = 0.1 * 1e-3
        self.t_max = 800 * 1e-3  # ms

        # length of pulse stimulus
        self.pulse_lim = 30 * 1e-3  # np.infty         # ms
        self.pulse_start = 100 * 1e-3  # ms     (to start from steady state if long enough)
        self.second_pulse_start = 300 * 1e-3
        self.third_pulse_start = 600 * 1e-3


def is_nan(x):
    """Determines if array is NaN"""
    return (x is np.nan or x != x)


def softplus_func_mean_field(x, k, t):
    """Softplus function using k_I and t_I, used in rate model simulation for figs. 5 and 10"""
    return np.log(1. + np.exp(k*(x + t)))


def gen_threshold_linear(x, k_x, t_x):
    """generic threshold linear function, approximation of the softplus function"""
    return (x >= -t_x)*(k_x * (x + t_x)) + (x < -t_x) * 0.


def eq_4d(x, t, net, sim, curr_value, constant=False, inj_curr='I_p', use_softplus=True):
    """Function used to simulate full 4d system using softplus (or threshold-linear) functions as activation functions

    :param x: 4d array, float
        Array of variables: P, B, A, d
    :param t: float
        Time point in simulation
    :param net: class FullParamFromDict
        Contains parameters of the rate model
    :param sim: class SimParam
        Contains parameters of the rate model simulation
    :param curr_value: float
        Value of current injected (in unit of pA)
    :param constant: bool
        If False, current is time-dependent (dafault)
    :param inj_curr: str
        To which population current should be injected
    :param use_softplus: bool
        If True (default), uses the softplus function to simulate the rate model. Alternatively, it uses its
        threshold-linear approximation

    :returns fv: 4d array, float
        Updated values of P,B,A, and d variables
    """
    P, B, A, d1 = x

    # case with all possible current
    current_I_p = 0.
    current_I_b = 0.
    current_I_a = 0.

    if constant:
        aux_curr = curr_value
    else:
        aux_curr = curr_value * (sim.pulse_start <= t) * (t < sim.pulse_lim + sim.pulse_start)

    if inj_curr == 'I_b':
        current_I_b = aux_curr
    elif inj_curr == 'I_p':
        current_I_p = aux_curr
    elif inj_curr == 'I_a':
        current_I_a = - aux_curr  # negative!!

    if use_softplus:

        f_p = 1.0 / net.tau_p * (- P + softplus_func_mean_field(net.W_pp * P - net.W_pb * B - net.W_pa * A + current_I_p,
                                                                net.k_p, net.t_p))
        f_b = 1.0 / net.tau_b * (- B + softplus_func_mean_field(net.W_bp * P - net.W_bb * B - net.W_ba * A + current_I_b,
                                                                net.k_b, net.t_b))
        f_a = 1.0 / net.tau_a * (
                    - A + softplus_func_mean_field(net.W_ap * P - net.W_ab * B * d1 - net.W_aa * A + current_I_a,
                                                   net.k_a, net.t_a))
    else:
        f_p = 1.0 / net.tau_p * (- P + gen_threshold_linear(net.W_pp * P - net.W_pb * B - net.W_pa * A + current_I_p,
                                                                net.k_p, net.t_p))
        f_b = 1.0 / net.tau_b * (- B + gen_threshold_linear(net.W_bp * P - net.W_bb * B - net.W_ba * A + current_I_b,
                                                                net.k_b, net.t_b))
        f_a = 1.0 / net.tau_a * (
                    - A + gen_threshold_linear(net.W_ap * P - net.W_ab * B * d1 - net.W_aa * A + current_I_a,
                                                   net.k_a, net.t_a))

    f_d1 = 1.0 / net.tau_d * (1. - d1) - net.eta * d1 * B

    fv = np.hstack((f_p, f_b, f_a, f_d1))
    return fv


def eq_clamp_depression(x, t, net, sim, depr_clamp, curr_value, constant=False, inj_curr='I_p', use_softplus=True):
    """Function used to simulate the 3d system using softplus (or thr linear) functions as activation functions.
    Needed to check if system goes to inside-SPW state upon current stimulation

    :param x: 4d array, float
        Array of variables: P, B, A, d
    :param t: float
        Time point in simulation
    :param net: class FullParamFromDict
        Contains parameters of the rate model
    :param sim: class SimParam
        Contains parameters of the rate model simulation
    :param depr_clamp: class DeprClamp
        Contains properties of the synaptic efficacy behvaior over time
    :param curr_value: float
        Value of current injected (in unit of pA)
    :param constant: bool
        If False, current is time-dependent (dafault)
    :param inj_curr: str
        To which population current should be injected
    :param use_softplus: bool
        If True (default), uses the softplus function to simulate the rate model. Alternatively, it uses its
        threshold-linear approximation

    :returns fv: 4d array, float
        Updated values of P,B,A, and d variables
    """

    P, B, A, d = x

    # case with all possible current
    current_I_p = 0.
    current_I_b = 0.
    current_I_a = 0.

    if constant:
        aux_curr = curr_value
    else:
        aux_curr = curr_value * (sim.pulse_start <= t) * (t < sim.pulse_lim + sim.pulse_start) \
                   + (-curr_value) * (sim.second_pulse_start <= t) * (t < sim.pulse_lim + sim.second_pulse_start) \
                   + curr_value * (sim.third_pulse_start <= t) * (t < sim.pulse_lim + sim.third_pulse_start)

    if inj_curr == 'I_b':
        current_I_b = aux_curr
    elif inj_curr == 'I_p':
        current_I_p = aux_curr
    elif inj_curr == 'I_a':
        current_I_a = - aux_curr  # negative!!

    d = depr_clamp.first_change * (t <= depr_clamp.t_first) \
        + depr_clamp.second_change * (t > depr_clamp.t_first) * (t <= depr_clamp.t_second) \
        + depr_clamp.third_change * (t > depr_clamp.t_second)

    if use_softplus:

        f_p = 1.0 / net.tau_p * (- P + softplus_func_mean_field(net.W_pp * P - net.W_pb * B - net.W_pa * A + current_I_p,
                                                                net.k_p, net.t_p))
        f_b = 1.0 / net.tau_b * (- B + softplus_func_mean_field(net.W_bp * P - net.W_bb * B - net.W_ba * A + current_I_b,
                                                                net.k_b, net.t_b))
        f_a = 1.0 / net.tau_a * (
                - A + softplus_func_mean_field(net.W_ap * P - net.W_ab * B * d - net.W_aa * A + current_I_a,
                                               net.k_a, net.t_a))
    else:
        # use the threshold-linear approximations of the softplus functions
        f_p = 1.0 / net.tau_p * (- P + gen_threshold_linear(net.W_pp * P - net.W_pb * B - net.W_pa * A + current_I_p,
                                                                net.k_p, net.t_p))
        f_b = 1.0 / net.tau_b * (- B + gen_threshold_linear(net.W_bp * P - net.W_bb * B - net.W_ba * A + current_I_b,
                                                                net.k_b, net.t_b))
        f_a = 1.0 / net.tau_a * (
                - A + gen_threshold_linear(net.W_ap * P - net.W_ab * B * d - net.W_aa * A + current_I_a,
                                               net.k_a, net.t_a))

    fv = np.hstack((f_p, f_b, f_a, 0))
    return fv


# ============== Plotting routines
def fancy_plotting_2d(ax, volt_range):
    """Aux function for membrane potential optimization plot.
    We fixed P, so only B and A play a role here"""
    ax.set_xlabel('B membpot')
    ax.set_ylabel('A membpot')
    ax.set_xlim([-1, len(volt_range)])
    ax.set_xticks(np.arange(0, len(volt_range), 2.))
    ax.set_xticklabels([int(i) for i in volt_range[::2]])
    ax.set_ylim([-1, len(volt_range)])
    ax.set_yticks(np.arange(0, len(volt_range), 2.))
    ax.set_yticklabels([int(i) for i in volt_range[::2]])


def fancy_plotting_3d(ax, volt_range_P, volt_range_B, volt_range_A, my_size):
    """Aux function for membrane potential optimization plot"""
    ax.set_zlabel('\nA memb. pot. [mV]', linespacing=1., fontsize=my_size)
    ax.set_xlim([-1, len(volt_range_P)])
    my_step = 6
    ax.set_xticks(np.arange(0, len(volt_range_P), my_step))
    ax.set_xticklabels([int(i) for i in volt_range_P[::my_step]])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_yticklabels([int(i) for i in volt_range_B[::my_step]], rotation=-25,
                       verticalalignment='baseline',
                       horizontalalignment='left')
    ax.set_ylim([-1, len(volt_range_B)])
    ax.set_yticks(np.arange(0, len(volt_range_B), my_step))
    ax.set_yticklabels([int(i) for i in volt_range_B[::my_step]])
    ax.yaxis.set_tick_params(pad=-3)

    ax.set_zlim([-1, len(volt_range_A) + 5])
    ax.set_zticks(np.arange(0, len(volt_range_A), my_step))
    ax.set_zticklabels([int(i) for i in volt_range_A[::my_step]])


def plot_fancy_results(num_row, num_col, num_subplot, x, y, col, label_str, title_str, pulse_start, pulse_lim,
                       curr_value, line_style='-', spiking=True):
    """To reduce the plotting commands - For populations results"""

    ax = plt.subplot(num_row, num_col, num_subplot)
    plt.plot(x, y, line_style, color=col, linewidth=3., label=label_str)
    font_size = 12

    if num_subplot < 4:
        if label_str is not None:
            plt.legend(loc='best', frameon=False, prop={'size': 16})
        if spiking:
            ax.set_ylabel('population FR [Hz]', fontsize=font_size)
        else:
            ax.set_ylabel('population activity [a.u.]', fontsize=font_size)

    if num_subplot == 4:
        if label_str is not None:
            plt.legend(loc='best', frameon=False, prop={'size': 16})
        ax.set_ylabel('syn. depr. activity [a.u.]', fontsize=font_size)

    if num_subplot in [4, 5, 6]:
        if not spiking:
            ax.set_yticks([0.6, 0.8, 1])

    if num_subplot in [1, 2, 3]:
        if title_str != '' and title_str != 'P,B,A':
            ax.set_title(title_str, fontsize=font_size)
        if not spiking:
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'], fontdict=16)

    if title_str == 'P,B,A' or title_str == 'd':

        if curr_value != 0.0:
            # dashed area is where current injection was on
            # Make the shaded region
            ix = np.linspace(pulse_start, pulse_lim + pulse_start)
            if spiking:
                iy = np.linspace(140, 140)
            else:
                iy = np.linspace(1.1, 1.1)

            verts = [(pulse_start, 0)] + list(zip(ix, iy)) + [(pulse_lim + pulse_start, 0)]
            poly = Polygon(verts, facecolor='#d4b021', edgecolor='#d4b021', alpha=0.5)
            ax.add_patch(poly)
