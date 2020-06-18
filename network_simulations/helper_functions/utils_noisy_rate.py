import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import os

from matplotlib import rc
rc('text', usetex=True)

from helper_functions.utils_spiking import create_butter_bandpass, fit_func
from helper_functions.detect_peaks import detect_peaks
import helper_functions.params_noisy_rate as pm

sys.path.append(os.path.dirname( __file__ ) + '/../../')
import bifurcation_analysis.figures_code.helper_functions.bifurcations as bif
import bifurcation_analysis.figures_code.helper_functions.aux_functions as aux

def get_peak_data(t, b, b_pulses, sim_type):

    dt = t[1] - t[0]
    simtime = (t[-1] - t[0])/1e3

    b_butter, a_butter = create_butter_bandpass(-1, pm.b_findpeak_cutoff, 1e3/dt, order=2, btype='low')
    lowpass_b = signal.filtfilt(b_butter, a_butter, b)

    peaks = detect_peaks(lowpass_b, mph=pm.b_findpeak_height, mpd=int(pm.b_findpeak_dist/dt), show=False)

    # Find start and end points:
    start = np.zeros(peaks.size,dtype=int)
    end = np.zeros(peaks.size,dtype=int)
    duration = np.zeros(peaks.size)
    if peaks.size > 0:
        IEI = np.zeros(peaks.size-1, dtype=float)

        i = 0
        j = 0
        in_peak = False
        halfmax = lowpass_b[peaks[0]]/2
        while(i < t.size and j < peaks.size):

            if (in_peak == False) and (lowpass_b[i] > halfmax) and (i < peaks[j]):
                # print('passed start')
                start[j] = i
                in_peak = True

            if (in_peak == True) and (lowpass_b[i] < halfmax) and (i > peaks[j]):
                # print('passed end')
                end[j] = i
                in_peak = False

            if (j < peaks.size - 1) and (i > peaks[j]) and (peaks[j+1] - i < i - peaks[j]):
                j = j + 1
                halfmax = lowpass_b[peaks[j]]/2

            i = i + 1

        duration = t[end] - t[start]

        if (duration <= 0).any():
            print('ERROR: Failed finding peaks')
            return False

        # Calculate Inter-Event-Intervals:
        for i in range(peaks.size-1):
            IEI[i] = (t[start[i+1]] - t[end[i]])*1e-3 # convert to seconds

        duration_prev = 0
        duration_next = 0
        IEI_prev = 0
        IEI_next = 0
        b_pulses_onset = 0
        b_pulses_success = 0

        if (sim_type is 'spont'):
            duration_prev = duration[1:]
            duration_next = duration[:-1]
            IEI_prev = IEI
            IEI_next = IEI

        if (sim_type is 'evoke'):
            if b_pulses.any() > 0:

                # Get array with start of each pulse:
                for i in range(len(b_pulses) - 1):
                    if (b_pulses[i+1] > 0) and (b_pulses[i] == 0):
                        b_pulses_onset = np.append(b_pulses_onset,t[i])
                b_pulses_onset = b_pulses_onset[1:]
                # Check whether each pulse triggers a spike in the 20 ms following pulse onset:
                b_pulses_success = np.zeros_like(b_pulses_onset)
                evoked_peaks = np.array([-1])
                for i in range(len(b_pulses_onset)):
                    for j in range(peaks.size):
                        if (t[start[j]] - b_pulses_onset[i]) > 0 and (t[start[j]] - b_pulses_onset[i]) <= 50:
                            b_pulses_success[i] = 1
                            evoked_peaks = np.append(evoked_peaks,j)
                if len(evoked_peaks) > 1:
                    evoked_peaks = evoked_peaks[1:]
                    if (evoked_peaks[-1] == peaks.size - 1):
                        evoked_peaks = evoked_peaks[:-1]

                    # Get data only from evoked peaks:
                    start = start[evoked_peaks]
                    end = end[evoked_peaks]
                    duration_prev = duration[evoked_peaks]
                    duration_next = duration[evoked_peaks]
                    IEI_prev = IEI[evoked_peaks-1]
                    IEI_next = IEI[evoked_peaks]
                else:
                    evoked_peaks = 0
                    start = 0
                    end = 0
                    duration_prev = 0
                    duration_next = 0
                    IEI_prev = 0
                    IEI_next = 0
            else:
                peak_data = False

        peak_data = start, end, duration_prev, duration_next, IEI_prev, IEI_next
    if peaks.size == 0:
        print('ERROR: Failed finding peaks')
        peak_data = False
        b_pulses_onset = 0
        b_pulses_success = 0

    fit_params = np.array([0,0,0])
    if peak_data != False:
        peak_start, peak_end, peaks_duration_prev, peaks_duration_next, peaks_IEI_prev, peaks_IEI_next = peak_data
        try: fit_params, _ = curve_fit(fit_func, peaks_IEI_prev, peaks_duration_prev,p0=(2,2,68), bounds=(0,[100, 100, 100]))
        except: None

    if peak_data != False:
        print('Minimum IEI is %.1lf ms'%(np.min(peaks_IEI_prev)*1000))
        print('Peak incidence = %.2lf Hz'%(len(peak_start)/simtime))
        print('Time constant is %.1lf ms'%(1e3/fit_params[1]))
        print('previous IEI = (%.2lf +/- %.2lf) s'%(np.mean(peaks_IEI_prev),np.std(peaks_IEI_prev)))
        print('next IEI = (%.2lf +/- %.2lf) s'%(np.mean(peaks_IEI_next),np.std(peaks_IEI_next)))
        c, p = pearsonr(peaks_IEI_prev,peaks_duration_prev)
        print('Correlation with Previous IEI: c = %.3lf, p = %.2e'%(c,p))
        c, p = pearsonr(peaks_IEI_next,peaks_duration_next)
        print('Correlation with Next IEI: c = %.3lf, p = %.2e'%(c,p))

    return lowpass_b, peak_data, fit_params, b_pulses_onset, b_pulses_success

def plot_one_side(fig, grid, sim_type, tstart, tstop, t, b, e, b_pulses, lowpass_b, peak_data, fit_params, b_pulses_onset, b_pulses_success):

    if peak_data != False:
        peaks_start, peaks_end, peaks_duration_prev, peaks_duration_next, peaks_IEI_prev, peaks_IEI_next = peak_data

    ax_t_B = fig.add_subplot(grid[0, 0:4])
    ax_t_e = fig.add_subplot(grid[1:3, 0:4])

    ax_e_B = fig.add_subplot(grid[0, 4:6])
    ax_IEI_hist = fig.add_subplot(grid[2, 4:6])

    ax_prev = fig.add_subplot(grid[4,0:3])
    ax_next = fig.add_subplot(grid[4,3:6])

    for ax in [ax_t_B, ax_t_e]:
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', direction='out', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', direction='out', bottom=False, top=False, labelbottom=False, labelsize=pm.fonts)
        ax.set_xlim([tstart,tstop])

    for ax in [ax_e_B, ax_IEI_hist, ax_prev, ax_next]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', direction='out', bottom=True, top=False, labelbottom=True, labelsize=pm.fonts)
        ax.tick_params(axis='y', which='both', direction='out', bottom=False, top=False, labelbottom=False, labelsize=pm.fonts)

    ax_IEI_hist.spines['left'].set_visible(False)

    B_ticks = [0,45,90]

    if sim_type is 'spont':
        ax_t_B.set_title(r'\textbf{A1}',loc='left',x=-0.1,y=0.95,fontsize=pm.fonts)
        ax_t_B.set_title(r'Spontaneous',loc='center',x=0.75,y=1.20,fontsize=pm.fonts)
    else:
        ax_t_B.set_title(r'\textbf{A2}',loc='left',x=-0.1,y=0.95,fontsize=pm.fonts)
        ax_t_B.set_title(r'Evoked',loc='center',x=0.75,y=1.20,fontsize=pm.fonts)

    ax_t_B.plot(t, b, color='#3c3fef', lw=1.5)
    ax_t_B.axhline(112,xmin=0.12,xmax=(0.12+1/6), linewidth=1, color='black')
    ax_t_B.text(tstart+(tstop-tstart)*0.08,120,'250 ms',fontsize=pm.fonts)

    if (sim_type == 'evoke') and (b_pulses.any() > 0) and (peak_data != False):
        ytop, ybottom = ax_t_B.get_ylim()
        ax_t_B.fill_between(t, ybottom, ytop, where = b_pulses > 0, facecolor='#d4b021')
        for i in range(len(b_pulses_onset)):
            if (b_pulses_onset[i] >= tstart) and (b_pulses_onset[i] <= tstop):
                if b_pulses_success[i] == 1:
                    ax_t_B.annotate('', xy=(b_pulses_onset[i], 0), xytext=(b_pulses_onset[i], -30),\
                        xycoords='data',arrowprops=dict(arrowstyle="->", lw=1.,color='black'))
                else:
                    ax_t_B.annotate('', xy=(b_pulses_onset[i], 0), xytext=(b_pulses_onset[i], -30),\
                        xycoords='data',arrowprops=dict(arrowstyle="->", lw=1.,color='lightgray'))
    ax_t_B.set_ylim([-0.1*B_ticks[-1],1.3*B_ticks[-1]])
    if sim_type is 'spont': ax_t_B.set_ylabel("B [1/s]",fontsize=pm.fonts)
    ax_t_B.set_yticks(B_ticks)
    ax_t_B.set_yticklabels(B_ticks,fontsize=pm.fonts)

    ax_t_e.plot(t, e, color='#e67e22', lw=1.5)
    ax_t_e.set_yticks([0.5,1.0])
    ax_t_e.set_yticklabels([0.5,1.0],fontsize=pm.fonts)
    ax_t_e.set_ylim([0.3,1.1])
    if sim_type is 'spont': ax_t_e.set_ylabel("e",fontsize=pm.fonts)

    bif_path = os.path.dirname( __file__ ) + '/../../bifurcation_analysis/bifurcation_diagrams/1param/'
    bs = bif.load_bifurcations(bif_path, 'e', 0, 1)

    if sim_type is 'spont':
        ax_e_B.set_title(r'\textbf{B1}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)
    else:
        ax_e_B.set_title(r'\textbf{B2}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)

    bif.plot_bifurcation(ax_e_B,aux,bs,'B',[0.25,1],1,'',[0.4,0.8],[0.4,0.8],B_ticks,B_ticks,pm.fonts,plot_color='gray',line_width=1.,inward_ticks=False)
    ax_e_B.set_ylabel('',fontsize=pm.fonts)
    ax_e_B.plot(e[(t >= tstart) & (t <= tstop)], b[(t >= tstart) & (t <= tstop)], color='#3c3fef', lw=1.5)
    ax_e_B.set_ylim([-0.1*B_ticks[-1],1.2*B_ticks[-1]])
    ax_e_B.set_title('e', y=0.95, fontsize=pm.fonts)

    if peak_data != False:

        if sim_type is 'spont':
            ax_IEI_hist.set_title(r'\textbf{C1}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)
        else:
            ax_IEI_hist.set_title(r'\textbf{C2}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)

        ax_IEI_hist.hist(peaks_IEI_prev,bins=30,color='gray')
        ax_IEI_hist.set_title('IEI [s]', y=0.95, fontsize=pm.fonts)
        ax_IEI_hist.set_xlim([0,3])
        ax_IEI_hist.set_xticks([0,1,2])
        ax_IEI_hist.set_xticklabels([0,1,2],fontsize=pm.fonts)
        ax_IEI_hist.set_yticks([])
        ax_IEI_hist.set_yticklabels([])

        if sim_type is 'spont':
            ax_prev.set_title(r'\textbf{D1}',loc='left',x=-0.15,y=0.95,fontsize=pm.fonts)
        else:
            ax_prev.set_title(r'\textbf{D2}',loc='left',x=-0.15,y=0.95,fontsize=pm.fonts)

        ax_prev.plot(peaks_IEI_prev, peaks_duration_prev, 'k.', ms=3)
        ax_prev.axvline(np.min(peaks_IEI_prev), linewidth=1, color='k', linestyle='--')
        if fit_params.all != 0:
            x_array = np.arange(np.min(peaks_IEI_prev),np.max(peaks_IEI_prev),0.01)
            ax_prev.plot(x_array,fit_func(x_array,*fit_params),color='red', lw=1.5)
        ax_prev.set_xlabel('Previous IEI [s]',fontsize=pm.fonts)
        if sim_type is 'spont': ax_prev.set_ylabel('FWHM [ms]',fontsize=pm.fonts)
        ax_prev.set_ylim([30,105])
        ax_prev.set_yticks([45,90])
        ax_prev.set_yticklabels([45,90],fontsize=pm.fonts)
        ax_prev.set_xlim([0,3])
        ax_prev.set_xticks([0,1,2])
        ax_prev.set_xticklabels([0,1,2],fontsize=pm.fonts)

        ax_next.plot(peaks_IEI_next, peaks_duration_next, 'k.', ms=3)
        ax_next.set_xlabel('Next IEI [s]',fontsize=pm.fonts)
        ax_next.set_ylim([30,105])
        ax_next.set_yticks([45,90])
        ax_next.set_yticklabels([45,90],fontsize=pm.fonts)
        ax_next.set_xlim([0,3])
        ax_next.set_xticks([0,1,2])
        ax_next.set_xticklabels([0,1,2],fontsize=pm.fonts)

def plot_fig_12():
    print('============Spontaneous Events============')
    data_spont = np.load('results/noisy_rate_model_long_sim_spont.npz', encoding='latin1', allow_pickle=True)
    dict_spont = dict(zip(("{}".format(k) for k in data_spont), (data_spont[k] for k in data_spont)))
    t_spont = dict_spont['t']
    b_spont = dict_spont['b']
    e_spont = dict_spont['e']
    b_pulses_spont = dict_spont['b_pulses']

    lowpass_b_spont, peak_data_spont, fit_data_spont, _, _ = get_peak_data(t_spont, b_spont, b_pulses_spont, 'spont')

    print('==============Evoked Events===============')
    data_evoke = np.load('results/noisy_rate_model_long_sim_evoke.npz', encoding='latin1', allow_pickle=True)
    dict_evoke = dict(zip(("{}".format(k) for k in data_evoke), (data_evoke[k] for k in data_evoke)))
    t_evoke = dict_evoke['t']
    b_evoke = dict_evoke['b']
    e_evoke = dict_evoke['e']
    b_pulses_evoke = dict_evoke['b_pulses']

    lowpass_b_evoke, peak_data_evoke, fit_data_evoke, b_pulses_onset, b_pulses_success = get_peak_data(t_evoke, b_evoke, b_pulses_evoke, 'evoke')

    fig_width = 17.6/2.54
    fig_height = 0.4*17.6/2.54

    fig = plt.figure(figsize=(fig_width,fig_height))
    gs = gridspec.GridSpec(8, 13, width_ratios=[1,1,1,1,1,1,0.2,1,1,1,1,1,1])
    gs_spont = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs[:, 0:6], height_ratios=[1.,0.45,0.3,0.25,1.])
    gs_evoke = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs[:, 7:13], height_ratios=[1.,0.45,0.3,0.25,1.])

    t_plot_start = 195.4*1e3
    t_plot_stop = 196.9*1e3

    plot_one_side(fig, gs_spont, 'spont', t_plot_start, t_plot_stop,\
                    t_spont, b_spont, e_spont, b_pulses_spont, lowpass_b_spont, peak_data_spont, fit_data_spont, None, None)
    plot_one_side(fig, gs_evoke, 'evoke', t_plot_start, t_plot_stop,\
                    t_evoke, b_evoke, e_evoke, b_pulses_evoke, lowpass_b_evoke, peak_data_evoke, fit_data_evoke, b_pulses_onset, b_pulses_success)

    plt.subplots_adjust(wspace=1.5, hspace=0.25)

    fig.savefig('results/fig_rate_model_noise.eps', bbox_inches='tight', dpi=800)