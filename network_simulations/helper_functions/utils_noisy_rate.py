'''
    Functions to detect noisy rate model simulation peaks,
    calculate their properties, and plot Fig. 12
'''

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
    ''' Calculates noisy rate model peaks (SWR events) and their properties

    :param t:
        Simulation time
    :param b:
        Rate model variable b
    :param b_pulses:
        Periodic current injection step pulses to evoke events
    :param sim_type:
        Simulation type {'spont','evoked'}
            'spont': simulation with spontaneous events
            'evoked': simulation with both spontaneous and evoked events

    :returns lowpass_b:
        Low-pass filtered b trace
    :returns peak_data:
        Data from detected peaks {False, array}
            False: Failed detecting peaks
            array: Array containing peak data
                start: Peak start index
                end: Peak end index
                amp_prev: Peak amplitude for correlation with Previous IEI
                amp_next: Peak amplitude for correlation with Next IEI
                duration_prev: Peak duration for correlation with Previous IEI
                duration_next: Peak duration for correlation with Next IEI
                IEI_prev: Previous IEI
                IEI_next: Next IEI
    :returns fit_params:
        Parameters from fitting exponential to Previous IEI vs. peak amplitude
            If peak_data == False: fit_params = {0, 0, 0}
            Else: fit_params = {a, b, c} from "a * (1. - np.exp(-b * x)) + c"
    :returns b_pulses_onset:
        Array with onset time of periodic current step pulses
    :returns b_pulses_success:
        Array of 0s and 1s; tracks whether each current step pulse
        successfully triggered an event

    '''

    # Simulation dt (in ms)
    dt = t[1] - t[0]

    # Total simualtion time (in secs)
    simtime = (t[-1] - t[0])/1e3

    # Initialise peak_data:
    peak_data = True

    ''' Apply low-pass filter to b trace '''
    b_butter, a_butter = create_butter_bandpass(-1, pm.b_findpeak_cutoff, 1e3/dt, order=2, btype='low')
    lowpass_b = signal.filtfilt(b_butter, a_butter, b)

    ''' Detect peaks in low-pass-filtered b '''
    peaks = detect_peaks(lowpass_b, mph=pm.b_findpeak_height, mpd=int(pm.b_findpeak_dist/dt), show=False)
    print('Found %d peaks'%len(peaks))

    if peaks.size == 0:
        print('ERROR1: Peak detection failed')
        peak_data = False
        b_pulses_onset = 0
        b_pulses_success = 0

    # If at least one peak was detected:
    if peaks.size > 0:
        # Initialize variables:
        start = np.zeros(peaks.size,dtype=int)
        end = np.zeros(peaks.size,dtype=int)
        duration = np.zeros(peaks.size)
        amplitudes = np.zeros(peaks.size)

        # Get peak amplitudes:
        amplitudes = lowpass_b[peaks]

        # Initialise IEI array:
        IEI = np.zeros(peaks.size-1, dtype=float)

        ''' Find peak start and end points from FWHM '''
        # Iterate through the whole simulation array to find peak start and end points:
        i = 0 # simulation step index
        j = 0 # peak index
        in_peak = False # track if we're "inside" a peak
        halfmax = lowpass_b[peaks[0]]/2 # current peak half-maximum
        while(i < t.size and j < peaks.size):

            # If outside peak; above the half-maximum; before the max:
            if (in_peak == False) and (lowpass_b[i] >= halfmax) and (i < peaks[j]):
                start[j] = i # peak j start point
                in_peak = True # tracks we're inside a peak

            # If inside peak; below the half-maximum; after the max:
            if (in_peak == True) and (lowpass_b[i] <= halfmax) and (i > peaks[j]):
                end[j] = i # peak j end point
                in_peak = False # tracks we're no longer inside a peak

            # If outside peak; more peaks exist; closer to peak j+1 than to j:
            if (in_peak == False) and (j < peaks.size - 1) and (i > peaks[j]) and (peaks[j+1] - i < i - peaks[j]):
                j = j + 1 # look for next peak
                halfmax = lowpass_b[peaks[j]]/2 # update half-maximum

            i = i + 1

        # Get peak durations:
        duration = t[end] - t[start]

        # If simulation ends during last peak, discard it:
        if (end[peaks.size-1] == 0) and (start[peaks.size-1] > 0):
            peaks = peaks[:-1]
            start = start[:-1]
            end = end[:-1]
            duration = duration[:-1]
            amplitudes = amplitudes[:-1]
            IEI = IEI[:-1]

        # Sanity check:
        if (duration <= 0).any():
            print('ERROR2: Failed finding peak start and end points')
            peak_data = False

            for i in range(len(peaks)):
                if duration[i] <= 0:
                    t_peak_err = t[peaks[i]]
                    print('error in peak %d at time %d'%(i,t_peak_err))
                    plt.plot(t,lowpass_b)
                    plt.xlim([t_peak_err - 500, t_peak_err + 500])
                    plt.axvline(t[peaks[i]])
                    plt.axvline(t[peaks[i-1]])
                    if (i + 1) < peaks.size:
                        plt.axvline(t[peaks[i+1]])
                    plt.axvline(t[start[i]], color='black')
                    plt.axvline(t[end[i]], color='red')
                    plt.axhline(lowpass_b[peaks[i]]/2,ls='--')
                    plt.show()


        ''' Calculate Inter-Event-Intervals '''
        if peak_data != False:
            for i in range(peaks.size-1):
                IEI[i] = (t[start[i+1]] - t[end[i]])*1e-3 # convert to seconds

        # Initialise outputs:
        amp_prev = 0
        amp_next = 0
        duration_prev = 0
        duration_next = 0
        IEI_prev = 0
        IEI_next = 0
        b_pulses_onset = 0
        b_pulses_success = 0

        ''' Get data for "spontaneous" events '''
        if (sim_type is 'spont'):
            amp_prev = amplitudes[1:]
            amp_next = amplitudes[:-1]
            duration_prev = duration[1:]
            duration_next = duration[:-1]
            IEI_prev = IEI
            IEI_next = IEI

        ''' Get data for "evoked" events '''
        if (sim_type is 'evoke'):
            # Checks there were current step pulses:
            if b_pulses.any() > 0:
                # Get array with start times of each pulse:
                for i in range(len(b_pulses) - 1):
                    if (b_pulses[i+1] > 0) and (b_pulses[i] == 0):
                        b_pulses_onset = np.append(b_pulses_onset,t[i])
                # Discard initializarion entry:
                b_pulses_onset = b_pulses_onset[1:]

                # Initialise array tracking success of each pulse:
                b_pulses_success = np.zeros_like(b_pulses_onset)
                # Initialise array with indices of successful pulses:
                evoked_peaks = np.array([-1])
                # Iterate through all pulses:
                for i in range(len(b_pulses_onset)):
                    # Iterate through all peaks:
                    for j in range(peaks.size):
                        # If peak height occurs within 50 ms of pulse onset:
                        if (t[peaks[j]] - b_pulses_onset[i]) > 0 and (t[peaks[j]] - b_pulses_onset[i]) <= 50:
                            b_pulses_success[i] = 1 # tracks peak success
                            evoked_peaks = np.append(evoked_peaks,j) # tracks successful peak index

                # If at least one pulse was successful:
                if len(evoked_peaks) > 1:
                    # Discard initialization entry:
                    evoked_peaks = evoked_peaks[1:]
                    # If the last evoked peak is the last peak in the simulation, discards it:
                    if (evoked_peaks[-1] == peaks.size - 1):
                        evoked_peaks = evoked_peaks[:-1]

                    # Get data only from evoked peaks:
                    start = start[evoked_peaks]
                    end = end[evoked_peaks]
                    amp_prev = amplitudes[evoked_peaks]
                    amp_next = amplitudes[evoked_peaks]
                    duration_prev = duration[evoked_peaks]
                    duration_next = duration[evoked_peaks]
                    IEI_prev = IEI[evoked_peaks-1]
                    IEI_next = IEI[evoked_peaks]
                # If no pulses were successful:
                else:
                    evoked_peaks = 0
                    start = 0
                    end = 0
                    amp_prev = 0
                    amp_next = 0
                    duration_prev = 0
                    duration_next = 0
                    IEI_prev = 0
                    IEI_next = 0
            else:
                peak_data = False

        # Store peak_data array:
        peak_data = start, end, amp_prev, amp_next, duration_prev, duration_next, IEI_prev, IEI_next

    ''' Fit Previous IEI vs. Peak Amplitude with exponential function '''
    fit_params = np.array([0,0,0])
    if peak_data != False:
        try: fit_params, _ = curve_fit(fit_func, IEI_prev, amp_prev, p0=(2,2,68), bounds=(0,[100, 100, 100]))
        except: None

    ''' Calculate and print peak data '''
    if peak_data != False:
        print('Minimum IEI is %.1lf ms'%(np.min(IEI_prev)*1000))
        print('Peak incidence = %.2lf Hz'%(len(start)/simtime))
        print('Time constant is %.1lf ms'%(1e3/fit_params[1]))
        if sim_type == 'spont':
            print('IEI = (%.2lf +/- %.2lf) s'%(np.mean(IEI_prev),np.std(IEI_prev)))
        elif sim_type == 'evoke':
            print('IEI = (%.2lf +/- %.2lf) s'%(np.mean(np.concatenate((IEI_prev,IEI_next))),np.std(np.concatenate((IEI_prev,IEI_next)))))
        c, p = pearsonr(IEI_prev,duration_prev)
        print('FWHM correlation with Previous IEI: c = %.3lf, p = %.2e'%(c,p))
        c, p = pearsonr(IEI_next,duration_next)
        print('FWHM correlation with Next IEI: c = %.3lf, p = %.2e'%(c,p))
        c, p = pearsonr(IEI_prev,amp_prev)
        print('Amplitude correlation with Previous IEI: c = %.3lf, p = %.2e'%(c,p))
        c, p = pearsonr(IEI_next,amp_next)
        print('Amplitude correlation with Next IEI: c = %.3lf, p = %.2e'%(c,p))

    return lowpass_b, peak_data, fit_params, b_pulses_onset, b_pulses_success


def plot_one_side(fig, grid, sim_type, tstart, tstop, t, b, e, b_pulses, lowpass_b,\
                    peak_data, fit_params, b_pulses_onset, b_pulses_success):
    ''' Plot one half of Fig. 12
    A) t vs. b and t vs. e
    B) e vs. b phase-plane
    C) Histogram of Inter-Event-Intervals
    D) Previous/Next IEI vs. Peak Amplitude
    '''

    # Unpack peak data:
    if peak_data != False:
        peaks_start, peaks_end, peaks_amp_prev, peaks_amp_next, peaks_duration_prev, peaks_duration_next, peaks_IEI_prev, peaks_IEI_next = peak_data

    # A) subplots:
    ax_t_B = fig.add_subplot(grid[0, 0:4])
    ax_t_e = fig.add_subplot(grid[1:3, 0:4])

    # B) subplot:
    ax_e_B = fig.add_subplot(grid[0, 4:6])

    # C) subplot:
    ax_IEI_hist = fig.add_subplot(grid[2, 4:6])

    # D) subplots:
    ax_prev = fig.add_subplot(grid[4,0:3])
    ax_next = fig.add_subplot(grid[4,3:6])

    # Axes properties:
    for ax in [ax_t_B, ax_t_e]:
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', direction='out', bottom=False,\
                        top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', direction='out', bottom=False,\
                        top=False, labelbottom=False, labelsize=pm.fonts)
        ax.set_xlim([tstart,tstop])
    for ax in [ax_e_B, ax_IEI_hist, ax_prev, ax_next]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', direction='out', bottom=True,\
                        top=False, labelbottom=True, labelsize=pm.fonts)
        ax.tick_params(axis='y', which='both', direction='out', bottom=False,\
                        top=False, labelbottom=False, labelsize=pm.fonts)
    ax_IEI_hist.spines['left'].set_visible(False)

    # B ticks for A) and B):
    B_ticks = [0,45,90]

    ''' A) Subplot '''

    # Subplot titles:
    if sim_type is 'spont':
        ax_t_B.set_title(r'\textbf{A1}',loc='left',x=-0.1,y=0.95,fontsize=pm.fonts)
        ax_t_B.set_title(r'Spontaneous',loc='center',x=0.75,y=1.20,fontsize=pm.fonts)
    else:
        ax_t_B.set_title(r'\textbf{A2}',loc='left',x=-0.1,y=0.95,fontsize=pm.fonts)
        ax_t_B.set_title(r'Evoked',loc='center',x=0.75,y=1.20,fontsize=pm.fonts)

    # Plot t vs. b:
    ax_t_B.plot(t, b, color='#3c3fef', lw=1.5)

    # Time scale bar:
    ax_t_B.axhline(112,xmin=0.12,xmax=(0.12+1/6), linewidth=1, color='black')
    ax_t_B.text(tstart+(tstop-tstart)*0.08,120,'250 ms',fontsize=pm.fonts)

    # Plot markings for evoked events:
    if (sim_type == 'evoke') and (b_pulses.any() > 0) and (peak_data != False):
        # Shade area where pulse is active in yellow:
        ytop, ybottom = ax_t_B.get_ylim()
        ax_t_B.fill_between(t, ybottom, ytop, where = b_pulses > 0, facecolor='#d4b021')

        # Iterate through all pulses:
        for i in range(len(b_pulses_onset)):
            # Only check pulses within plotting range:
            if (b_pulses_onset[i] >= tstart) and (b_pulses_onset[i] <= tstop):
                # If pulse evokes event, mark onset with black arrow:
                if b_pulses_success[i] == 1:
                    ax_t_B.annotate('', xy=(b_pulses_onset[i], 0), xytext=(b_pulses_onset[i], -30),\
                        xycoords='data',arrowprops=dict(arrowstyle="->", lw=1.,color='black'))
                # If pulse fails evoking event, mark onset with gray arrow:
                else:
                    ax_t_B.annotate('', xy=(b_pulses_onset[i], 0), xytext=(b_pulses_onset[i], -30),\
                        xycoords='data',arrowprops=dict(arrowstyle="->", lw=1.,color='lightgray'))

    # Axes limits, ticks, and labels:
    ax_t_B.set_ylim([-0.1*B_ticks[-1],1.3*B_ticks[-1]])
    if sim_type is 'spont': ax_t_B.set_ylabel("B [1/s]",fontsize=pm.fonts)
    ax_t_B.set_yticks(B_ticks)
    ax_t_B.set_yticklabels(B_ticks,fontsize=pm.fonts)

    ax_t_e.plot(t, e, color='#e67e22', lw=1.5)
    ax_t_e.set_yticks([0.5,1.0])
    ax_t_e.set_yticklabels([0.5,1.0],fontsize=pm.fonts)
    ax_t_e.set_ylim([0.3,1.1])
    if sim_type is 'spont': ax_t_e.set_ylabel("e",fontsize=pm.fonts)

    ''' B) Subplot '''

    # Subplot titles:
    if sim_type is 'spont':
        ax_e_B.set_title(r'\textbf{B1}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)
    else:
        ax_e_B.set_title(r'\textbf{B2}',loc='left',x=-0.22,y=0.95,fontsize=pm.fonts)

    # Load e vs. b bifurcation diagram:
    bif_path = os.path.dirname( __file__ ) + '/../../bifurcation_analysis/bifurcation_diagrams/1param/'
    bs = bif.load_bifurcations(bif_path, 'e', 0, 1)

    # Plot bifurcation diagram:
    bif.plot_bifurcation(ax_e_B,aux,bs,'B',[0.25,1],1,'',[0.4,0.8],[0.4,0.8],\
                        B_ticks,B_ticks,pm.fonts,plot_color='gray',line_width=1.,inward_ticks=False)
    # Plot e vs. b trace:
    ax_e_B.plot(e[(t >= tstart) & (t <= tstop)], b[(t >= tstart) & (t <= tstop)], color='#3c3fef', lw=1.5)

    # Axes limits, ticks, and labels:
    ax_e_B.set_ylabel('',fontsize=pm.fonts)
    ax_e_B.set_ylim([-0.1*B_ticks[-1],1.2*B_ticks[-1]])
    ax_e_B.set_title('e', y=0.95, fontsize=pm.fonts)

    # If peak detection was successful:
    if peak_data != False:

        ''' C) Subplot '''

        # "Spontaneous" simulation:
        if sim_type is 'spont':
            # Subplot title:
            ax_IEI_hist.set_title(r'\textbf{C1}',loc='left',x=-0.22,y=0.80,fontsize=pm.fonts)
            # Histogram of previous IEIs
            # (for spontaneous simulation, previous and next IEIs are the same arrays):
            hist_n, _, _ = ax_IEI_hist.hist(peaks_IEI_prev,bins=30,color='gray')
        # "Evoked" simulation:
        else:
            # Subplot title:
            ax_IEI_hist.set_title(r'\textbf{C2}',loc='left',x=-0.22,y=0.80,fontsize=pm.fonts)
            # Histogram of both previous and next (w.r.t. evoked events) IEIs
            hist_n, _, _ = ax_IEI_hist.hist(np.concatenate((peaks_IEI_prev,peaks_IEI_next)),bins=30,color='gray')

        # Axes limits, ticks, and labels:
        ax_IEI_hist.set_title('IEI [s]', y=0.80, fontsize=pm.fonts)
        ax_IEI_hist.set_xlim([0,3.6])
        ax_IEI_hist.set_xticks([0,1,2,3])
        ax_IEI_hist.set_xticklabels([0,1,2,3],fontsize=pm.fonts)
        ax_IEI_hist.set_yticks([])
        ax_IEI_hist.set_yticklabels([])

        ''' D) Subplot '''

        # Subplot titles:
        if sim_type is 'spont':
            ax_prev.set_title(r'\textbf{D1}',loc='left',x=-0.15,y=0.95,fontsize=pm.fonts)
        else:
            ax_prev.set_title(r'\textbf{D2}',loc='left',x=-0.15,y=0.95,fontsize=pm.fonts)

        ''' D) Previous IEI '''

        # Plot Previous IEI vs. Peak Amplitude:
        ax_prev.plot(peaks_IEI_prev, peaks_amp_prev, 'k.', ms=3)

        # Mark minimum IEI with vertical line:
        ax_prev.axvline(np.min(peaks_IEI_prev), linewidth=1, color='k', linestyle='--')

        # If exponential fit was successful, plot it:
        if fit_params.all != 0:
            x_array = np.arange(np.min(peaks_IEI_prev),np.max(peaks_IEI_prev),0.01)
            ax_prev.plot(x_array,fit_func(x_array,*fit_params),color='red', lw=1.5)

        # Axes limits, ticks, and labels:
        ax_prev.set_xlabel('Previous IEI [s]',fontsize=pm.fonts)
        if sim_type is 'spont': ax_prev.set_ylabel('Amplitude [1/s]',fontsize=pm.fonts)
        ax_prev.set_ylim([55,105])
        ax_prev.set_yticks([75,100])
        ax_prev.set_yticklabels([75,100],fontsize=pm.fonts)
        ax_prev.set_xlim([0,3.6])
        ax_prev.set_xticks([0,1,2,3])
        ax_prev.set_xticklabels([0,1,2,3],fontsize=pm.fonts)

        ''' D) Next IEI '''

        # Plot Next IEI vs. Peak Amplitude:
        ax_next.plot(peaks_IEI_next, peaks_amp_next, 'k.', ms=3)

        # Axes limits, ticks, and labels:
        ax_next.set_xlabel('Next IEI [s]',fontsize=pm.fonts)
        ax_next.set_ylim([55,105])
        ax_next.set_yticks([75,100])
        ax_next.set_yticklabels([75,100],fontsize=pm.fonts)
        ax_next.set_xlim([0,3.6])
        ax_next.set_xticks([0,1,2,3])
        ax_next.set_xticklabels([0,1,2,3],fontsize=pm.fonts)

def plot_fig_12():
    ''' Plot Fig. 12 '''

    # Load data from "spontaneous" events simulation:
    print('============Spontaneous Events============')
    data_spont = np.load('results/noisy_rate_model_long_sim_spont.npz', encoding='latin1', allow_pickle=True)
    dict_spont = dict(zip(("{}".format(k) for k in data_spont), (data_spont[k] for k in data_spont)))
    t_spont = dict_spont['t']
    b_spont = dict_spont['b']
    e_spont = dict_spont['e']
    b_pulses_spont = dict_spont['b_pulses']
    # Get peak data for "spontaneous" events:
    lowpass_b_spont, peak_data_spont, fit_data_spont, _, _ = get_peak_data(t_spont, b_spont, b_pulses_spont, 'spont')

    # Load data from "evoked" events simulation:
    print('==============Evoked Events===============')
    data_evoke = np.load('results/noisy_rate_model_long_sim_evoke.npz', encoding='latin1', allow_pickle=True)
    dict_evoke = dict(zip(("{}".format(k) for k in data_evoke), (data_evoke[k] for k in data_evoke)))
    t_evoke = dict_evoke['t']
    b_evoke = dict_evoke['b']
    e_evoke = dict_evoke['e']
    b_pulses_evoke = dict_evoke['b_pulses']
    # Get peak data for "evoked" events:
    lowpass_b_evoke, peak_data_evoke, fit_data_evoke, b_pulses_onset, b_pulses_success = get_peak_data(t_evoke, b_evoke, b_pulses_evoke, 'evoke')

    # Create Fig. 12:
    fig_width = 17.6/2.54
    fig_height = 0.4*17.6/2.54
    fig = plt.figure(figsize=(fig_width,fig_height))
    gs = gridspec.GridSpec(8, 13, width_ratios=[1,1,1,1,1,1,0.2,1,1,1,1,1,1])
    gs_spont = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs[:, 0:6], height_ratios=[1.,0.35,0.45,0.25,1.])
    gs_evoke = gridspec.GridSpecFromSubplotSpec(5, 6, subplot_spec=gs[:, 7:13], height_ratios=[1.,0.35,0.45,0.25,1.])

    # A), B), C) subplots time limits:
    t_plot_start = 195.4*1e3
    t_plot_stop = 196.9*1e3

    # Subplots for "spontaneous" events:
    plot_one_side(fig, gs_spont, 'spont', t_plot_start, t_plot_stop,\
                    t_spont, b_spont, e_spont, b_pulses_spont, lowpass_b_spont,\
                     peak_data_spont, fit_data_spont, None, None)

    # Subplots for "evoked" events:
    plot_one_side(fig, gs_evoke, 'evoke', t_plot_start, t_plot_stop,\
                    t_evoke, b_evoke, e_evoke, b_pulses_evoke, lowpass_b_evoke,\
                     peak_data_evoke, fit_data_evoke, b_pulses_onset, b_pulses_success)

    # Adjust distance between subplots:
    plt.subplots_adjust(wspace=2.0, hspace=0.25)

    # Export Fig. 12:
    fig.savefig('results/fig_rate_model_noise.eps', bbox_inches='tight', dpi=800)
