import numpy as np
import math as m
import scipy.integrate as integrate

import datetime

import helper_functions.params_noisy_rate as pm
from helper_functions.utils_noisy_rate import get_peak_data

# Square current pulses:
def u_square(arr,t):
    y = np.zeros_like(t)
    for i in range(len(arr)):
        y[(t <= arr[i,0]+arr[i,1]) & (t >= arr[i,0])] = arr[i,2]
    return y

# Calculate poisson noise with exponential kernel:
def calc_poisson_noise(t,arr,tausyn):
    poisson_freq, poisson_amp = arr

    y = np.zeros_like(t)

    if (poisson_freq != 0) and (poisson_amp != 0):
        dt = t[1] - t[0] # ms
        sim_time = t[-1] # ms

        # Select random spike times:
        total_spikes = int(poisson_freq*sim_time/1e3)
        spike_indices = np.random.choice(len(y),total_spikes,replace=True)

        # Sum each spike as unitary pulse
        for si in spike_indices:
            y[si] += 1

        # Calculate exponential decay kernel (until 0.1% of height)
        time_spike_end = -tausyn*np.log(0.001)
        arg_spike_end = np.argmin(np.abs(t - time_spike_end))
        spike_kernel = np.exp(-t[:arg_spike_end]/tausyn)

        # Convolve spike train with exponential kernel
        y = np.convolve(y,spike_kernel,mode='same')

        # Multiply by amplitude of single spike:
        y = y*poisson_amp

    return y

# Periodic square inputs:
def u_periodic_square(arr,t):
    square_per, square_dur, square_amp = arr

    y = np.zeros_like(t)

    if (square_per != 0) and (square_amp != 0) and (square_dur !=0):
        dt = (t[1] - t[0])

        # Get periodic times at which each stimulation starts:
        stim_times = np.arange(square_per,t[-1]/1e3,square_per)*1e3

        # Get random delay of [0,90] ms for each stimulation
        jitters = np.random.randint(91,size=len(stim_times))

        # Apply random delays to stimulation times:
        stim_times = stim_times + jitters

        # Apply stimulations:
        for stim in stim_times:
            y[(t >= stim) & (t <= (stim + square_dur))] = square_amp

    return y

# Soft-plus function:
def spf(x,k,t):
    # try:
    y = m.log(1+m.exp(k*(x+t)))
    # except:
        # y = float('inf')
    return y

# Model equations:
def dp(t,dt,p,*args):
    #show progres:
    # if (round(t,4) % 100) == 0:
        # print('t = %f'%t)
    b, a, e, input_p = args
    if pm.d_pb_on == 1:
        e_pb = e
    else:
        e_pb = 1

    x_p = pm.w_pp*p-pm.w_pb*b*e_pb-pm.w_pa*a+input_p[int(t/dt)]
    y = (-p+spf(x_p,pm.k_p,pm.t_p))/pm.tau_p
    return y

def db(t,dt,b,*args):
    p, a, e, input_b = args
    x_b = pm.w_bp*p-pm.w_bb*b-pm.w_ba*a+input_b[int(t/dt)]
    y = (-b+spf(x_b,pm.k_b,pm.t_b))/pm.tau_b
    return y

def da(t,dt,a,*args):
    p, b, e, z, input_a = args
    if pm.d_ab_on == 1:
        e_ab = e
    else:
        e_ab = 0.5
    x_a = pm.w_ap*p*(1+z)-pm.w_ab*b*e_ab-pm.w_aa*a+input_a[int(t/dt)]
    y = (-a+spf(x_a,pm.k_a,pm.t_a))/pm.tau_a
    return y

def de(t,dt,e,*args):
    p, b, a = args
    if (pm.d_ab_on == 1) or (pm.d_pb_on == 1):
        y = ((1-e)/pm.tau_d) - pm.eta_d*b*e # depression
    else:
        y = 0.0
    return y

def dz(t,dt,z,*args):
    p, b, a = args
    if pm.f_ap_on == 1:
        y = (-z/pm.tau_f)+pm.eta_f*p*(pm.z_max-z) # depression
    else:
        y = 0.0
    return y


def get_noises(time, noise_args):

    noise_params_PI, noise_params_BI, noise_params_AI = noise_args
    noise_params_PP, noise_params_PB, noise_params_PA = noise_params_PI
    noise_params_BP, noise_params_BB, noise_params_BA = noise_params_BI
    noise_params_AP, noise_params_AB, noise_params_AA = noise_params_AI

    current_PP = calc_poisson_noise(time,noise_params_PP,pm.tausyn_P)
    current_PB = calc_poisson_noise(time,noise_params_PB,pm.tausyn_B)
    current_PA = calc_poisson_noise(time,noise_params_PA,pm.tausyn_A)
    noise_to_p = (current_PP + current_PB + current_PA)\
                        - np.mean(current_PP + current_PB + current_PA)

    current_BP = calc_poisson_noise(time,noise_params_BP,pm.tausyn_P)
    current_BB = calc_poisson_noise(time,noise_params_BB,pm.tausyn_B)
    current_BA = calc_poisson_noise(time,noise_params_BA,pm.tausyn_A)
    noise_to_b = (current_BP + current_BB + current_BA)\
                        - np.mean(current_BP + current_BB + current_BA)

    current_AP = calc_poisson_noise(time,noise_params_AP,pm.tausyn_P)
    current_AB = calc_poisson_noise(time,noise_params_AB,pm.tausyn_B)
    current_AA = calc_poisson_noise(time,noise_params_AA,pm.tausyn_A)
    noise_to_a = (current_AP + current_AB + current_AA)\
                        - np.mean(current_AP + current_AB + current_AA)

    noises = noise_to_p, noise_to_b, noise_to_a

    # save individual currents for plotting
    noise_currents = current_PP, current_BP, current_AP, \
                     current_PB, current_BB, current_AB, \
                     current_PA, current_BA, current_AA

    return noises, noise_currents

def derivs(t, dt, state, *args):
    """
    Map the state variable [p,b,a,e,z] to the derivitives
    [dp,db,da,de,dz] at time t
    """
    input_p, input_b, input_a = args
    p, b, a, e, z = state  # all populations

    deltap = dp(t, dt, p, b, a, e, input_p)  # change in p
    deltab = db(t, dt, b, p, a, e, input_b) # change in b
    deltaa = da(t, dt, a, p, b, e, z, input_a) # change in a
    deltae = de(t, dt, e, p, b, a) # change in e
    deltaf = dz(t, dt, z, p, b, a) # change in z

    return deltap, deltab, deltaa, deltae, deltaf

def solve_model(time, init0, noise_args, pulse_args):
    noises, noise_currents = get_noises(time, noise_args)
    noise_p, noise_b, noise_a = noises

    b_pulses = u_periodic_square(pulse_args,time)

    input_p = noise_p
    input_b = noise_b + b_pulses
    input_a = noise_a

    dt = (time[1] - time[0])

    t = np.zeros_like(time)
    p = np.zeros_like(time)
    b = np.zeros_like(time)
    a = np.zeros_like(time)
    e = np.zeros_like(time)
    z = np.zeros_like(time)

    max_interval = 9999.99 # 10 seconds
    i_start = int(0/dt)
    i_stop = int(max_interval/dt)
    i_interval = i_stop - i_start

    if i_stop >= len(time): i_stop = len(time) - 1

    init = init0

    start_time = datetime.datetime.now()
    print("Starting simulation...")
    while 1:
        tspan = time[i_start:i_stop+1]

        out = integrate.solve_ivp(fun=lambda t, y: derivs(t, dt, y, input_p, input_b, input_a),\
                                    t_span=[tspan[0],tspan[-1]], y0=init, t_eval=tspan, first_step=dt/10, max_step=dt)

        t[i_start:i_stop+1] = out.t
        p[i_start:i_stop+1] = out.y[0]
        b[i_start:i_stop+1] = out.y[1]
        a[i_start:i_stop+1] = out.y[2]
        e[i_start:i_stop+1] = out.y[3]
        z[i_start:i_stop+1] = out.y[4]

        init = [p[i_stop-1],b[i_stop-1],a[i_stop-1],e[i_stop-1],z[i_stop-1]]

        time_since = datetime.datetime.now() - start_time
        mins_since = int(time_since.seconds/60)
        secs_since = int(time_since.seconds - mins_since*60)

        time_left = (time_since*time[-1]/time[i_stop]) - time_since
        mins_left = int(time_left.seconds/60)
        secs_left = int(time_left.seconds - mins_left*60)

        print('Calculated %.1lfs (of %.1lfs) in %dm:%02ds ... (ca. %dm:%02ds left)'
                        %(time[i_stop]/1e3,time[-1]/1e3, mins_since, secs_since, mins_left, secs_left))

        i_start = i_stop + 1
        if i_start >= len(time) - 1:
            break
        i_stop = i_start + i_interval
        if i_stop >= len(time): i_stop = len(time) - 1


    return t, p, b, a, e, z, noises, noise_currents, b_pulses

def run_fig_12():
    pm.set_default_parameters()

    noise_args = (pm.poisson_P, pm.poisson_B, pm.poisson_A)
    pulse_args = (pm.B_pulses_per, pm.B_pulses_dur, pm.B_pulses_amp)

    t = np.arange(0.0, 10*60*1e3, pm.sim_dt)

    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    print('running network ...')

    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,noise_args,(0,0,0))
    t, p, b, a, e, z, noises, noise_currents, b_pulses = sim_out

    dic_to_save = {'t': t[::pm.compress_step],
                   'p': p[::pm.compress_step],
                   'b': b[::pm.compress_step],
                   'a': a[::pm.compress_step],
                   'e': e[::pm.compress_step],
                   'z': z[::pm.compress_step],
                   'noises': noises[::pm.compress_step],
                   'noise_currents': noise_currents[::pm.compress_step],
                   'b_pulses': b_pulses[::pm.compress_step]
                   }

    np.savez_compressed('results/noisy_rate_model_long_sim_spont', **dic_to_save)

    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,noise_args,pulse_args)
    t, p, b, a, e, z, noises, noise_currents, b_pulses = sim_out

    dic_to_save = {'t': t[::pm.compress_step],
                   'p': p[::pm.compress_step],
                   'b': b[::pm.compress_step],
                   'a': a[::pm.compress_step],
                   'e': e[::pm.compress_step],
                   'z': z[::pm.compress_step],
                   'noises': noises[::pm.compress_step],
                   'noise_currents': noise_currents[::pm.compress_step],
                   'b_pulses': b_pulses[::pm.compress_step]
                   }

    np.savez_compressed('results/noisy_rate_model_long_sim_evoke', **dic_to_save)

def run_fig_13():

    # Default plasticity:
    pm.set_default_parameters()
    noise_args = (pm.poisson_P, pm.poisson_B, pm.poisson_A)

    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    print('Run default network ...')
    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,noise_args,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_default', **dic_to_save)

    # Extra depression:
    pm.set_default_parameters('extra_dpr')
    noise_args = (pm.poisson_P, pm.poisson_B, pm.poisson_A)

    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    print('Run extra depression network ...')
    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,noise_args,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_extra_dpr', **dic_to_save)

def run_fig_15():

    # Facilitation only:
    pm.set_default_parameters('facil_only')
    noise_args = (pm.poisson_P, pm.poisson_B, pm.poisson_A)

    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    print('Run facilitation only network ...')
    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,noise_args,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_facil', **dic_to_save)
