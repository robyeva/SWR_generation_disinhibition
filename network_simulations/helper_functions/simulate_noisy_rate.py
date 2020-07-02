'''
    Rate model equations, input (noisy and square pulse) generation,
    integration of noisy rate model, and simulations for
    Fig. 12, Fig. 13E, and Fig. 15D
'''

import numpy as np
import math as m
import scipy.integrate as integrate

import datetime

import helper_functions.params_noisy_rate as pm

# Soft-plus function:
def spf(x,k,t):
    y = m.log(1+m.exp(k*(x+t)))
    return y

''' Rate model equations '''

def dp(t,dt,p,*args):
    b, a, e, input_p = args

    # B->P depression:
    if pm.d_pb_on == 1: e_pb = e
    else: e_pb = 1

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

    # B->A depression:
    if pm.d_ab_on == 1: e_ab = e
    else: e_ab = 0.5

    x_a = pm.w_ap*p*(1+z)-pm.w_ab*b*e_ab-pm.w_aa*a+input_a[int(t/dt)]
    y = (-a+spf(x_a,pm.k_a,pm.t_a))/pm.tau_a

    return y

def de(t,dt,e,*args):
    p, b, a = args

    # If B->A or B->P depression is on:
    if (pm.d_ab_on == 1) or (pm.d_pb_on == 1):
         # (time is simulated in ms, so b firing rate must be scaled by 1e-3)
        y = ((1-e)/pm.tau_d) - pm.eta_d*b*1e-3*e

    # If no depression is on:
    else: y = 0

    return y

def dz(t,dt,z,*args):
    p, b, a = args

    # If P->A facilitation is on:
    if pm.f_ap_on == 1:
        # (time is simulated in ms, so p firing rate must be scaled by 1e-3_
        y = (-z/pm.tau_f)+pm.eta_f*p*1e-3*(pm.z_max-z)

    # If no facilitation in on:
    else: y = 0

    return y

''' Rate model noise '''

# Calculate Poisson noise with exponential kernel:
def calc_poisson_noise(t,arr,tausyn):
    poisson_freq, poisson_amp = arr

    # Initialise noisy current array:
    y = np.zeros_like(t)

    # If Poisson process frequency and amplitude != 0:
    if (poisson_freq != 0) and (poisson_amp != 0):

        # Simulation dt (ms):
        dt = t[1] - t[0]

        # Total simulation time (ms):
        sim_time = t[-1] - t[0]

        # Select random spike times:
        total_spikes = int(poisson_freq*sim_time/1e3)
        spike_indices = np.random.choice(len(y),total_spikes,replace=True)

        # Sum each spike as a point with amplitude 1:
        for si in spike_indices:
            y[si] += 1

        # Calculate exponential decay kernel (until 0.1% of height):
        time_spike_end = -tausyn*np.log(0.001)
        arg_spike_end = np.argmin(np.abs(t - time_spike_end))
        spike_kernel = np.exp(-t[:arg_spike_end]/tausyn)

        # Convolve spike train with exponential kernel:
        y = np.convolve(y,spike_kernel,mode='same')

        # Multiply by amplitude of single spike:
        y = y*poisson_amp

    return y

# Get nine inpedendent rate model noisy currents:
def get_noises(time):

    current_PP = calc_poisson_noise(time,pm.noise_params_PP,pm.tausyn_P)
    current_PB = calc_poisson_noise(time,pm.noise_params_PB,pm.tausyn_B)
    current_PA = calc_poisson_noise(time,pm.noise_params_PA,pm.tausyn_A)
    # Subtract mean to inject noise to P:
    noise_to_p = (current_PP + current_PB + current_PA)\
                        - np.mean(current_PP + current_PB + current_PA)

    current_BP = calc_poisson_noise(time,pm.noise_params_BP,pm.tausyn_P)
    current_BB = calc_poisson_noise(time,pm.noise_params_BB,pm.tausyn_B)
    current_BA = calc_poisson_noise(time,pm.noise_params_BA,pm.tausyn_A)
    # Subtract mean to inject noise to B:
    noise_to_b = (current_BP + current_BB + current_BA)\
                        - np.mean(current_BP + current_BB + current_BA)

    current_AP = calc_poisson_noise(time,pm.noise_params_AP,pm.tausyn_P)
    current_AB = calc_poisson_noise(time,pm.noise_params_AB,pm.tausyn_B)
    current_AA = calc_poisson_noise(time,pm.noise_params_AA,pm.tausyn_A)
    # Subtract mean to inject noise to A:
    noise_to_a = (current_AP + current_AB + current_AA)\
                        - np.mean(current_AP + current_AB + current_AA)

    noises = noise_to_p, noise_to_b, noise_to_a

    # Save individual currents for plotting:
    noise_currents = current_PP, current_BP, current_AP, \
                     current_PB, current_BB, current_AB, \
                     current_PA, current_BA, current_AA

    return noises, noise_currents

''' Square pulses to evoke events '''

def u_periodic_square(arr,t):
    square_per, square_dur, square_amp = arr

    # Initialise current array:
    y = np.zeros_like(t)

    # If any of the parameters != 0:
    if (square_per != 0) and (square_amp != 0) and (square_dur !=0):

        # Simulation dt (ms):
        dt = t[1] - t[0]

        # Total simulation time (secs):
        sim_time = (t[-1] - t[0])/1e3

        # Get random periodic times at which each stimulation starts:
        stim_times = np.arange(square_per,sim_time,square_per)*1e3

        # Get random delay of [0,90] ms for each stimulation:
        jitters = np.random.randint(91,size=len(stim_times))

        # Apply random delays to stimulation times:
        stim_times = stim_times + jitters

        # Apply stimulations:
        for stim in stim_times:
            y[(t >= stim) & (t <= (stim + square_dur))] = square_amp

    return y

''' Model integration '''

def derivs(t, dt, state, *args):
    '''
    Map the state variable [p,b,a,e,z] to the derivitives
    [dp,db,da,de,dz] at time t
    '''
    input_p, input_b, input_a = args
    p, b, a, e, z = state

    deltap = dp(t, dt, p, b, a, e, input_p)
    deltab = db(t, dt, b, p, a, e, input_b)
    deltaa = da(t, dt, a, p, b, e, z, input_a)
    deltae = de(t, dt, e, p, b, a)
    deltaf = dz(t, dt, z, p, b, a)

    return deltap, deltab, deltaa, deltae, deltaf

def solve_model(time, init0, pulse_args):
    ''' Solves rate model in intervals of 10 seconds

    :param time:
        Simulation time array
    :param init0:
        Initial conditions
    :param pulse_args:
        Arguments to generate random square current pulses:

    :returns t:
        Simulation variable t
    :returns p:
        Simulation variable p
    :returns b:
        Simulation variable b
    :returns a:
        Simulation variable a
    :returns e:
        Simulation variable e
    :returns z:
        Simulation variable z
    :returns noises:
        Noisy currents injected to each population
    :returns noise_currents:
        Individual nine noisy currents
    :returns b_pulses:
        Additional current injected to b population, with square current pulses

    '''

    # Calculate rate model noise:
    noises, noise_currents = get_noises(time)
    noise_p, noise_b, noise_a = noises

    # Calculate random current pulses to add to b:
    b_pulses = u_periodic_square(pulse_args,time)

    # Population inputs:
    input_p = noise_p
    input_b = noise_b + b_pulses
    input_a = noise_a

    # Simulation dt (ms):
    dt = time[1] - time[0]

    # Initialise simulation variables:
    t = np.zeros_like(time)
    p = np.zeros_like(time)
    b = np.zeros_like(time)
    a = np.zeros_like(time)
    e = np.zeros_like(time)
    z = np.zeros_like(time)

    # Divide simulation time in 10-sec intervals:
    max_interval = 9999.99 # 10 seconds
    i_start = int(0/dt)
    i_stop = int(max_interval/dt)
    i_interval = i_stop - i_start
    if i_stop >= len(time): i_stop = len(time) - 1

    # Get initial conditions for first integration interval:
    init = init0

    # Get current time:
    start_time = datetime.datetime.now()

    print("Starting simulation...")

    # Integrate 10-sec intervals until end of simulation time is reached:
    while 1:
        # Time span of current interval:
        tspan = time[i_start:i_stop+1]

        # Integrate for current interval
        out = integrate.solve_ivp(fun=lambda t, y: derivs(t, dt, y, input_p, input_b, input_a),\
                                    t_span=[tspan[0],tspan[-1]], y0=init, t_eval=tspan, first_step=dt/10, max_step=dt)

        # Store results of integration for current interval
        t[i_start:i_stop+1] = out.t
        p[i_start:i_stop+1] = out.y[0]
        b[i_start:i_stop+1] = out.y[1]
        a[i_start:i_stop+1] = out.y[2]
        e[i_start:i_stop+1] = out.y[3]
        z[i_start:i_stop+1] = out.y[4]

        # Store end point as initial conditions for next integration interval
        init = [p[i_stop-1],b[i_stop-1],a[i_stop-1],e[i_stop-1],z[i_stop-1]]

        # Estimate how much time is left until simulation ends:
        time_since = datetime.datetime.now() - start_time
        mins_since = int(time_since.seconds/60)
        secs_since = int(time_since.seconds - mins_since*60)
        time_left = (time_since*time[-1]/time[i_stop]) - time_since
        mins_left = int(time_left.seconds/60)
        secs_left = int(time_left.seconds - mins_left*60)

        if (mins_left+secs_left == 0):
            print('Calculated %.1lfs (of %.1lfs) in %dm:%02ds . (DONE)'\
                %(time[i_stop]/1e3,time[-1]/1e3, mins_since, secs_since))
        else:
            print('Calculated %.1lfs (of %.1lfs) in %dm:%02ds ... (ca. %dm:%02ds left)'\
                %(time[i_stop]/1e3,time[-1]/1e3, mins_since, secs_since, mins_left, secs_left))

        # Move to next integration interval:
        i_start = i_stop + 1
        if i_start >= len(time) - 1:
            break
        i_stop = i_start + i_interval
        if i_stop >= len(time): i_stop = len(time) - 1


    return t, p, b, a, e, z, noises, noise_currents, b_pulses

''' Long simulations for Fig. 12 '''
def run_long_noisy_rate_spont_and_evoked():

    # Set model parameters:
    pm.set_parameters()

    # Simulation time array:
    t = np.arange(0.0, 10*60*1e3, pm.sim_dt)

    ''' Simulation for Spontaneous events '''

    # Initial conditions:
    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    # Initialise random seed:
    np.random.seed(pm.seed)

    # Simulate noisy rate model:
    print('\n')
    print('=====Long noisy rate model simulation with Spontaneous SWRs=====')
    sim_out = solve_model(t,y0,(0,0,0))
    t, p, b, a, e, z, noises, noise_currents, b_pulses = sim_out

    # Save simulation results:
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

    ''' Simulation for Evoked events '''

    # Initialise random seed:
    np.random.seed(pm.seed)

    # Parameters to generate random current square pulses (for evoked events):
    pulse_args = (pm.B_pulses_per, pm.B_pulses_dur, pm.B_pulses_amp)

    # Simulate noisy rate model:
    print('\n')
    print('=====Long noisy rate model simulation with Spontaneous and Evoked SWRs=====')
    sim_out = solve_model(t,y0,pulse_args)
    t, p, b, a, e, z, noises, noise_currents, b_pulses = sim_out

    # Save simulation results:
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

''' Short simulations for Fig. 13E '''
def run_short_noisy_rate_extra_dpr():

    ''' Default model (B->A depression only) '''

    # Set model parameters:
    pm.set_parameters()

    # Simulation time array:
    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    # Initial conditions:
    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    # Initialise random seed:
    np.random.seed(pm.seed)

    # Simulate noisy rate model:
    print('\n')
    print('=====Short noisy rate model simulation (B->A depression)=====')
    sim_out = solve_model(t,y0,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out

    # Save simulation resutls:
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_default', **dic_to_save)

    ''' Model with extra depression (B->A and B->P depression) '''

    # Set model parameters:
    pm.set_parameters('extra_dpr')

    # Simulation time array:
    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    # Initial conditions:
    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    # Initialise random seed:
    np.random.seed(pm.seed)

    # Simulate noisy rate model:
    print('\n')
    print('=====Short noisy rate model simulation (B->A and B->P depression)=====')
    sim_out = solve_model(t,y0,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out

    # Save simulation results:
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_extra_dpr', **dic_to_save)

''' Short simulation for Fig. 15D '''
def run_short_noisy_rate_facil_only():

    # Set model parameters:
    pm.set_parameters('facil_only')

    # Simulation time array:
    t = np.arange(0.0, 10*1e3, pm.sim_dt)

    # Initial conditions:
    y0 = [pm.P_0, pm.B_0, pm.A_0, pm.e_0, pm.z_0]

    # Simulate noisy rate model:
    print('\n')
    print('=====Short noisy rate model simulation (P->A facilitation only)=====')
    np.random.seed(pm.seed)
    sim_out = solve_model(t,y0,(0,0,0))
    t, p, b, a, e, z, _, _, _ = sim_out

    # Save simulation results:
    dic_to_save = {'t': t,
       'p': p,
       'b': b,
       'a': a,
       'e': e,
       'z': z
       }
    np.savez_compressed('results/noisy_rate_model_short_sim_facil', **dic_to_save)
