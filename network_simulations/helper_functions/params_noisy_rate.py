'''
Parameters used to simualte noisy rate model and create Fig. 12, Fig. 13E, and Fig. 15D
'''

# Import python libraries:
import numpy as np

# Plotting font size:
fonts = 9

# Compression step for export:
compress_step = 10

# Random seed:
seed = 123

# Simulation dt:
sim_dt = 0.1 # ms

# Initial conditions:
P_0 = 0.
B_0 = 0.
A_0 = 12.5
e_0 = 0.8
z_0 = 0.

# Spiking model parameters:
p_PP = 0.01; p_BP = 0.20; p_AP = 0.01
p_PB = 0.50; p_BB = 0.20; p_AB = 0.20
p_PA = 0.60; p_BA = 0.60; p_AA = 0.60
N_P = 8200; N_B = 135; N_A = 50
g_PP = 0.20; g_BP = 0.05; g_AP = 0.20
g_PB = 0.70; g_BB = 5.00; g_AB = 8.00
g_PA = 6.00; g_BA = 7.00; g_AA = 4.00
E_rev_P = 0; E_rev_B = -70; E_rev_A = -70;
V_P = -52.5; V_B = -54.0; V_A = -52.5;
tausyn_P = 2; tausyn_B = 1.5; tausyn_A = 4

# Synaptic weights:
def get_w_IJ(N_J, p_IJ, g_IG, V_I, tausyn_J, E_rev_J):
    if E_rev_J < 0:
        sign = -1
    else:
        sign = 1
    return sign*round(N_J*p_IJ*g_IG*tausyn_J*(E_rev_J - V_I)*1e-3,2)

w_pp = get_w_IJ(N_P, p_PP, g_PP, V_P, tausyn_P, E_rev_P);
w_bp = get_w_IJ(N_P, p_BP, g_BP, V_B, tausyn_P, E_rev_P);
w_ap = get_w_IJ(N_P, p_AP, g_AP, V_A, tausyn_P, E_rev_P);
w_pb = get_w_IJ(N_B, p_PB, g_PB, V_P, tausyn_B, E_rev_B);
w_bb = get_w_IJ(N_B, p_BB, g_BB, V_B, tausyn_B, E_rev_B);
w_ab = get_w_IJ(N_B, p_AB, g_AB, V_A, tausyn_B, E_rev_B);
w_pa = get_w_IJ(N_A, p_PA, g_PA, V_P, tausyn_A, E_rev_A);
w_ba = get_w_IJ(N_A, p_BA, g_BA, V_B, tausyn_A, E_rev_A);
w_aa = get_w_IJ(N_A, p_AA, g_AA, V_A, tausyn_A, E_rev_A);

# Activation function parameters:
k_p = 0.47; k_b = 0.41; k_a = 0.48
t_p = 131.660; t_b = 131.960; t_a = 131.090

# Time consants (ms):
tau_p = 3; tau_b = 2; tau_a = 6;

# Synaptic depression:
tau_d = 250; # (ms)
eta_d = 0.18;

# Synaptic facilitation:
tau_f = 230; # (ms)
eta_f = 0.32;
z_max = 1;

# Periodic square inputs for evoked events:
B_pulses_per = 2 # secs
B_pulses_dur = 10 # ms
B_pulses_amp = 150 # pA

# Poisson noise parameters for "spontaneous" events:
def get_noise_params(J_freq, p_IJ, N_J, g_IJ, E_rev_J, V_I, noise_amp):
    poisson_freq = J_freq*p_IJ*N_J
    current_update = noise_amp*g_IJ*(E_rev_J - V_I)
    return np.array([poisson_freq, current_update])
noise_params_PP = 0; noise_params_BP = 0; noise_params_AP = 0;
noise_params_PB = 0; noise_params_BB = 0; noise_params_AB = 0;
noise_params_PA = 0; noise_params_BA = 0; noise_params_AA = 0;

# Short-term plasticity mechanisms:
d_ab_on = 0; d_pb_on = 0; f_ap_on = 0;

# Plasticity-dependent parameters:
def set_parameters(plasticity_type='default'):

    # Synaptic plasticity:
    global d_ab_on
    global d_pb_on
    global f_ap_on

    # Poisson noise parameters:
    global noise_params_PP
    global noise_params_BP
    global noise_params_AP
    global noise_params_PB
    global noise_params_BB
    global noise_params_AB
    global noise_params_PA
    global noise_params_BA
    global noise_params_AA

    if (plasticity_type == 'default') or (plasticity_type == 'extra_dpr'):
        noise_amp = 1/8
        if plasticity_type == 'default':
            d_ab_on = 1; d_pb_on = 0; f_ap_on = 0;
        elif plasticity_type == 'extra_dpr':
            d_ab_on = 1; d_pb_on = 1; f_ap_on = 0;

    elif plasticity_type == 'facil_only':
        noise_amp = 1/7
        d_ab_on = 0; d_pb_on = 0; f_ap_on = 1;

    # Spiking model firing rates in non-SWR state:
    P_freq = 1.94; B_freq = 1.32; A_freq = 12.56;

    # Get Poisson noise parameters for "spontaneous" events:
    noise_params_PP = get_noise_params(P_freq,p_PP,N_P,g_PP,E_rev_P,V_P,noise_amp)
    noise_params_BP = get_noise_params(P_freq,p_BP,N_P,g_BP,E_rev_P,V_B,noise_amp)
    noise_params_AP = get_noise_params(P_freq,p_AP,N_P,g_AP,E_rev_P,V_A,noise_amp)
    noise_params_PB = get_noise_params(B_freq,p_PB,N_B,g_PB,E_rev_B,V_P,noise_amp)
    noise_params_BB = get_noise_params(B_freq,p_BB,N_B,g_BB,E_rev_B,V_B,noise_amp)
    noise_params_AB = get_noise_params(B_freq,p_AB,N_B,g_AB,E_rev_B,V_A,noise_amp)
    noise_params_PA = get_noise_params(A_freq,p_PA,N_A,g_PA,E_rev_A,V_P,noise_amp)
    noise_params_BA = get_noise_params(A_freq,p_BA,N_A,g_BA,E_rev_A,V_B,noise_amp)
    noise_params_AA = get_noise_params(A_freq,p_AA,N_A,g_AA,E_rev_A,V_A,noise_amp)

# B trace low-pass filter for peak finding:
b_findpeak_cutoff = 10 # Hz
# Peak detection height limit:
b_findpeak_height = 45 # 1/s
# Peak detection distance limit:
b_findpeak_dist = 100 # ms
