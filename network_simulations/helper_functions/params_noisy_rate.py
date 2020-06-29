import numpy as np

fonts = 9
compress_step = 10

seed = 123

sim_dt = 0.1 # 0.1 ms

# Periodic square inputs:
B_pulses_per = 2 #secs
B_pulses_dur = 10 # ms
B_pulses_amp = 150 #pA

# Poisson noise inputs:
tausyn_P = 2; tausyn_B = 1.5; tausyn_A = 4

poisson_P = 0
poisson_B = 0
poisson_A = 0

# Filter B at a cutoff frequency to find peaks:
b_findpeak_cutoff = 10 # Hz
# B peak detection height:
b_findpeak_height = 45 # 1/s
# B peak detection distance:
b_findpeak_dist = 100 # ms

P_0 = 0.
B_0 = 0.
A_0 = 12.5
# e_0 = 0.8
z_0 = 0.

d_ab_on = 0; d_pb_on = 0; f_ap_on = 0;

# synaptic weights
w_pp = 0; w_bp = 0; w_ap = 0;
w_pb = 0; w_bb = 0; w_ab = 0;
w_pa = 0; w_ba = 0; w_aa = 0;

# activation constants
k_p = 0; k_b = 0; k_a = 0;
t_p = 0; t_b = 0; t_a = 0;

# time constants (ms)
tau_p = 0; tau_b = 0; tau_a = 0;

# synaptic depression
tau_d = 0; eta_d = 0;

# synaptic facilitation
tau_f = 0; eta_f = 0; z_max = 0

# Connection weights:
def get_w_IJ(N_J, p_IJ, g_IG, V_I, tausyn_J, E_rev_J):
    if E_rev_J < 0:
        sign = -1
    else:
        sign = 1
    return sign*round(N_J*p_IJ*g_IG*tausyn_J*(E_rev_J - V_I)*1e-3,2)

def get_noise_params(J_freq, p_IJ, N_J, g_IJ, E_rev_J, V_I, amp):
    poisson_freq = J_freq*p_IJ*N_J
    current_update = amp*g_IJ*(E_rev_J - V_I)
    return np.array([poisson_freq, current_update])

# Default parameters
def set_default_parameters(plasticity_type='default'):

    global w_pp
    global w_bp
    global w_ap
    global w_pb
    global w_bb
    global w_ab
    global w_pa
    global w_ba
    global w_aa

    # activation constants
    global k_p
    global k_b
    global k_a
    global t_p
    global t_b
    global t_a

    # time constants (ms)
    global tau_p
    global tau_b
    global tau_a

    # synaptic depression
    global tau_d
    global eta_d

    # synaptic facilitation
    global tau_f
    global eta_f
    global z_max

    # synaptic plasticity
    global d_ab_on
    global d_pb_on
    global f_ap_on

    # initial condition
    global e_0

    # poisson noise
    global poisson_P
    global poisson_B
    global poisson_A

    # Calc A->I synapse frequency:
    p_PP = 0.01; p_BP = 0.20; p_AP = 0.01
    p_PB = 0.50; p_BB = 0.20; p_AB = 0.20
    p_PA = 0.60; p_BA = 0.60; p_AA = 0.60

    N_P = 8200; N_B = 135; N_A = 50

    # Calc current jump after synpse:
    g_PP = 0.20; g_BP = 0.05; g_AP = 0.20
    g_PB = 0.70; g_BB = 5.00; g_AB = 8.00
    g_PA = 6.00; g_BA = 7.00; g_AA = 4.00

    E_rev_P = 0; E_rev_B = -70; E_rev_A = -70;
    V_P = -52.5; V_B = -54.0; V_A = -52.5;

    # synaptic plasticity:
    if (plasticity_type == 'default') or (plasticity_type == 'extra_dpr'):
        e_0 = 0.8
        amp = 1/8
        if plasticity_type == 'default':
            d_ab_on = 1; d_pb_on = 0; f_ap_on = 0;
        elif plasticity_type == 'extra_dpr':
            d_ab_on = 1; d_pb_on = 1; f_ap_on = 0;

    elif plasticity_type == 'facil_only':
        e_0 = 0.5;
        amp = 1/7
        d_ab_on = 0; d_pb_on = 0; f_ap_on = 1;

    # synaptic weights
    w_pp = get_w_IJ(N_P, p_PP, g_PP, V_P, tausyn_P, E_rev_P);
    w_bp = get_w_IJ(N_P, p_BP, g_BP, V_B, tausyn_P, E_rev_P);
    w_ap = get_w_IJ(N_P, p_AP, g_AP, V_A, tausyn_P, E_rev_P);

    w_pb = get_w_IJ(N_B, p_PB, g_PB, V_P, tausyn_B, E_rev_B);
    w_bb = get_w_IJ(N_B, p_BB, g_BB, V_B, tausyn_B, E_rev_B);
    w_ab = get_w_IJ(N_B, p_AB, g_AB, V_A, tausyn_B, E_rev_B);

    w_pa = get_w_IJ(N_A, p_PA, g_PA, V_P, tausyn_A, E_rev_A);
    w_ba = get_w_IJ(N_A, p_BA, g_BA, V_B, tausyn_A, E_rev_A);
    w_aa = get_w_IJ(N_A, p_AA, g_AA, V_A, tausyn_A, E_rev_A);

    # activation constants
    k_p = 0.47; k_b = 0.41; k_a = 0.48
    t_p = 131.660; t_b = 131.960; t_a = 131.090

    # time constants (ms)
    tau_p = 3; tau_b = 2; tau_a = 6;

    # synaptic platicity
    tau_d = 250; # (ms)
    eta_d = 0.18;
    tau_f = 230; # (ms)
    eta_f = 0.32;
    z_max = 1;

    P_freq = 1.94; B_freq = 1.32; A_freq = 12.56;
    poisson_PP = get_noise_params(P_freq,p_PP,N_P,g_PP,E_rev_P,V_P,amp)
    poisson_BP = get_noise_params(P_freq,p_BP,N_P,g_BP,E_rev_P,V_B,amp)
    poisson_AP = get_noise_params(P_freq,p_AP,N_P,g_AP,E_rev_P,V_A,amp)

    poisson_PB = get_noise_params(B_freq,p_PB,N_B,g_PB,E_rev_B,V_P,amp)
    poisson_BB = get_noise_params(B_freq,p_BB,N_B,g_BB,E_rev_B,V_B,amp)
    poisson_AB = get_noise_params(B_freq,p_AB,N_B,g_AB,E_rev_B,V_A,amp)

    poisson_PA = get_noise_params(A_freq,p_PA,N_A,g_PA,E_rev_A,V_P,amp)
    poisson_BA = get_noise_params(A_freq,p_BA,N_A,g_BA,E_rev_A,V_B,amp)
    poisson_AA = get_noise_params(A_freq,p_AA,N_A,g_AA,E_rev_A,V_A,amp)

    poisson_P = poisson_PP, poisson_PB, poisson_PA
    poisson_B = poisson_BP, poisson_BB, poisson_BA
    poisson_A = poisson_AP, poisson_AB, poisson_AA
