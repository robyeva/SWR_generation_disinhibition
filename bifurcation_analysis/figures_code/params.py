'''
Default rate model parameters
'''

# Synaptic weights: [pA.s]
w_pp = 1.72; w_bp = 8.86; w_ap = 1.72
w_pb = 1.24; w_bb = 3.24; w_ab = 5.66
w_pa = 12.60; w_ba = 13.44; w_aa = 8.40

# Activation function constants:
k_p = 0.47; k_b = 0.41; k_a = 0.48 # [1/pA]
t_p = 131.660; t_b = 131.960; t_a = 131.090 # [pA]

# Time constants: [s]
tau_p = 0.003; tau_b = 0.002; tau_a = 0.006

# Synaptic efficacy constants:
tau_d = 0.250; tau_f = 0.230 # [s]
eta_d = 0.18; eta_f = 0.32; z_max = 1 # [1]
