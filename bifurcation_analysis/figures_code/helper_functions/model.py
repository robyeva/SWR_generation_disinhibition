'''
Rate model functions and linear approximation of
pathway-strength requirements
'''

# Import python libraries:
import numpy as np

# Soft-plus function:
def spf(x,k,t):
    return np.log(1+np.exp(k*(x+t)))

# P:
def dp(p,b,a,e,*params):
    w_pp, w_pb, w_pa, k_p, t_p, tau_p = params
    x_p = w_pp*p-w_pb*b-w_pa*a
    return np.array((-p+spf(x_p,k_p,t_p))/tau_p)

# B:
def db(b,p,a,e,*params):
    w_bp, w_bb, w_ba, k_b, t_b, tau_b = params
    x_b = w_bp*p-w_bb*b-w_ba*a
    return np.array((-b+spf(x_b,k_b,t_b))/tau_b)

# A:
def da(a,p,b,e,*params):
    w_ap, w_ab, w_aa, k_a, t_a, tau_a = params
    x_a = w_ap*p-w_ab*b*e-w_aa*a
    return np.array((-a+spf(x_a,k_a,t_a))/tau_a)

# Synaptic depression:
def de(e, b, *params):
    tau_d, eta_d = params
    y = ((1 - e) / tau_d) - eta_d * b * e
    return y

# Synaptic facilitation:
def dz(z, p, *args):
    tau_f, eta_f, z_max = args
    y = (-z / tau_f) + eta_f * p * (z_max - z)
    return y

# Linear approx of pathway strength Requirement 1
# [Activation of P] => [Inactivation of A]
def req1(arg,x,*params):
    k_b, w_bp, w_ab, w_bb = params
    if arg == ('wbp'):
        return x*k_b*w_ab/(1+k_b*w_bb)
    if arg == ('wab'):
        return x*k_b*w_bp/(1+k_b*w_bb)
    if arg == ('wbb'):
        return w_ab*k_b*w_bp/(1+k_b*x)

# Linear approx of pathway strength Requirement 2
# [Activation of P] => [Activation of B]
def req2(arg,x,*params):
    k_a, w_ba, w_ap, w_aa = params
    if arg == ('wba'):
        return x*k_a*w_ap/(1+k_a*w_aa)
    if arg == ('wap'):
        return x*k_a*w_ba/(1+k_a*w_aa)
    if arg == ('waa'):
        return w_ba*k_a*w_ap/(1+k_a*x)

# Linear approx of pathway strength Requirement 3
# [Activation of B] => [Inactivation of P]
def req3(arg,x,*params):
    k_a, w_pa, w_ab, w_aa = params
    if arg == ('wpa'):
        return x*k_a*w_ab/(1+k_a*w_aa)
    if arg == ('wab'):
        return x*k_a*w_pa/(1+k_a*w_aa)
    if arg == ('waa'):
        return w_ab*k_a*w_pa/(1+k_a*x)

# Linear approx of pathway strength Requirement 4
# [Activation of A] => [Inactivation of P]
def req4(arg,x,*params):
    k_b, w_pb, w_ba, w_bb = params
    if arg == ('wpb'):
        return x*k_b*w_ba/(1+k_b*w_bb)
    if arg == ('wba'):
        return x*k_b*w_pb/(1+k_b*w_bb)
    if arg == ('wbb'):
        return w_ba*k_b*w_pb/(1+k_b*x)
