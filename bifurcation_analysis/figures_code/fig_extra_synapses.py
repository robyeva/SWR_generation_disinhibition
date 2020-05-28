'''
Code to generate bifurcation diagrams of additional synaptic plasticities

1. Depression in B->A and B->P
    fig_bif_extra_depr.eps

2. Facilitation in P->A
    fig_bif_extra_facil.eps
'''

# Import python libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Import additional code:
import helper_functions.bifurcations as bif
import helper_functions.nullclines as nc
import helper_functions.model as model
import helper_functions.params as params
import helper_functions.aux_functions as aux

# Get directory of current file:
file_dir = os.path.dirname(os.path.abspath(__file__))

# LaTeX fonts:
rc('text', usetex=True)

# Fontsize:
fonts = 9

'''
Figure extra depression
e acting on B->A and B->P
'''

# x ticks:
e_ticks=[0, 0.25, 0.5, 0.75, 1]
e_ticklabels=[0,'',0.5,'',1]

# y ticks:
P_ticks=[0,50,100]
B_ticks=[0,100,200]
A_ticks=[0,5,10]

# y range:
pmax = 135
bmax = 250
amax = 15

# Create grid spanning e and B space:
E, B = np.meshgrid(np.arange(0, 1, .01), np.arange(-1, 250, .5))
# Get e nullcline for values in grid:
dE = model.de(E, B, params.tau_d, params.eta_d)

# Load XPPAUT bifurcation diagram:
bs = bif.load_bifurcations(file_dir + '/../bifurcation_diagrams/1param/','e_double',0,1)

# Define figure and subplots:
fig_width = 16.25/2.54
fig, ax = plt.subplots(1,3,figsize=(fig_width,0.2*fig_width))

# Plot e-P bifurcation diagram:
bif.plot_bifurcation(ax[0],aux,bs,'P',[0,1],pmax,'e',e_ticks,e_ticklabels,P_ticks,P_ticks,fonts)
# Plot e-B bifurcation diagram:
bif.plot_bifurcation(ax[1],aux,bs,'B',[0,1],bmax,'e',e_ticks,e_ticklabels,B_ticks,B_ticks,fonts)
# Plot e nullcline:
nc.plot_nullcline(ax[1],E,B,dE,'e nullcline','upper right',(1.05,1.05),fonts)
# Plot e-A bifurcation diagram:
bif.plot_bifurcation(ax[2],aux,bs,'A',[0,1],amax,'e',e_ticks,e_ticklabels,A_ticks,A_ticks,fonts)

# Adjust space between subplots:
plt.subplots_adjust(wspace=0.4)

# Export figure:
fig.savefig(file_dir + '/../figures_output/fig_bif_extra_depr.eps', bbox_inches='tight')

'''
Figure extra facilitation
(1+z) acting on P->A
'''

# x ticks:
z_ticks = e_ticks
z_ticklabels = e_ticklabels

# y ticks:
P_ticks=[0,20,40]
B_ticks=[0,50,100]
A_ticks=[0,5,10]

# y range:
pmax = 50
bmax = 120
amax = 15

# Create grid spanning z and P space:
Z, P = np.meshgrid(np.arange(0, 1, .01), np.arange(-1, 120, .5))
# Get z nullcline for values in grid:
dZ = model.dz(Z, P, params.tau_f, params.eta_f, params.z_max)

# Load XPPAUT bifurcation diagram:
bs = bif.load_bifurcations(file_dir + '/../bifurcation_diagrams/1param/','z',0,1)

# Define figure and subplots:
fig_width = 16.25/2.54
fig, ax = plt.subplots(1,3,figsize=(fig_width,0.2*fig_width))

# Plot z-P bifurcation diagram:
bif.plot_bifurcation(ax[0],aux,bs,'P',[0,1],pmax,'z',z_ticks,z_ticklabels,P_ticks,P_ticks,fonts)
# Plot z nullcline:
nc.plot_nullcline(ax[0],Z,P,dZ,'z nullcline','lower right',(1.05,0.0),fonts)
# Plot z-B bifurcation diagram:
bif.plot_bifurcation(ax[1],aux,bs,'B',[0,1],bmax,'z',z_ticks,z_ticklabels,B_ticks,B_ticks,fonts)
# Plot z-A bifurcation diagram:
bif.plot_bifurcation(ax[2],aux,bs,'A',[0,1],amax,'z',z_ticks,z_ticklabels,A_ticks,A_ticks,fonts)

# Adjust space between subplots:
plt.subplots_adjust(wspace=0.4)

# Export figure:
fig.savefig(file_dir + '/../figures_output/fig_bif_extra_facil.eps', bbox_inches='tight')
