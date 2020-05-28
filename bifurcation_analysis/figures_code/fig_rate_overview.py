'''
Code to generate rate model overview figure
    fig_rate_overview.eps

A) Diagram of rate model with connection strengths

B) "Pseudo-nullclines" in P-B and P-A space for e=0.4
    ("pseudo-nullcline" is the nullcline in 2D space,
     assuming the third population is in steady-state)

C) "Pseudo-nullclines" in P-B and P-A space for e=0.5

D) Bifurcation diagram wrt e
'''

# Import python libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

# Define figure and subplots:
fig_width = 11.6/2.54
fig = plt.figure(figsize=(fig_width,1.1*fig_width))
gs = GridSpec(10,10)
ax_diagram = fig.add_subplot(gs[0:4, 0:6])
ax_n11 = fig.add_subplot(gs[4:7, 0:3])
ax_n21 = fig.add_subplot(gs[7:10, 0:3])
ax_n12 = fig.add_subplot(gs[4:7, 3:6])
ax_n22 = fig.add_subplot(gs[7:10, 3:6])
ax_b1 = fig.add_subplot(gs[1:4, 7:10])
ax_b2 = fig.add_subplot(gs[4:7, 7:10])
ax_b3 = fig.add_subplot(gs[7:10, 7:10])

'''
A) Subplot with rate model diagram
'''
diagram = plt.imread(file_dir + '/rate_model.png')
ax_diagram.imshow(diagram,interpolation='nearest')
ax_diagram.axis('off')
ax_diagram.set_ylim((1450,-150))
ax_diagram.set_xlim((-100,2100))
ax_diagram.set_title(r'\textbf{A}',loc='left',x=+0.0,y=.78,fontsize=fonts)

'''
B) Calculate and plot pseudo-nullclines for e=0.4
'''
e=0.4

# Range of pseudo-nullcline calculation:
pmax = 70
bmax = 120
amax = 15

# Calculate P-B nullcline, assuming A in steady-state:
nPB1 = nc.calc_pseudo_nullcline(model,params,aux,'P','B',pmax,bmax,e,file_dir)
# Calculate P-A nullcline, assuming B in steady-state:
nPA1 = nc.calc_pseudo_nullcline(model,params,aux,'P','A',pmax,amax,e,file_dir)

# Axis ticks:
p_ticks=[0,25,50]
b_ticks=[0,50,100]
a_ticks=[0,5,10]

# Subplot title:
ax_n11.set_title(r'\textbf{B}',loc='left',x=-0.01,y=1.02,fontsize=fonts)
ax_n11.set_title('e = 0.4',loc='center',x=0.5,y=1.02,fontsize=fonts)

# Plot pseudo_nullclines:
nc.plot_pseudo_nullclines(ax_n11,aux,nPB1,'lower right',p_ticks,[],b_ticks,b_ticks,'','B [1/s]',fonts)
nc.plot_pseudo_nullclines(ax_n21,aux,nPA1,'upper right',p_ticks,p_ticks,a_ticks,a_ticks,'P [1/s]','A [1/s]',fonts)

'''
C) Calculate and plot pseudo-nullclines for e=0.5
'''
e=0.5

# Calculate P-B nullcline, assuming A in steady-state:
nPB2 = nc.calc_pseudo_nullcline(model,params,aux,'P','B',pmax,bmax,e,file_dir)
# Calculate P-A nullcline, assuming B in steady-state:
nPA2 = nc.calc_pseudo_nullcline(model,params,aux,'P','A',pmax,amax,e,file_dir)

# Subplot title:
ax_n12.set_title(r'\textbf{C}',loc='left',x=-0.01,y=1.02,fontsize=fonts)
ax_n12.set_title('e = 0.5',loc='center',x=0.5,y=1.02,fontsize=fonts)

# Plot pseudo_nullclines:
nc.plot_pseudo_nullclines(ax_n12,aux,nPB2,'lower right',p_ticks,[],b_ticks,[],'','',fonts)
nc.plot_pseudo_nullclines(ax_n22,aux,nPA2,'upper right',p_ticks,p_ticks,a_ticks,[],'P [1/s]','',fonts)

'''
D) e bifurcation diagrams and nullcline
'''
# Axis range for P:
pmax = 60

# Create grid spanning e and B space:
E, B = np.meshgrid(np.arange(0, 1, .01), np.arange(-1, 250, .5))
# Get e nullcline for values in grid:
dE = model.de(E, B,params.tau_d,params.eta_d)

# e axis ticks:
e_ticks=[0, 0.25, 0.5, 0.75, 1]

# Load e XPPAUT bifurcation diagram:
bs = bif.load_bifurcations(file_dir + '/../bifurcation_diagrams/1param/','e',0,1)

# Subplot title:
ax_b1.set_title(r'\textbf{D}',loc='left',x=-0.01,y=1.02,fontsize=fonts)

# Plot e-P bifurcation diagram:
bif.plot_bifurcation(ax_b1,aux,bs,'P',[0,1],pmax,'',e_ticks,[],p_ticks,p_ticks,fonts,vlines=[0.4,0.5])
# Plot e-B bifurcation diagram:
bif.plot_bifurcation(ax_b2,aux,bs,'B',[0,1],bmax,'',e_ticks,[],b_ticks,b_ticks,fonts,vlines=[0.4,0.5],maxvline=0.86,ylabelpad=0)
# Plot e nullcline:
nc.plot_nullcline(ax_b2,E,B,dE,'e nullcline','upper right',(1.08,1.08),fonts)
# Plot e-A bifurcation diagram:
bif.plot_bifurcation(ax_b3,aux,bs,'A',[0,1],amax,'e',e_ticks,[0,'',0.5,'',1],a_ticks,a_ticks,fonts,vlines=[0.4,0.5])

# Adjust space between subplots:
plt.subplots_adjust(hspace = 0.25,wspace=0.20)

# Export figure:
fig.savefig(file_dir + '/../figures_output/fig_rate_overview.eps',bbox_inches='tight',dpi=800)
