'''
Code to generate figure with bifurcation diagrams
for all 9 connection strengths across populations
'''

# Import python libraries:
import os
import matplotlib.pyplot as plt
from matplotlib import rc

# Import additional code:
import helper_functions.params as params
import helper_functions.bifurcations as bif
import helper_functions.aux_functions as aux

# Get directory of current file:
file_dir = os.path.dirname(os.path.abspath(__file__))

# LaTeX fonts:
rc('text', usetex=True)

# Fontsize:
fonts=9

# Define figure and subplots:
fig_width = 11.6/2.54
fig, ax = plt.subplots(11, 3, figsize=(fig_width,1.75*fig_width),
                        gridspec_kw={'height_ratios': [1, 1, 1, 0.75, 1, 1, 1, 0.75, 1, 1, 1]})
for i in range(3):
    ax[3,i].axis('off')
    ax[7,i].axis('off')

# Folder with XPPAUT bifurcation diagrams:
f = file_dir + '/../bifurcation_diagrams/1param/'

# Parameter range:
pmin=0; pmax=15

# P->P bifurcation diagrams:
bif.plot_weight_bifs_1d(0,0,ax,aux,f,'wpp',params.w_pp,pmin,3.7,15,(75,120,15),fonts)
# B->P bifurcation diagrams:
bif.plot_weight_bifs_1d(0,1,ax,aux,f,'wpb',params.w_pb,pmin,pmax,15,(75,120,15),fonts)
# A->P bifurcation diagrams:
bif.plot_weight_bifs_1d(0,2,ax,aux,f,'wpa',params.w_pa,pmin,pmax,15,(75,120,15),fonts, vlinemax=0.70)

# P->B bifurcation diagrams:
bif.plot_weight_bifs_1d(4,0,ax,aux,f,'wbp',params.w_bp,pmin,pmax,15,(75,120,15),fonts)
# B->B bifurcation diagrams:
bif.plot_weight_bifs_1d(4,1,ax,aux,f,'wbb',params.w_bb,pmin,pmax,15,(75,120,15),fonts)
# A->B bifurcation diagrams:
bif.plot_weight_bifs_1d(4,2,ax,aux,f,'wba',params.w_ba,pmin,pmax,15,(75,120,15),fonts, vlinemax=0.70)

# P->A bifurcation diagrams:
bif.plot_weight_bifs_1d(8,0,ax,aux,f,'wap',params.w_ap,pmin,pmax,15,(75,120,15),fonts)
# B->A bifurcation diagrams:
bif.plot_weight_bifs_1d(8,1,ax,aux,f,'wab',params.w_ab,pmin,pmax,15,(75,120,15),fonts)
# A->A bifurcation diagrams:
bif.plot_weight_bifs_1d(8,2,ax,aux,f,'waa',params.w_aa,pmin,pmax,15,(75,120,15),fonts)

# Adjust space between subplots:
plt.subplots_adjust(hspace=0.15,wspace=0.2)

# Export figure:
fig.savefig(file_dir + '/../figures_output/fig_bifurcations_1d.eps', bbox_inches='tight')
