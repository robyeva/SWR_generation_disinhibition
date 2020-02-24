'''
Code to generate figure with 2D bifurcation
diagrams showing bistability regions

A-D) Connection strengths involved in pathway-strength requirements
     - Numerical XPPAUT two parameter bifurcation point
     - Linear approximation of requirements

E-G) Slope and threshold of activation function for P, B, and A
'''

# Import python libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

# Import additional code:
import bifurcations as bif
import params
import model

# Get directory of current file:
file_dir = os.path.dirname(os.path.abspath(__file__))

# LaTeX fonts:
rc('text', usetex=True)

# Fontsize:
fonts = 9

# Define figure and subplots:
fig_width = 11.6/2.54
fig = plt.figure(figsize=(fig_width,1.25*fig_width))
gs = gridspec.GridSpec(56, 5, width_ratios=[0.75, 0.75, 0.75, 0.4, 1])
ax_req11 = fig.add_subplot(gs[0:8, 0])
ax_req12 = fig.add_subplot(gs[0:8, 1])
ax_req13 = fig.add_subplot(gs[0:8, 2])
ax_blankh1 = fig.add_subplot(gs[8:16,0:3])
ax_req21 = fig.add_subplot(gs[16:24, 0])
ax_req22 = fig.add_subplot(gs[16:24, 1])
ax_req23 = fig.add_subplot(gs[16:24, 2])
ax_blankh2 = fig.add_subplot(gs[24:32,0:3])
ax_req31 = fig.add_subplot(gs[32:40, 0])
ax_req32 = fig.add_subplot(gs[32:40, 1])
ax_req33 = fig.add_subplot(gs[32:40, 2])
ax_blankh3 = fig.add_subplot(gs[40:48,0:3])
ax_req41 = fig.add_subplot(gs[48:56, 0])
ax_req42 = fig.add_subplot(gs[48:56, 1])
ax_req43 = fig.add_subplot(gs[48:56, 2])
ax_blankv = fig.add_subplot(gs[0:56,3])
ax_afp = fig.add_subplot(gs[0:14, 4])
ax_blankh4 = fig.add_subplot(gs[14:21,4])
ax_afb = fig.add_subplot(gs[21:35, 4])
ax_blankh5 = fig.add_subplot(gs[35:42,4])
ax_afa = fig.add_subplot(gs[42:56, 4])
ax_blankv.axis('off')
ax_blankh1.axis('off')
ax_blankh2.axis('off')
ax_blankh3.axis('off')
ax_blankh4.axis('off')
ax_blankh5.axis('off')

# Folder with XPPAUT 2D bifurcation diagrams:
f = file_dir + '/../bifurcation_diagrams/2param/'

'''
A-D) Connection strengths
'''

# Axes ranges:
w1min, w1max, w2min, w2max = 0, 20, 0, 20

# Axes ticks:
xlabels = [0, 7.5, 15]
ylabels = [7.5, 15]

# Fake hatch for linear approximation region:
imax = 15
x = np.arange(0,20,0.1)
def y(x,i,m=1):
    b=(-6+i)*3
    return m*x+b
hatchwidth = 0.75
hatchcolor = '#2e8a57'

'''
A) Requirement 1, [Activation of P] => [Inactivation of A]
'''
# Subplot title:
ax_req11.set_title(r'\textbf{A}',loc='left',x=-0.55,y=1.05,fontsize=fonts)
ax_req12.set_title(r'[P$\rightarrow$A] + [P$\rightarrow$B$\rightarrow$A] $<$ 0',
                   loc='center',x=0.5,y=1.05,fontsize=fonts,color=hatchcolor)

# Plot 2D numerical bifurcation diagram:
bif.plot_weight_bifs_2d(f,ax_req11, 'wap', 'wbp', True, 'd', params.w_ap, params.w_bp, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_req12, 'wap', 'wab', True, 'd', params.w_ap, params.w_ab, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)
bif.plot_weight_bifs_2d(f,ax_req13, 'wap', 'wbb', True, 'l', params.w_ap, params.w_bb, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)

# Plot region where where requirement is met in its linear approximation:
req1_params = params.k_b, params.w_bp, params.w_ab, params.w_bb
ax_req11.plot(x,model.req1('wbp',x,*req1_params),linewidth=hatchwidth,color=hatchcolor)
ax_req12.plot(x,model.req1('wab',x,*req1_params),linewidth=hatchwidth,color=hatchcolor)
ax_req13.plot(x,model.req1('wbb',x,*req1_params),linewidth=hatchwidth,color=hatchcolor)
for i in range(imax):
    xx=x[y(x,i) < model.req1('wbp',x,*req1_params)]
    ax_req11.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) < model.req1('wab',x,*req1_params)]
    ax_req12.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) < model.req1('wbb',x,*req1_params)]
    ax_req13.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)

'''
B) Requirement 2, [Activation of P] => [Activation of B]
'''
# Subplot title:
ax_req21.set_title(r'\textbf{B}',loc='left',x=-0.55,y=1.05,fontsize=fonts)
ax_req22.set_title(r'[P$\rightarrow$B] + [P$\rightarrow$A$\rightarrow$B] $>$ 0',
                   loc='center',x=0.5,y=1.05,fontsize=fonts,color=hatchcolor)

# Plot 2D numerical bifurcation diagram:
bif.plot_weight_bifs_2d(f,ax_req21, 'wbp', 'wba', True, 'u', params.w_bp, params.w_ba, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_req22, 'wbp', 'wap', True, 'u', params.w_bp, params.w_ap, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)
bif.plot_weight_bifs_2d(f,ax_req23, 'wbp', 'waa', True, 'u', params.w_ba, params.w_aa, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)

# Plot region where where requirement is met in its linear approximation:
req2_params = params.k_a, params.w_ba, params.w_ap, params.w_aa
ax_req21.plot(x,model.req2('wba',x,*req2_params),linewidth=hatchwidth,color=hatchcolor)
ax_req22.plot(x,model.req2('wap',x,*req2_params),linewidth=hatchwidth,color=hatchcolor)
ax_req23.plot(x,model.req2('waa',x,*req2_params),linewidth=hatchwidth,color=hatchcolor)
for i in range(imax):
    xx=x[y(x,i) > model.req2('wba',x,*req2_params)]
    ax_req21.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) > model.req2('wap',x,*req2_params)]
    ax_req22.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) > model.req2('waa',x,*req2_params)]
    ax_req23.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)

'''
C) Requirement 3, [Activation of B] => [Activation of P]
'''
# Subplot title:
ax_req31.set_title(r'\textbf{C}',loc='left',x=-0.55,y=1.05,fontsize=fonts)
ax_req32.set_title(r'[B$\rightarrow$P] + [B$\rightarrow$A$\rightarrow$P] $>$ 0',
                   loc='center',x=0.5,y=1.05,fontsize=fonts,color=hatchcolor)

# Plot 2D numerical bifurcation diagram:
bif.plot_weight_bifs_2d(f,ax_req31, 'wpb', 'wpa', True, 'd', params.w_pb, params.w_pa, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_req32, 'wpb', 'wab', True, 'd', params.w_pb, params.w_ab, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)
bif.plot_weight_bifs_2d(f,ax_req33, 'wpb', 'waa', True, 'd', params.w_pb, params.w_aa, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)

# Plot region where where requirement is met in its linear approximation:
req3_params = params.k_a, params.w_pa, params.w_ab, params.w_aa
ax_req31.plot(x,model.req3('wpa',x,*req3_params),linewidth=hatchwidth,color=hatchcolor)
ax_req32.plot(x,model.req3('wab',x,*req3_params),linewidth=hatchwidth,color=hatchcolor)
ax_req33.plot(x,model.req3('waa',x,*req3_params),linewidth=hatchwidth,color=hatchcolor)
for i in range(imax):
    xx=x[y(x,i) < model.req3('wpa',x,*req3_params)]
    ax_req31.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) < model.req3('wab',x,*req3_params)]
    ax_req32.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) < model.req3('waa',x,*req3_params)]
    ax_req33.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)

'''
D) Requirement 4, [Activation of A] => [Inactivation of P]
'''
# Subplot title:
ax_req41.set_title(r'\textbf{D}',loc='left',x=-0.55,y=1.05,fontsize=fonts)
ax_req42.set_title(r'[A$\rightarrow$P] + [A$\rightarrow$B$\rightarrow$P] $<$ 0',
                   loc='center',x=0.5,y=1.05,fontsize=fonts,color=hatchcolor)

# Plot 2D numerical bifurcation diagram:
bif.plot_weight_bifs_2d(f,ax_req41, 'wpa', 'wpb', True, 'u', params.w_pa, params.w_pb, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_req42, 'wpa', 'wba', True, 'u', params.w_pa, params.w_ba, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)
bif.plot_weight_bifs_2d(f,ax_req43, 'wpa', 'wbb', True, 'u', params.w_pa, params.w_bb, w1min, w1max, w2min, w2max, xlabels, ylabels, fonts, ytext=False)

# Plot region where where requirement is met in its linear approximation:
req4_params = params.k_b, params.w_pb, params.w_ba, params.w_bb
ax_req41.plot(x,model.req4('wpb',x,*req4_params),linewidth=hatchwidth,color=hatchcolor)
ax_req42.plot(x,model.req4('wba',x,*req4_params),linewidth=hatchwidth,color=hatchcolor)
ax_req43.plot(x,model.req4('wbb',x,*req4_params),linewidth=hatchwidth,color=hatchcolor)
for i in range(imax):
    xx=x[y(x,i) > model.req4('wpb',x,*req4_params)]
    ax_req41.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) > model.req4('wba',x,*req4_params)]
    ax_req42.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)
    xx=x[y(x,i) > model.req4('wbb',x,*req4_params)]
    ax_req43.plot(xx,y(xx,i),linewidth=hatchwidth,color=hatchcolor)

'''
E-G) Slope (k) and threshold (t) of softplus activation function
'''
# Axes tick labels:
xlabels = [0, 100, 200]
ylabels = [0.4, 0.8]

# Subplot title:
ax_afp.set_title(r'\textbf{E}',loc='left',x=-0.40,y=1.05,fontsize=fonts)
ax_afb.set_title(r'\textbf{F}',loc='left',x=-0.40,y=1.05,fontsize=fonts)
ax_afa.set_title(r'\textbf{G}',loc='left',x=-0.40,y=1.05,fontsize=fonts)

# Plot 2D numerical bifurcation diagram for P, B, and A:
bif.plot_weight_bifs_2d(f,ax_afp, 'tp', 'kp', False, 'u', params.t_p, params.k_p, 0, 250, 0, 1, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_afb, 'tb', 'kb', False, 'u', params.t_b, params.k_b, 0, 250, 0, 1, xlabels, ylabels, fonts)
bif.plot_weight_bifs_2d(f,ax_afa, 'ta', 'ka', False, 'u', params.t_a, params.k_a, 0, 250, 0, 1, xlabels, ylabels, fonts)

# Adjust space between subplots:
plt.subplots_adjust(hspace=1.5,wspace=0.10)

# Export figure:
fig.savefig(file_dir + '/../figures_output/fig_bifurcations_2d.eps', bbox_inches='tight')
