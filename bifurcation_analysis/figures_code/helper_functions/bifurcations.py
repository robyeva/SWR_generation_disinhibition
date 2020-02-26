'''
Functions used to plot bifurcation diagrams across figures.
'''

# Import python libraries:
import numpy as np

# Import additional code:
import helper_functions.aux as aux

# Return column index to be read in AUTO .dat files:
def get_index(name):
    if name == 'P': return 9
    if name == 'B': return 10
    if name == 'A': return 8

# Load AUTO .dat files and return branches of bifurcation diagram:
def load_bifurcations(folder, param, pmin, pmax):
    # Load line branch:
    b1 = np.loadtxt(folder + 'auto_' + param + '_line.dat')
    # Load fold branch:
    b2 = np.loadtxt(folder + 'auto_' + param + '_fold.dat')

    # Trim branches:
    b1 = b1[(b1[:, 3] >= (pmin + pmax * 0.005)) & (b1[:, 3] <= pmax * 0.995)]
    b2 = b2[(b2[:, 3] >= (pmin + pmax * 0.005)) & (b2[:, 3] <= pmax * 0.995)]

    # Separate branches w.r.t. stability:
    b1_stable = b1[b1[:, 0] == 1]
    b1_unstable = b1[b1[:, 0] == 2]
    b2_stable = b2[b2[:, 0] == 1]
    b2_unstable = b2[b2[:, 0] == 2]

    return b1_stable, b1_unstable, b2_stable, b2_unstable

# Plot all branches of bifurcation diagram:
def plot_branches(ax, bs, name, line_width=2):
    b_stable = bs[0], bs[2]
    b_unstable = bs[1], bs[3]
    for bi in b_stable:
        ax.plot(bi[:, 3], bi[:, get_index(name)], '-', c=aux.pop_color(name), linewidth=line_width)
    for bi in b_unstable:
        ax.plot(bi[:, 3], bi[:, get_index(name)], '--', c=aux.pop_color(name), linewidth=line_width)

# Plot and frame bifurcation diagram:
def plot_bifurcation(ax,bs,name,xlim,ymax,xlabel,xticks,xticklabels,yticks,yticklabels,font_size,vlines=[],spine_width=0.75,line_width=2,maxvline=1,ylabelpad=2):
    # Plot all branches of bifurcation diagram:
    plot_branches(ax,bs,name)

    # Axes limits:
    ax.set_xlim(xlim)
    ax.set_ylim([-0.05*ymax,ymax])

    # Axes labels:
    ax.set_xlabel(xlabel,fontsize=font_size)
    ax.set_ylabel(name + ' [1/s]',fontsize=font_size, labelpad=ylabelpad)

    # Tick settings:
    ax.tick_params(top=False,which='both',labelsize=font_size,direction='in',width=spine_width)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    if yticks != []: ax.set_yticks(yticks)
    if yticklabels != []: ax.set_yticklabels(yticklabels)

    # Vertical lines:
    for i in vlines:
        ax.axvline(x=i, ymin=0, ymax=maxvline, color='darkgray', ls='--', lw=line_width/2)

    # Spine settings:
    for axis in ['top','right']: ax.spines[axis].set_visible(False)
    for axis in ['bottom','left']: ax.spines[axis].set_linewidth(spine_width)

# Plot bifurcation diagrams for P, B, and A w.r.t. a synaptic weight:
def plot_weight_bifs_1d(i, j, ax, folder, w_name, w_val, pmin, pmax, xmax, ymax,font_size,spine_width=0.75, vlinemax=1):
    pop_names = ['P', 'B', 'A']
    bs = load_bifurcations(folder, w_name, pmin, pmax)

    # Write text in top graph:
    ax[i, j].set_title(r'\textbf{'+aux.connection_name(w_name)+'}', fontsize=font_size, x = 0.85, y = 0.65)

    # Left column, middle graph:
    if j == 0: ax[i+1,j].set_ylabel('Population rate [1/s]', fontsize=font_size)

    # Bottom graph where x axis is plotted:
    ax[i+2, j].set_xticks([0,7.5,15])
    ax[i+2, j].set_xticklabels(labels=[0,7.5,15], fontsize=font_size)
    ax[i+2, j].set_xlabel(aux.param_name(w_name) + ' ' + aux.param_units(w_name),fontsize=font_size)

    for k in range(3):
        # Plot all branches of bifurcation diagram:
        plot_branches(ax[i+k,j], bs, pop_names[k], line_width=1.5)

        # Set axes limits:
        ax[i+k, j].set_xlim([pmin, xmax])
        ax[i+k, j].set_ylim([-0.1*ymax[k], ymax[k]])

        # Set tick parameters:
        ax[i+k, j].tick_params(top=False,right=False,which='both',labelsize=font_size,direction='in',width=spine_width)

        # Hide x axis for top and center graphs:
        if k < 2:
            ax[i+k, j].set_xticks([])
            ax[i+k, j].set_xticklabels(labels=[])
            ax[i+k, j].spines['bottom'].set_visible(False)

        # Set y ticks:
        yticks = [int(0+ymax[k]*l/3) for l in range(3)]
        ax[i+k, j].set_yticks(yticks)
        if j == 0: ax[i+k, j].set_yticklabels(labels=yticks, fontsize=font_size)
        if j > 0: ax[i+k, j].set_yticklabels(labels=[])

        # Spine settings:
        for axis in ['bottom','left']:
            ax[i+k, j].spines[axis].set_linewidth(spine_width)
        for axis in ['top','right']:
            ax[i+k, j].spines[axis].set_visible(False)
        # For center and bottom graphs plot whole vertical line:
        if k > 0: ax[i+k, j].axvline(x=w_val, color='black', ls='--', lw=0.75)
        # For top graph trim vertical line to avoid overlap with title:
        if k == 0: ax[i+k, j].axvline(x=w_val, ymin=0, ymax=vlinemax, color='black', ls='--', lw=0.75)

# Plot 2D bifurcation diagrams:
def plot_weight_bifs_2d(folder, ax, p1, p2, invert, region, w1, w2, p1min, p1max, p2min, p2max, xlabels, ylabels, font_size, ytext = True, spine_width=0.75):
    a = np.loadtxt(folder + 'auto_' + p1 + '_' + p2 + '.dat')
    # Plot w1 on y axis and w2 on x axis:
    if invert == True:
        px, pxmin, pxmax, wx = p2, p2min, p2max, w2
        py, pymin, pymax, wy = p1, p1min, p1max, w1
        xx = a[:, 1]
        yy = a[:, 0]
    # Plot w1 on x axis and w2 on y axis:
    if invert == False:
        px, pxmin, pxmax, wx = p1, p1min, p1max, w1
        py, pymin, pymax, wy = p2, p2min, p2max, w2
        xx = a[:, 0]
        yy = a[:, 1]

    # Plot bifurcation line:
    ax.plot(xx, yy, '-', color='black',lw=1)

    # Set axes limits:
    ax.set_xlim(pxmin, pxmax)
    ax.set_ylim(pymin, pymax)

    # Fill bistable region according to manual command:
    if region == 'u':
        ax.fill_between(xx, yy, 100, color='darkgray')
    if region == 'd':
        ax.fill_between(xx, -100, yy, color='darkgray')
    if region == 'l':
        ax.fill_betweenx(yy, -100, xx, color='darkgray')
    if region == 'r':
        ax.fill_betweenx(yy, xx, 100, color='darkgray')

    # Mark default parameter values:
    ax.scatter(wx, wy, marker='x', color='black', s=5, lw=0.5)

    # x axis settings:
    ax.set_xticks(xlabels)
    ax.set_xticklabels(labels=xlabels, fontsize=font_size)
    ax.set_xlabel(aux.param_name(px), fontsize=font_size)

    # y axis settings:
    ax.set_yticks(ylabels)
    if ytext == True:
        ax.set_yticklabels(labels=ylabels, fontsize=font_size)
        ax.set_ylabel(aux.param_name(py), fontsize=font_size)
    else:
        ax.set_yticklabels(labels=[])

    # Tick parameters settings:
    ax.tick_params(top=False,right=False,which='both',labelsize=font_size,direction='in',width=spine_width)

    # Spine settings:
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(spine_width)
    for axis in ['top','right']:
        ax.spines[axis].set_visible(False)
