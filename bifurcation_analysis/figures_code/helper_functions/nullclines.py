'''
Functions used to calculate and plot nullclines across figures.
    "Pseudo-nullcline" is the nullcline in 2D space, assuming the third
    population is in steady-state.
'''

# Import python libraries:
import numpy as np
from scipy.optimize import fsolve

# Plot nullcline:
def plot_nullcline(ax,X,Y,dX,legend,legend_loc,box,font_size, line_width=2):
    c = ax.contour(X, Y, dX, levels=[0], linewidths=line_width, colors='gray')
    h, _ = c.legend_elements()
    ax.legend([h[0]], [legend], frameon=False,loc=legend_loc, bbox_to_anchor=box, fontsize=font_size, handlelength=0.5, handletextpad=0.25)

# Return population differential equation:
def dpop(model,x):
    if x == 0: return model.dp
    if x == 1: return model.db
    if x == 2: return model.da

# Return parameters for dx/dt model equation:
def pop_params(params,x):
    if x == 0:
        return params.w_pp, params.w_pb, params.w_pa,params.k_p, params.t_p, params.tau_p
    if x == 1:
        return params.w_bp, params.w_bb, params.w_ba,params.k_b, params.t_b, params.tau_b
    if x == 2:
        return params.w_ap, params.w_ab, params.w_aa,params.k_a, params.t_a, params.tau_a

# Reorder arguments in desired order:
def order_pops(x1, x2, X1, X2, e, params):
    if x1 > x2: return (X2, X1, e) + params
    if x1 < x2: return (X1, X2, e) + params

# Return index for given population:
def pop_number(pop):
    if pop == 'P': return 0
    if pop == 'B': return 1
    if pop == 'A': return 2

# Calculate pseudo-nullcline (nullcline pop1-pop2 assuming pop3 is in steady-state):
def calc_pseudo_nullcline(model,params,aux,pop1, pop2, pop1max, pop2max, e, file_dir):
    # Attribute index {0,1,2} to each population:
    x1 = pop_number(pop1)
    x2 = pop_number(pop2)

    # Don't calculate if both populations are the same:
    if x1 == x2:
        print("Input error.")
        return 0

    # Find index of thirs population:
    x = np.array([0, 1, 2])
    x3 = x[(x != x1) & (x != x2)].item(0)

    # Get differential equation for each population:
    dx1 = dpop(model,x1)
    dx2 = dpop(model,x2)
    dx3 = dpop(model,x3)

    # Name of file where calculaton will be saved:
    fname = file_dir + '/pseudo_nullclines/ncline_'+aux.pop_name(x1)+aux.pop_name(x2)+'_e=%.1lf'%e
    # Try to load file; if it exists, skip calculation:
    try:
        X1, X2, dX1, dX2 = np.load(fname+'.npy')
    # If file doesn't exist make calculation:
    except:
        print('calculating nullcline '+fname+' ...')

        # Create grid spanning 2D space of the two populations:
        X1, X2 = np.meshgrid(np.arange(-0.05 * pop1max, pop1max, 0.005 * pop1max), np.arange(-0.05 * pop2max, pop2max, 0.005 * pop2max))

        # Initialize array where population 3 steady-state will be saved:
        X3 = np.zeros(X1.shape)

        # Calculate steady-states of third population for each element in 2D grid:
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X3[i, j] = fsolve(dx3, 0, order_pops(x1, x2, X1[i, j], X2[i, j], e, pop_params(params,x3)))

        # Get nullclines with third population in steady-state:
        dX1 = dx1(X1, *order_pops(x2, x3, X2, X3, e, pop_params(params,x1)))
        dX2 = dx2(X2, *order_pops(x1, x3, X1, X3, e, pop_params(params,x2)))

        # Save results in a file:
        np.save(fname,(X1,X2,dX1,dX2))

    return e, X1, X2, dX1, dX2, pop1, pop2, aux.pop_name(x3)

# Plot pseudo-nullclines:
def plot_pseudo_nullclines(ax,aux,nullcline,location,xticks,xticklabels,yticks,yticklabels,xlabel,ylabel,font_size,spine_width=0.75, line_width=2):
    e, X1, X2, dX1, dX2, pop1, pop2, pop3 = nullcline

    # Plot pseudo_nullclines:
    c1 = ax.contour(X1, X2, dX1, levels=[0], linewidths=line_width, colors=aux.pop_color(pop1))
    c2 = ax.contour(X1, X2, dX2, levels=[0], linewidths=line_width, colors=aux.pop_color(pop2))

    # Legend settings:
    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    if location == 'lower right': box = (1.08,-0.05)
    else: box = (1.08,1.05)
    ax.legend([h1[0], h2[0]], [pop1 + ' nullcline', pop2 + ' nullcline'], frameon=False,loc=location, bbox_to_anchor=box,
            fontsize=font_size, handlelength=0.5, handletextpad=0.25, labelspacing=0.25)

    # Tick settings:
    ax.tick_params(top=False,which='both',labelsize=font_size,direction='in',width=spine_width)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels=yticklabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=xticklabels)

    # Axes labels:
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    # Spine settings:
    for axis in ['bottom','left']: ax.spines[axis].set_linewidth(spine_width)
    for axis in ['top','right']: ax.spines[axis].set_visible(False)
