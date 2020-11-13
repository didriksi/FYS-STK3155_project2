import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.text as text
plt.style.use('ggplot')

simple_plotter = lambda ax, x, y, *args, **kwargs: ax.plot(x, y, *args, **kwargs)

def trisurface_plotter(ax, x, y, *args, **kwargs):
    ax.plot_trisurf(x[:,0], x[:,1], y, *args, cmap=cm.coolwarm, linewidth=0, antialiased=False, **kwargs)

def confidence_interval_plotter(ax, x, lower_midddle_upper_y, *args, color="C0", **kwargs):
    """Plots line with confidence interval, figure must be saved or shown after function call.

    Parameters:
    -----------
    ax:         matplotlib.pyplot.axes object
                Axis to plot on
    x:          array of shape (n, )
                x positions
    lower_midddle_upper_y:
                array of shape (3, n) or array of shape (2, n)
                Lower and upper bound for confidence interval, and the middle line to be emphasized.
    *args:      Not used, only here for compatibility.
    color_MSE:  str
                Color of MSE of optimal model, C0 gives the next color in normal matplotlib order
    color_shading:
                str
                Color of confidence interval, C0 gives the next color in normal matplotlib order
    **kwargs:   Passed onwards to plt.fill_between()
    """
    fill_kwargs = {'alpha': 0.5}
    fill_kwargs.update(kwargs)
    if lower_midddle_upper_y.shape[0] == 3:
        lower = lower_midddle_upper_y[0]
        middle = lower_midddle_upper_y[1]
        upper = lower_midddle_upper_y[2]
        ax.plot(x, middle, color)
        ax.fill_between(x, lower, upper, color=color, **fill_kwargs)
    elif lower_midddle_upper_y.shape[0] == 2:
        lower = lower_midddle_upper_y[0]
        upper = lower_midddle_upper_y[1]
        ax.fill_between(x, lower, upper, color=color, **fill_kwargs)
    else:
        if lower_midddle_upper_y.shape[0] == 1:
            middle = lower_midddle_upper_y[0]
        else:
            middle = lower_midddle_upper_y
        ax.plot(x, middle, color)

def box_plotter(ax, x, y, *args, **kwargs):
    ax.boxplot(y, *args, **kwargs)
    ax.set_xticks([]) #x[::len(x)//10])

def scatter_plotter(ax, x, y, *args, **kwargs):
    ax.scatter(x, y, *args, **kwargs)

def side_by_side(*plots, plotter=simple_plotter, axis_labels=('x', 'y', 'z'), title="plot", projection=None, **kwargs):
    """Plots several plots with the same x side by side. Can also make an animation of them from different angles.
    
    Parameters:
    -----------
    plots:      (title: str, x and y: arrays of shape plotter can handle. (n, ) by default)
                The different data you want plotted, in up to 8 lists. y can also be a list of y's
                and args and kwargs that are sent to the plotter function.
                If an *arg is a function, it overrides the plotter function for this y.
    plotter:    ax, x, y, *args, **kwargs -> matplotlib.pyplot plot
                Function that makes plot. Must be compatible with projection type.
                Default is lambda ax, x, y, *args, **kwargs: ax.plot(x, y, *args, **kwargs)
    axis_labels:(str, str, str)
                Labels for each axis. Default is ['x', 'y', 'z']
    title:      str, (str, str), or (str, str, str)
                Title for entire plot and filename. If list of two strings the second element is filename.
                If list of three strings, the last one is suptitle.
    view_angles:(float, float)
                Elevation and azimuth angles of plot if projection=='3d'.
    """

    if len(plots) <= 3:
        fig = plt.figure(figsize=(4+len(plots)*4, 8))
        subplot_shape = (1, len(plots))
    elif len(plots) <= 8:
        fig = plt.figure(figsize=(4+len(plots)*4, 15))
        subplot_shape = (2, int(np.ceil(len(plots)/2)))
    elif len(plots) == 9:
        fig = plt.figure(figsize=(18, 15))
        subplot_shape = (3, int(len(plots)/3))

    fig.suptitle(title[0] if isinstance(title, list) else title, y = 0.96, fontsize=22)
    if isinstance(title, list) and len(title) == 3:
        fig.text(0.02, 0.02, (title[2] if isinstance(title, list) else title), fontsize=14)
        
    y0 = plots[0][2]
    if isinstance(y0, list):
        ylim = (np.min(y0[0][0]), np.max(y0[0][0]))
    else:
        ylim = (np.min(y0), np.max(y0))
    
    axs = []
    for i, (ax_title, x, y, *legend) in enumerate(plots):
        axs.append(fig.add_subplot(*subplot_shape, i+1, projection=projection))
        axs[i].set_title(ax_title)
        
        for _y, *args_kwargs in (y if isinstance(y, list) else [[y]]):
            this_plotter = plotter
            plotter_kwargs = {}; plotter_args = []
            for arg_kwarg in args_kwargs:
                if callable(arg_kwarg):
                    this_plotter = arg_kwarg
                elif isinstance(arg_kwarg, dict):
                    plotter_kwargs.update(arg_kwarg)
                else:
                    plotter_args.append(arg_kwarg)

            this_plotter(axs[i], x, _y, *plotter_args, **plotter_kwargs)

            ylim = (min(np.min(_y), ylim[0]), max(np.max(_y), ylim[1]))

        if legend is not None and legend != []:
            legend_args = legend[0][0]
            legend_kwargs = legend[0][1]

            axs[i].legend(*legend_args, **legend_kwargs)
        else:
            axs[i].legend()
        
    for ax in axs:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if 'yscale' in kwargs:
            ax.set_yscale(kwargs['yscale'])

        if projection == '3d':
            if 'view_angles' in kwargs:
                ax.view_init(elev=kwargs['view_angles'][0], azim=kwargs['view_angles'][1])
            ax.set_zlim(*ylim)
            ax.set_zlabel(axis_labels[2])
        else:
            ax.set_ylim(*ylim)
    
    plt.savefig(f"../plots/{title[1] if isinstance(title, list) else title}.png")

class LegendObject(object):
    """Creates legend for plot_MSE_and_CI

    Parameters:
    -----------
    facecolor:  str
                The color of the legend for the corresponding plot
    """

    def __init__(self, facecolor='red'):
        self.facecolor = facecolor

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            (x0, y0), width, height, facecolor=self.facecolor)
        handlebox.add_artist(patch)

        return patch

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_text=None, font="Georgia"):
    '''
    Draw a neural network cartoon using matplotilb.

    Created by github user @craffel. Node annotation added by @@anbrjohn, modifed by @severs-ml.
    Layer labels added by @severs-ml

    :usage:
        >> fig = plt.figure(figsize=(12, 12))
        >> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], ['x1', 'x2','x3','x4'])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
        - layer_text : list of str
            List of node annotations in top-down left-right order
    '''
    font_layer = {'fontname': font}
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing
            circle = plt.Circle((x, y), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                if len(layer_text[n]) > 0:
                    text = layer_text[n].pop(0)
                    plt.annotate(text, xy=(x, y), zorder=5, ha='center', va='center')

    #Layer labels

    input_layer_y = v_spacing * (layer_sizes[0] - 1) / 2. + (top + bottom) / 2. + .08
    plt.annotate("Input", xy=(0.062, input_layer_y), **font_layer)

    if len(layer_sizes[1:-1]) > 1:
        text = "Hidden layers"
    else:
        text = "Hidden layer"
    hidden_layer_y = max([v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2. for layer_size in layer_sizes[1:-1]]) + 0.02
    plt.annotate(text, xy=(0.5, hidden_layer_y+0.07), ha='center', va='center', **font_layer)

    output_layer_y = v_spacing * (layer_sizes[-1] - 1) / 2. + (top + bottom) / 2. + .08
    plt.annotate("Output", xy=(x-0.05, output_layer_y), **font_layer)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k', lw=0.8)
                ax.add_artist(line)