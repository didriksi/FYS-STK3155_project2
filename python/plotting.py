import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
plt.style.use('ggplot')

def side_by_side(*plots, axis_labels=['x', 'y', 'z'], title="plot", filename="plot", animation=False, _3d=True, init_azim=240):
    """Plots two plots with the same x side by side. Can also make an animation of them from different angles.
    
    Parameters:
    -----------
    plots:      (title: str, x: array of shape (n, 2) if _3d==True and (n, ) if not, y: arrays of shape (n, ))
                The different data you want plotted, in up to 8 lists. y can also be a list of y's and optional plot labels
    axis_labels:(x: str, y: str, z: str)
                Labels for eachaxis
    title:      str
                Title for entire plot.
    filename:   str
                Name of file. Should not include filetype.
    animation:  bool
                Set to true if you want to get a .mp4 of the plots spinning around their own Z-axis. Only works for 3d.
    init_azim:  float
                Azimuth angle of plot, or initial azimuth angle of animation.
    """
    if len(plots) == 1:
        fig = plt.figure(figsize=(5,5))
        subplot_shape = (1, 1)
    elif len(plots) == 2:
        fig = plt.figure(figsize=(10,5))
        subplot_shape = (1, 2)
    elif len(plots) == 3:
        fig = plt.figure(figsize=(15,5))
        subplot_shape = (1, 3)
    elif len(plots) == 4:
        fig = plt.figure(figsize=(12,10))
        subplot_shape = (2, 2)
    elif len(plots) <= 6:
        fig = plt.figure(figsize=(12,12))
        subplot_shape = (2, 3)
    elif len(plots) <= 8:
        fig = plt.figure(figsize=(12,15))
        subplot_shape = (2, 4)

    fig.suptitle(title)

    if isinstance(plots[0][2], list):
        ylim = (np.min(plots[0][2][0][0]), np.max(plots[0][2][0][0]))
    else:
        ylim = (np.min(plots[0][2]), np.max(plots[0][2]))
    axs = []

    if _3d:
        for i, (title, x, y) in enumerate(plots):
            axs.append(fig.add_subplot(*subplot_shape, i+1, projection='3d'))
            axs[i].set_title(title)
            if isinstance(y, list):
                for _y, *label in y:
                    ylim = (min(np.min(_y), ylim[0]), max(np.max(_y), ylim[1]))
                    if len(label) == 1:
                        axs[i].plot_trisurf(x[:,0], x[:,1], _y, cmap=cm.coolwarm, linewidth=0, antialiased=False, label=label[0])
                    else:
                        axs[i].plot_trisurf(x[:,0], x[:,1], _y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            else:
                axs[i].plot_trisurf(x[:,0], x[:,1], y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ylim = (min(np.min(y), ylim[0]), max(np.max(y), ylim[1]))

        for ax in axs:
            ax.set_zlim(*ylim)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            ax.view_init(elev=15, azim=init_azim)
            ax.legend()

        if animation:
            frames = 10
            def frame(i):
                print(f"Drawing frame {i+1}/{frames}")
                axs[0].view_init(elev=15, azim=init_azim+i*36)
                axs[1].view_init(elev=15, azim=init_azim+i*36)
                return fig

            animation = FuncAnimation(fig, frame, frames=range(frames), interval=100)
            animation.save(f"../plots/animations/{filename}.mp4")
        else:
            plt.savefig(f"../plots/{filename}.png")
    else:
        for i, (title, x, y) in enumerate(plots):
            axs.append(fig.add_subplot(*subplot_shape, i+1))
            axs[i].set_title(title)
            if isinstance(y,  list):
                for _y, *label in y:
                    ylim = (min(np.min(_y), ylim[0]), max(np.max(_y), ylim[1]))
                    if len(label) == 1:
                        axs[i].plot(x, _y, label=label[0])
                    else:
                        axs[i].plot(x, _y)
            else:
                axs[i].plot(x, y)
                ylim = (min(np.min(y), ylim[0]), max(np.max(y), ylim[1]))

        for ax in axs:
            ax.set_ylim(*ylim)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.legend()

        plt.savefig(f"../plots/{filename}.png")

def validation_errors(val_errors, bias_variance=False, animation=True, fig_prescript=''):
    """Plots the data from dataframe val_errors, giving an animation with error vs complexity vs lambda.

    Parameters:
    -----------
    val_errors: pandas dataframe
                As created by tune.Tune.validate.
    bias_variance:
                bool
                Set to true to plot bias-variance vs complexity vs lambda,
                instead of MSE from kfold and bootstrap vs  complexity vs lambda
    animation:  bool
                Set to false to plot a classic grid-search style still image instead of an animation.
    fig_prescript:
                str
                Text put on the beggining of filename.
    """

    for model, model_val_errors in val_errors.groupby(level='Model'):
        #_lambdas = np.unique(val_errors.index.get_level_values('Lambda').values)

        polys = model_val_errors.columns.values

        if animation:
            if model == "OLS":
                continue
            plt.clf()
            plt.cla()
            plt.close()
            fig = plt.figure(figsize=(7,7))

            _lambdas = val_errors.loc['Boot MSE', model][polys[0]].index.values

            if bias_variance:

                y_min = np.min((val_errors.loc['Boot Bias', model].min().min(), val_errors.loc['Boot Var', model].min().min()))
                y_max = val_errors.loc['Boot MSE', model].max().max()

                bootstrap_lines = plt.plot( val_errors.loc['Boot MSE', model][polys[0]].index.values,
                                            val_errors.loc['Boot MSE', model][polys[0]].values,
                                            c='midnightblue',
                                            label="Bootstrap MSE")

                bootstrap_bias_lines = plt.plot(val_errors.loc['Boot Bias', model][polys[0]].index.values,
                                                val_errors.loc['Boot Bias', model][polys[0]].values,
                                                c='royalblue',
                                                label="Bootstrap bias")
                bootstrap_variance_lines = plt.plot(val_errors.loc['Boot Var', model][polys[0]].index.values,
                                                    val_errors.loc['Boot Var', model][polys[0]].values,
                                                    c='mediumblue',
                                                    label="Bootstrap variance")
                
                lines = [bootstrap_lines, bootstrap_bias_lines, bootstrap_variance_lines]

            else:
                y_min = np.min((val_errors.loc['Boot MSE', model].min().min(), val_errors.loc['Kfold MSE', model].min().min()))
                y_max = np.max((val_errors.loc['Boot MSE', model].max().max(), val_errors.loc['Kfold MSE', model].max().max()))

                bootstrap_lines = plt.plot(val_errors.loc['Boot MSE', model][polys[0]].index.values,
                                           val_errors.loc['Boot MSE', model][polys[0]].values,
                                           c='midnightblue',
                                           label="Bootstrap MSE")

                kfold_lines = plt.plot(val_errors.loc['Kfold MSE', model][polys[0]].index.values,
                                       val_errors.loc['Kfold MSE', model][polys[0]].values,
                                       c='darkred',
                                       label="k-fold estimate")
               
                lines = [bootstrap_lines, kfold_lines]

            plt.axis((_lambdas[0], _lambdas[-1], y_min, y_max))
            plt.title(f"p = {model_val_errors.columns.values[0]}")
            plt.ylabel("MSE")
            plt.xlabel("$\\lambda$")
            plt.legend()
            plt.xscale("log")

            def frame(p):
                plt.title(f"p = {p}")

                if bias_variance:
                    bootstrap_bias_lines[0].set_ydata(val_errors.loc['Boot Bias', model][p].values)
                    bootstrap_variance_lines[0].set_ydata(val_errors.loc['Boot Var', model][p].values)
                else:
                    kfold_lines[0].set_ydata(val_errors.loc['Kfold MSE', model][p].values)

                bootstrap_lines[0].set_ydata(val_errors.loc['Boot MSE', model][p].values)
                
                return lines

            animation = FuncAnimation(fig, frame, frames=polys[1:], interval=500, repeat_delay=2000)
            animation.save(f"../plots/animations/{fig_prescript}_tune_{model}_{['bootstrapKfold', 'biasVariance'][int(bias_variance)]}.mp4")

        else:
            for metric in ['Kfold MSE', 'Boot MSE']:
                fig = plt.figure(figsize=(10,7))
                plt.imshow(np.log10(val_errors.loc[metric, model]), cmap=plt.cm.plasma)
                plt.xlabel('Max polynomials')
                plt.ylabel('$\\lambda$')
                plt.colorbar()
                plt.xticks(np.arange(len(polys)), polys)
                plt.yticks(np.arange(len(val_errors.loc['Kfold MSE', model].index.values)), [f"{err:.2e}" for err in val_errors.loc['Kfold MSE', model].index.values])
                plt.title(f'Grid Search {metric} for {model}')
                plt.grid(alpha=0.2)
                plt.savefig(f"../plots/{fig_prescript}_gridsearch_{model}_{metric}.png")
                plt.close()

def validation_test(name):
    """Plots validation MSE against test MSE, in a scatter plot.

    Parameters:
    -----------
    name:       str
                Name of dataset that tune_and_evaluate.Tune().validate(test_as_well=True) has been run on.
    """

    val_error_columns = pd.read_csv(f'../dataframe/{name}_validation_errors.csv', nrows = 1)
    val_errors = pd.read_csv(f'../dataframe/{name}_validation_errors.csv', names=val_error_columns.columns, index_col=[0,1,2], skiprows=1)

    boot_errors = val_errors.loc['Boot MSE'].values.reshape(-1)
    kfold_errors = val_errors.loc['Kfold MSE'].values.reshape(-1)
    test_errors = val_errors.loc['Test MSE'].values.reshape(-1)
    
    # Mask out outliers
    sorted_x = np.sort(boot_errors)
    sorted_y = np.sort(kfold_errors)
    m_boot = boot_errors < sorted_x[7*boot_errors.size//8]
    m_kfold = kfold_errors < sorted_y[7*kfold_errors.size//8]
    mask = np.logical_and(m_boot, m_kfold)
    
    boot_errors = boot_errors[mask]
    kfold_errors = kfold_errors[mask]
    test_errors = test_errors[mask]

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(boot_errors, kfold_errors, c=test_errors, cmap="RdBu_r")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel("Test MSE")

    plt.title("Test and validation error")
    plt.xlabel("Bootstrap MSE")
    plt.ylabel("k-fold MSE")
    plt.savefig(f"../plots/test_validation_error_{name}.png")

if __name__ == '__main__':
    franke_val_error_columns = pd.read_csv('../dataframe/franke_validation_errors.csv', nrows = 1)
    franke_val_errors = pd.read_csv('../dataframe/franke_validation_errors.csv', names=franke_val_error_columns.columns, index_col=[0,1,2], skiprows=1)
    validation_errors(franke_val_errors, fig_prescript="franke", bias_variance=True)
    validation_errors(franke_val_errors, fig_prescript="franke", bias_variance=False)
    validation_errors(franke_val_errors, fig_prescript="franke", animation=False)

    real_val_error_columns = pd.read_csv('../dataframe/real_validation_errors.csv', nrows = 1)
    real_val_errors = pd.read_csv('../dataframe/real_validation_errors.csv', names=real_val_error_columns.columns, index_col=[0,1,2], skiprows=1)
    validation_errors(real_val_errors, fig_prescript="real", bias_variance=True)
    validation_errors(real_val_errors, fig_prescript="real", bias_variance=False)
    validation_errors(real_val_errors, fig_prescript="real", animation=False)

    validation_test('franke')
    validation_test('real')


def plot_MSE_and_CI(MSE, lb, ub, color_MSE="C0", color_shading="C0"):
    """Plots line with confidence interval, figure must be saved or shown after function call.

    Parameters:
    -----------
    line:       array/list
                Plots centre line
    lb:         array/list
                Lower bound of confidence interval
    ub:         array/list
                Upper bound of confidence interval
    color_line: str
                Colour of
    """
    plt.fill_between(range(1,MSE.shape[0]+1), ub, lb,
                     color=color_shading, alpha=.5)
    plt.plot(MSE, color_MSE)

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
            [x0, y0], width, height, facecolor=self.facecolor)
        handlebox.add_artist(patch)

        return patch