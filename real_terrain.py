from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import data_handling
import plotting

def get_data(*data_dict_args, random=False):
    """Returns terrain data dict with training, validation and testing data based on grid-size for training data.

    Data is real terrain measurements from Ireland. 

    Parameters:
    -----------
    x_sparsity: int
                Grid size for selection of training and validation data. Must be at least 2.
    
    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data.
    """
    #y = imread('ireland2.tif')[3200:3600,1400:1800]
    y = imread('ireland2.tif')[3200:3600,1400:1800]
    return data_handling.make_data_dict(y, *data_dict_args, random=random)

def plot_data(*data_args, random=False):
    """Plots terrain data created by get_data function.

    Parameters:
    -----------
    *data_args:  
                Passed on to get_data()

    """
    data = get_data(*data_args, random=random)

    plotting.side_by_side(
        ['Train', data['x_train'], data['y_train']],
        ['Validate', data['x_train'], data['y_train']],
        ['Test', data['x_test'], data['y_test']],
        title=f"Training, validation and testing data for real terrain",
        filename=f"real_terrain")

if __name__ == '__main__':
    plot_data(20)