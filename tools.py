import scipy as sp
import scipy.io.wavfile
import scipy.signal
import numpy as np
from matplotlib import pyplot as plt

## Util functions

np.set_printoptions(suppress=True)

def todB(array):
    mean = np.mean(array)
    return 10 * np.log10(array / mean)

def plotSpectrogram(to_plot, yticks = np.array([]), cmap = plt.cm.gist_heat, subplot=111, _fig=False, fig_size=(15,7)):
    if (_fig == False):
        fig = plt.figure(figsize=fig_size)
    else:
        fig = _fig
    ax = fig.add_subplot(subplot)
    plt.imshow(to_plot, cmap=cmap)
    ax.set_aspect('auto')

    ## making pretty
    ax.set_ylim(ax.get_ylim()[::-1]) ## reversing axis
    plt.title("Spectrogram in the human hearing range")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    num_labels = 10
    xpoints = np.arange(to_plot.shape[1], step=int(to_plot.shape[1]/num_labels))
    plt.xticks(xpoints, np.round(t_points[xpoints], 1))
    if (yticks.size > 0):
        ypoints = np.arange(to_plot.shape[0], step=int(to_plot.shape[0]/num_labels))
        plt.yticks(ypoints, yticks[ypoints])
    plt.colorbar(orientation='vertical')
    return fig

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x,y])

vpol2cart = np.vectorize(pol2cart)

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points