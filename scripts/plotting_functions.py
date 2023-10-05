import mne
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

"""
Plotting functions. It stores functions to plot temporal decoding (for both TF power and voltage data),
temporal generalization matrices, compare evoked activities and represent statistical comparisons
between TF power and voltage. 
"""

def plotTemporalDecoding(data, se, times, color, color_se, data_type, title, save_figure):
    """ 
    Function to plot temporal decoding scores for TF power and voltage data. It adds dots to represent
    statistical comparisons between both datasets at each time point. Does not show the plot,
    saves it to a .png file.

    Args:
        data (array): decoding scores for the TF power/voltage data. Must be an array of averaged 
        data for all subjects
        se (int): mean SE for all subjects in the sample (used to plot a shaded area)
        times (array): temporal data
        color (str): Matplotlib named color for the plotted lines
        color_se (str): Matplotlib named color for the shaded areas (SE)
        data_type (str): must be either 'volt' or 'tf'
        title (str): plot title
        save_figure (str): directory where we want to save the plot (.png file)
    """
    fig, ax = plt.subplots()
    if data_type == 'volt':
        ax.plot(times, signal.savgol_filter(data,9,3), color=color label='Decoding scores')
    else:
        ax.plot(times, data, color=color, label='Decoding scores')
    ax.fill_between(times, (data + se), (data - se), color = color_se, alpha=0.5)
    ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth = 0.5)
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-', linewidth = 0.5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title)
    fig.set_size_inches(15, 10)
    plt.savefig(save_figure)
    plt.close()
    
def plotTemporalGeneralization(data, times, title, figures_path, color_signal, color_smooth):
    fig, ax = plt.subplots()
    ax.plot(times, np.diag(data), color=color_signal, label='score')
    ax.plot(times, signal.savgol_filter(np.diag(data),9,3), color=color_smooth, label='smoothed')
    ax.axhline(0.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title(title)
    plt.savefig(figures_path)
    plt.close()    

def plotGeneralizationMatrix(data, times, title, figures_path):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data, interpolation='lanczos', origin='lower', cmap='Reds',
                    extent=times[[0, -1, 0, -1]], vmin=0.5)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC') 
    plt.savefig(figures_path)
    plt.close()

def plotCompareEvokeds(data, picks, color_dict, axes, combine, title, figures_path):
    mne.viz.plot_compare_evokeds(data, # data file to plot
                                  combine=combine, # can be 'mean' for multiple channels, or None
                                  legend='upper right',
                                  picks=picks,  # channel number or 'eeg' to plot all
                                  show_sensors='lower right',
                                  cmap=color_dict, # colormap, can be conditions dictionary
                                  title=title,
                                  axes=axes) # 'topo' for topographical map, if None one figure per channel
    plt.savefig(figures_path)
    plt.close()
    
def plot_stats_tfvsvoltage(data_tf, data_volt, se_tf, se_volt, color_tf, color_volt, color_tf_se, 
                            color_volt_se, times, sign_points, x_distance, title, save_figure):
    """ 
    Function to plot temporal decoding scores for TF power and voltage data. It adds dots to represent
    statistical comparisons between both datasets at each time point. Does not show the plot,
    saves it to a .png file.

    Args:
        data_tf (array), data_volt (array): decoding scores for the TF power/voltage data. 
        Must be an array of averaged data for all subjects
        se_tf (int), se_volt (int): mean SE for all subjects in the sample. Used to plot
        a shaded area
        color_tf (str), color_volt (str): Matplotlib named color for the plotted lines 
        color_tf_se (str), color_volt_se (str): Matplotlib named color for the shaded areas (SE)
        times (array): temporal data
        sign_points (array): time points where the comparison is significant 
        x_distance (int): at which distance from the X-axis we want the dots to be plotted
        title (str): plot title
        save_figure (str): directory where we want to save the plot (.png file)
    """
    fig, ax = plt.subplots()
    ax.plot(times, signal.savgol_filter(data_volt,9,3), color_volt, label='Voltage')
    ax.fill_between(times, (data_volt + se_volt), (data_volt - se_volt), color = color_volt_se, alpha=0.5)
    ax.plot(times, data_tf, color_tf, label='TF power')
    ax.fill_between(times, (data_tf + se_tf), (data_tf - se_tf), color = color_tf_se, alpha=0.5)
    ax.axhline(0.5, color='k', linestyle='--', label='Chance level', linewidth = 0.5)
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-', linewidth = 0.5)
    for i in sign_points:
        plot_points = [y_distance]*1024
        ax.scatter(times[i], plot_points[i], color = color_tf, alpha=0.5, s = 20)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title)
    fig.set_size_inches(15, 10)
    plt.savefig(save_figure)
    plt.close()