import numpy as np
import matplotlib.pyplot as plt

from pensa.dimensionality import \
    get_components_pca, \
    get_components_tica
from .statistics import \
    feature_correlation


def pca_feature_correlation(features, data, pca=None, num=3, threshold=0.1, plot_file=None, add_labels=False):
    """
    Calculates and plots the correlation between principal components and the underlying features.
    Prints all features with a correlation above the threshold.

    Parameters
    ----------
        features : list of str
            Names of the features for which the PCA was/is performed.
        data : float array
            Trajectory data for the features. Format: [frames, frame_data]
        pca : PCA obj, default = None
            The PCA of which to plot the features.
            If no PCA is provided, it is calculated from the trajectory.
        num : float, default = 3
            Number of feature correlations to plot.
        threshold : float, default = 0.1
            Features with a correlation above this will be printed.
        plot_file : str, optional, default = None
            Path and name of the file to save the plot.
        add_labels : bool, optional, default = False
            Add labels of the features to the x axis.


    """
    # Project the trajectory data on the principal components
    # (performs PCA if none is provided)
    projection = get_components_pca(data, num, pca)[1]
    # Plot the highest PC correlations and print relevant features
    test_graph = []
    test_corr = []
    height = num * 2 + 2 if add_labels else num * 2
    fig, ax = plt.subplots(num, 1, figsize=[4, height], dpi=300, sharex=True)
    pca_feature_PC_correlation = feature_correlation(data, projection)
    for i in range(num):
        relevant = pca_feature_PC_correlation[:, i]**2 > threshold**2
        print("Features with abs. corr. above a threshold of %3.1f for PC %i:" % (
            threshold, i + 1))
        for j, ft in enumerate(features):
            if relevant[j]:
                print(ft, "%6.3f" % (pca_feature_PC_correlation[j, i]))
                test_corr.append(pca_feature_PC_correlation[j, i])
        ax[i].bar(np.arange(len(features)), pca_feature_PC_correlation[:, i])
        ax[i].set_ylabel('corr. with PC%i' % (i + 1))
        test_graph.append(pca_feature_PC_correlation[:, i])
    if add_labels:
        ax[-1].set_xticks(np.arange(len(features)))
        ax[-1].set_xticklabels(features, rotation=90)
    else:
        ax[-1].set_xlabel('feature index')
    fig.tight_layout()
    # Save the figure to a file
    if plot_file:
        fig.savefig(plot_file, dpi=300)
    return test_graph, test_corr


def tica_feature_correlation(features, data, num=3, tica=None, threshold=0.1, plot_file=None, add_labels=False):
    """
    Prints relevant features and plots feature correlations.

    Parameters
    ----------
        features : list of str
            Features for which the TICA was performed.
        num : float
            Number of feature correlations to plot.
        threshold : float
            Features with a correlation above this will be printed.
        plot_file : str, optional, default = None
            Path and name of the file to save the plot.
        add_labels : bool, optional, default = False
            Add labels of the features to the x axis.

    """
    # Plot the highest TIC correlations and print relevant features.
    height = num * 2 + 2 if add_labels else num * 2
    fig, ax = plt.subplots(num, 1, figsize=[4, height], dpi=300, sharex=True)
    for i in range(num):
        relevant = tica.feature_component_correlation[:, i] ** 2 > threshold ** 2
        print("Features with abs. corr. above a threshold of %3.1f for TIC %i:" % (
            threshold, i + 1))
        for j, ft in enumerate(features):
            if relevant[j]:
                print(ft, "%6.3f" % (tica.feature_component_correlation[j, i]))
        ax[i].plot(tica.feature_component_correlation[:, i])
        test_feature = tica.feature_component_correlation[:, i]
        ax[i].set_ylabel('corr. with TIC%i' % (i + 1))
    if add_labels:
        ax[-1].set_xticks(np.arange(len(features)))
        ax[-1].set_xticklabels(features, rotation=90)
    else:
        ax[-1].set_xlabel('feature index')
    fig.tight_layout()
    # Save the figure to a file.
    if plot_file:
        fig.savefig(plot_file, dpi=300)
    return test_feature
