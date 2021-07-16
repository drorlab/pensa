import numpy as np
from pensa import *


    """
    Calculates the average and maximum Jensen-Shannon distance and the Kullback-Leibler divergences for each feature from two ensembles. Each of four functions uses the relative_entropy_analysis function with the same parameters.

    Parameters
    ----------
        features_a : list of str
            Feature names of the first ensemble.
            Can be obtained from features object via .describe().
        features_b : list of str
            Feature names of the first ensemble.
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check.
        all_data_a : float array
            Trajectory data from the first ensemble. Format: [frames,frame_data].
        all_data_b : float array
            Trajectory data from the second ensemble. 
            For kld functions, the second ensemble should be the reference ensemble.
            Format: [frames,frame_data].
        bin_width : float, default=None
            Bin width for the axis to compare the distributions on.
            If bin_width is None, bin_num (see below) bins are used and the width is determined from the common histogram.
        bin_num : int, default=10
            Number of bins for the axis to compare the distributions on (only if bin_width=None).
        verbose : bool, default=True
            Print intermediate results.
        override_name_check : bool, default=False
            Only check number of features, not their names.

    Returns
    -------
        Each function returns one value.
        
        average_jsd : float
            Average Jensen-Shannon distance from two ensembles.
        max_jsd : float
            Maximum Jensen-Shannon distance from two ensembles.
        average_kld : float
            Average Kullback-Leibler divergence from two ensembles.
        max_kld : float
            Maximum Kullback-Leibler divergence from two ensembles.
    """

def average_jsd(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, verbose=True, override_name_check=False):
    _, data_jsdist, _, _ = relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=bin_width, bin_num=bin_num, verbose=verbose, override_name_check=override_name_check)
    return np.mean(data_jsdist)


def max_jsd(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, verbose=True, override_name_check=False):
    _, data_jsdist, _, _ = relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=bin_width, bin_num=bin_num, verbose=verbose, override_name_check=override_name_check)
    return np.max(data_jsdist)


def average_kld(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, verbose=True, override_name_check=False):
    _, _, data_kld_ab, _ = relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=bin_width, bin_num=bin_num, verbose=verbose, override_name_check=override_name_check)
    return np.mean(data_kld_ab)


def max_kld(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, verbose=True, override_name_check=False):
    _, _, data_kld_ab, _ = relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=bin_width, bin_num=bin_num, verbose=verbose, override_name_check=override_name_check)
    return np.max(data_kld_ab)


    """
    Calculates the average and maximum Kolmogorov-Smirnov statistic for two distributions. Each of two functions uses the kolmogorov_smirnov_analysis function with the same parameters. 

    Parameters
    ----------
        features_a : list of str
            Feature names of the first ensemble.
            Can be obtained from features object via .describe().
        features_b : list of str
            Feature names of the first ensemble.
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check.
        all_data_a : float array
            Trajectory data from the first ensemble. Format: [frames,frame_data].
        all_data_b : float array
            Trajectory data from the second ensemble. Format: [frames,frame_data].
        verbose : bool, default=True
            Print intermediate results.
        override_name_check : bool, default=False
            Only check number of features, not their names.

    Returns
    -------
        Each function returns one value.

        average_kss : float
            Average Kolmogorov-Smirnov statistic for two distributions.
        max_kss : float
            Maximum Kolmogorov-Smirnov statistic for two distributions.
    """


def average_kss(features_a, features_b, all_data_a, all_data_b, verbose=True, override_name_check=False):
    _, data_kss, _ = kolmogorov_smirnov_analysis(features_a, features_b, all_data_a, all_data_b, verbose=verbose, override_name_check=override_name_check)
    return np.mean(data_kss)


def max_kss(): 
    _, data_kss, _ = kolmogorov_smirnov_analysis(features_a, features_b, all_data_a, all_data_b, verbose=verbose, override_name_check=override_name_check)
    return np.max(data_kss)


    """
    Calculates average and maximum State Specific Information statistic for a feature across two ensembles. Each of two functions uses the ssi_ensemble_analysis function with the same parameters. 
    
    Parameters
    ----------
    features_a : list of str
        Feature names of the first ensemble. 
    features_b : list of str
        Feature names of the first ensemble. 
        Must be the same as features_a. Provided as a sanity check. 
    all_data_a : float array
        Trajectory data from the first ensemble. Format: [frames,frame_data].
    all_data_b : float array
        Trajectory data from the second ensemble. Format: [frames,frame_data].
    torsions : str
        Torsion angles to use for SSI, including backbone - 'bb', and sidechain - 'sc'. 
        Default is None.
    pocket_occupancy : bool, optional
        Set to 'True' if the data input is pocket occupancy distribution.
        The default is None.
    pbc : bool, optional
        If true, the apply periodic bounary corrections on angular distribution inputs.
        The input for periodic correction must be radians. The default is True.
    verbose : bool, default=True
        Print intermediate results.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    override_name_check : bool, default=False
        Only check number of features, not their names.   
        
    Returns
    -------
        Each function returns one value.

        average_ssi : float
            Average of State Specific Information for a feature across two ensembles.
        max_ssi : float
            Maximum of State Specific Information for a feature across two ensembles.
    """

def average_ssi(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbc=True, verbose=True, write_plots=None, override_name_check=False):
    _, data_ssi = ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=torsions, pocket_occupancy=pocket_occupancy, pbc=pbc, verbose=verbose, write_plots=write_plots, override_name_check=override_name_check)
    return np.mean(data_ssi)


def max_ssi(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbs=True, verbose=True, write_plots=None, override_name_check=False):
    _, data_ssi = ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=torsions, pocket_occupancy=pocket_occupancy, pbc=pbc, verbose=verbose, write_plots=write_plots, override_name_check=override_name_check)
    return np.max(data_ssi)
