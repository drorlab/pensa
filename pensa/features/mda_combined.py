from .mda_torsions import read_protein_backbone_torsions, read_protein_sidechain_torsions
from .mda_distances import read_calpha_distances
from pensa.features.processing import get_feature_timeseries
from pensa.preprocessing.coordinates import sort_coordinates


# MDAnalysis-based reimplementation of the old standard feature loader
# The old feature loader is now called get_pyemma_features
#
# Note: It only loads protein features
#

def read_structure_features(pdb, xtc, start_frame=0, step_width=1, cossin=False,
                           features=['bb-torsions', 'sc-torsions', 'bb-distances'],
                           resnum_offset=0):
    """
    Load the features. Currently implemented: bb-torsions, sc-torsions, bb-distances

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    start_frame : int, default=0
        First frame to return of the features. Already takes subsampling by stride>=1 into account.
    step_width : int, default=1
        Subsampling step width when reading the frames.
    cossin : bool, default=False
        Use cosine and sine for angles.
    features : list of str, default=['bb-torsions', 'sc-torsions']
        Names of the features to be extracted.
    resnum_offset : int, default=0
        Number to subtract from the residue numbers that are loaded from the reference file.

    Returns
    -------
    feature_names : dict of lists of str
        Names of all features
    features_data : dict of numpy arrays
        Data for all features

    """
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
    # Add backbone torsions.
    if 'bb-torsions' in features:
        bbtorsions = read_protein_backbone_torsions(
            pdb, xtc, selection='all',
            first_frame=start_frame, last_frame=None, step=step_width,
            naming='segindex', radians=True,
            include_omega=False
        )
        feature_names['bb-torsions'] = bbtorsions[0]
        features_data['bb-torsions'] = bbtorsions[1]
    # Add sidechain torsions.
    if 'sc-torsions' in features:
        sctorsions = read_protein_sidechain_torsions(
            pdb, xtc, selection='all',
            first_frame=start_frame, last_frame=None, step=step_width,
            naming='segindex', radians=True
        )
        feature_names['sc-torsions'] = sctorsions[0]
        features_data['sc-torsions'] = sctorsions[1]
    # Add backbone C-alpha distances.
    if 'bb-distances' in features:
        bbdistances = read_calpha_distances(
            pdb, xtc,
            first_frame=start_frame, last_frame=None, step=step_width,
        )
        feature_names['bb-distances'] = bbdistances[0]
        features_data['bb-distances'] = bbdistances[1]
    # Remove the residue-number offset
    if resnum_offset != 0:
        feature_names = _remove_resnum_offset(feature_names, resnum_offset)
    # Return the dictionaries.
    return feature_names, features_data


def _remove_resnum_offset(features, offset):
    """
    Removes (subtracts) the offset from residue numbers in PyEMMA structure features.

    Parameters
    ----------
    features : list
        The feature names to be modified.
    offset : int
        The number to subtract from the residue numbers.

    Returns
    -------
    new_feastures : str
        The feature names without the offset.

    """
    new_features = {}
    for key in features.keys():
        new_features[key] = []

    if 'bb-torsions' in features.keys():
        for f in features['bb-torsions']:
            fsplit = f.split(' ')
            resnum = int(f.split(' ')[3]) - offset
            fsplit[3] = str(resnum)
            new_features['bb-torsions'].append(' '.join(fsplit))

    if 'sc-torsions' in features.keys():
        for f in features['sc-torsions']:
            fsplit = f.split(' ')
            resnum = int(f.split(' ')[3]) - offset
            fsplit[3] = str(resnum)
            new_features['sc-torsions'].append(' '.join(fsplit))

    if 'bb-distances' in features.keys():
        for f in features['bb-distances']:
            fsplit = f.split(' ')
            resnum1 = int(f.split(' ')[2]) - offset
            resnum2 = int(f.split(' ')[6]) - offset
            fsplit[2] = str(resnum1)
            fsplit[6] = str(resnum2)
            new_features['bb-distances'].append(' '.join(fsplit))

    return new_features


def sort_traj_along_combined_feature(feat, data, feature_name, feature_type,
                                     ref_name, trj_name, out_name,
                                     start_frame=0):
    """
    Sort a trajectory along one feature in a combined set.

    Parameters
    ----------
        feat : list of str
            List with all feature names.
        data : float array
            Feature values data from the simulation.
        feature_name : str
            Name of the selected feature.
        feature_type : str
            Type of the selected feature.
        ref_name: string
            Reference topology for the trajectory.
        trj_name: string
            Trajetory from which the frames are picked.
            Usually the same as the values are from.
        out_name: string.
            Name of the output files.
        start_frame: int
            Offset of the data with respect to the trajectories.

    Returns
    -------
        d_sorted: float array
            Sorted data of the selected feature.

    """
    d = get_feature_timeseries(feat, data, feature_type, feature_name)
    sort_idx, oidx_sort = sort_coordinates(d, ref_name, trj_name, out_name, start_frame=start_frame)
    d_sorted = d[sort_idx]
    return d_sorted
