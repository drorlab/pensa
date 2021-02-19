import numpy as np
import pyemma
from pyemma.util.contexts import settings



# -- Loading the Features --


def get_features(pdb, xtc, start_frame=0, step_width=1, cossin=False,
                 features=['bb-torsions','sc-torsions','bb-distances']):
    """
    Load the features. Currently implemented: bb-torsions, sc-torsions, bb-distances
    http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html
    
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
        
    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features
    
    """
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
    # Add backbone torsions.
    if 'bb-torsions' in features:
        bbtorsions_feat = pyemma.coordinates.featurizer(pdb)
        bbtorsions_feat.add_backbone_torsions(cossin=cossin, periodic=False)
        bbtorsions_data = pyemma.coordinates.load(xtc, features=bbtorsions_feat, stride=step_width)[start_frame:]
        feature_names['bb-torsions'] = bbtorsions_feat.describe()
        features_data['bb-torsions'] = bbtorsions_data
    # Add sidechain torsions.
    if 'sc-torsions' in features:
        sctorsions_feat = pyemma.coordinates.featurizer(pdb)
        sctorsions_feat.add_sidechain_torsions(cossin=cossin, periodic=False)
        sctorsions_data = pyemma.coordinates.load(xtc, features=sctorsions_feat, stride=step_width)[start_frame:]
        feature_names['sc-torsions'] = sctorsions_feat.describe()
        features_data['sc-torsions'] = sctorsions_data
    # Add backbone C-alpha distances.
    if 'bb-distances' in features:
        bbdistances_feat = pyemma.coordinates.featurizer(pdb)
        bbdistances_feat.add_distances(bbdistances_feat.pairs(bbdistances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
        bbdistances_data = pyemma.coordinates.load(xtc, features=bbdistances_feat, stride=step_width)[start_frame:]
        feature_names['bb-distances'] = describe_dist_without_atom_numbers(bbdistances_feat)
        features_data['bb-distances'] = bbdistances_data
    # Return the dictionaries.
    return feature_names, features_data



# -- Utilities to process the features


def remove_atom_numbers_from_distance(feat_str):
    """
    Remove atom numbers from a distance feature string.
    
    Parameters
    ----------
        feat_str : str
            The string describing a single feature.
    
    Returns
    -------
        new_feat : str
            The feature string without atom numbers.
    
    """
    # Split the feature string in its parts
    parts = feat_str.split(' ')
    # Glue the desired parts back together
    new_feat = parts[0]
    for nr in [1,2,3,5,6,7,8]:
        new_feat += ' '+parts[nr]
    return new_feat


def describe_dist_without_atom_numbers(feature_names):
    """
    Provides feature descriptors without atom numbers.
    
    Parameters
    ----------
        feature_names : dict
            Names of all features (assumes distances).
    
    Returns
    -------
        desc : list of str
            The feature descriptors without atom numbers.
    
    """
    desc = feature_names.describe()
    desc = [ remove_atom_numbers_from_distance(d) for d in desc ]
    return desc



# -- Utilities to sort the features


def sort_sincos_torsions_by_resnum(tors, data):
    """
    Sort sin/cos of torsion features by the residue number..

    Parameters
    ----------
        tors : list of str
            The list of torsion features.

    Returns
    -------
        new_tors : list of str
            The sorted list of torsion features.

    """
    renamed = []
    for t in tors:
        rn = t.split(' ')[-1].replace(')','')
        ft = t.split(' ')[0].replace('(',' ')
        sincos, angle = ft.split(' ')
        renamed.append('%09i %s %s'%(int(rn),angle,sincos))
    new_order = np.argsort(renamed)
    new_tors = np.array(tors)[new_order].tolist()
    new_data = data[:,new_order]
    return new_tors, new_data


def sort_distances_by_resnum(dist, data):
    """
    Sort distance features by the residue number..

    Parameters
    ----------
        dist : list of str
            The list of distance features.

    Returns
    -------
        new_dist : list of str
            The sorted list of distance features.

    """
    renamed = []
    for d in dist:
        rn1, at1, rn2, at2 = np.array(d.split(' '))[np.array([2,3,6,7])]
        renamed.append('%09i %s %09i %s'%(int(rn1),at1,int(rn2),at2))
    new_order = np.argsort(renamed)
    new_dist = np.array(dist)[new_order].tolist()
    new_data = data[:,new_order]
    return new_dist, new_data


