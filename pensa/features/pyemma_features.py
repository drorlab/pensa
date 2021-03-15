# -*- coding: utf-8 -*-
"""
Methods to featurize a protein, based on PyEMMA.

More options are described here:
http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html
"""

import warnings
import numpy as np
import pyemma
from pyemma.util.contexts import settings



# -- Loading the Features --


def get_structure_features(pdb, xtc, start_frame=0, step_width=1, cossin=False,
                           features=['bb-torsions','sc-torsions','bb-distances'],
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
        feature_names['bb-distances'] = _describe_dist_without_atom_numbers(bbdistances_feat)
        features_data['bb-distances'] = bbdistances_data
    # Remove the residue-number offset
    if resnum_offset != 0:
        feature_names = _remove_resnum_offset(feature_names,resnum_offset)
    # Return the dictionaries.
    return feature_names, features_data


def _describe_dist_without_atom_numbers(feature_names):
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
    desc = [ _remove_atom_numbers_from_distance(d) for d in desc ]
    return desc


def _remove_atom_numbers_from_distance(feat_str):
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
            resnum = int(f.split(' ')[3])-offset
            fsplit[3] = str(resnum)
            new_features['bb-torsions'].append(' '.join(fsplit))
    
    if 'sc-torsions' in features.keys():
        for f in features['sc-torsions']:
            fsplit = f.split(' ')
            resnum = int(f.split(' ')[3])-offset
            fsplit[3] = str(resnum)
            new_features['sc-torsions'].append(' '.join(fsplit))
        
    if 'bb-distances' in features.keys():
        for f in features['bb-distances']:
            fsplit = f.split(' ')
            resnum1 = int(f.split(' ')[2])-offset
            resnum2 = int(f.split(' ')[6])-offset
            fsplit[2] = str(resnum1)
            fsplit[6] = str(resnum2)
            new_features['bb-distances'].append(' '.join(fsplit))
        
    return new_features
    
    

