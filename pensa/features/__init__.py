# - * - coding: utf-8 - * -
"""
Methods to read in and process features from coordinates.

"""

from .mda_distances import \
    get_atom_group_distances, \
    get_atom_self_distances, \
    get_calpha_distances, \
    get_gpcr_calpha_distances, \
    select_gpcr_residues

from .mda_torsions import \
    get_torsions, \
    get_protein_backbone_torsions, \
    get_protein_sidechain_torsions, \
    get_nucleicacid_backbone_torsions, \
    get_nucleicacid_pseudotorsions

from .mda_combined import \
    get_structure_features, \
    sort_traj_along_combined_feature

from .processing import \
    get_feature_subset, \
    get_feature_data, \
    get_common_features_data, \
    get_feature_timeseries, \
    get_multivar_res, \
    get_multivar_res_timeseries, \
    correct_angle_periodicity, \
    correct_spher_angle_periodicity, \
    select_common_features, match_sim_lengths, \
    sort_features, \
    sort_features_alphabetically, \
    sort_distances_by_resnum, \
    sort_sincos_torsions_by_resnum, \
    sort_torsions_by_resnum, \
    sort_traj_along_feature

from .atom_features import \
    get_atom_features

from .water_features import \
    get_water_features

from .txt_features import \
    get_txt_features_ala2

from .csv_features import \
    get_drormd_features, \
    read_csv_features

from .hbond_features import \
    get_h_bonds, \
    get_cavity_bonds
