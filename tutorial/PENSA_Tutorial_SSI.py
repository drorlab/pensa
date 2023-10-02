import os
from pensa import \
    download_from_gpcrmd, extract_coordinates, \
    extract_aligned_coords, extract_combined_grid, \
    get_structure_features, get_water_features, \
    get_multivar_res_timeseries, get_discrete_states, \
    ssi_ensemble_analysis, ssi_feature_analysis, cossi_featens_analysis \


# Define where to save the GPCRmd files
root_dir = './mor-data'
# Define which files to download
md_files = ['11427_dyn_151.psf', '11426_dyn_151.pdb',  # MOR-apo
            '11423_trj_151.xtc', '11424_trj_151.xtc', '11425_trj_151.xtc',
            '11580_dyn_169.psf', '11579_dyn_169.pdb',  # MOR-BU72
            '11576_trj_169.xtc', '11577_trj_169.xtc', '11578_trj_169.xtc']
# Download all the files that do not exist yet
for file in md_files:
    if not os.path.exists(os.path.join(root_dir, file)):
        download_from_gpcrmd(file, root_dir)

root_dir = './mor-data'
# Simulation A
ref_file_a = root_dir + '/11427_dyn_151.psf'
pdb_file_a = root_dir + '/11426_dyn_151.pdb'
trj_file_a = [root_dir + '/11423_trj_151.xtc',
              root_dir + '/11424_trj_151.xtc',
              root_dir + '/11425_trj_151.xtc']
# Simulation B
ref_file_b = root_dir + '/11580_dyn_169.psf'
pdb_file_b = root_dir + '/11579_dyn_169.pdb'
trj_file_b = [root_dir + '/11576_trj_169.xtc',
              root_dir + '/11577_trj_169.xtc',
              root_dir + '/11578_trj_169.xtc']
# Base for the selection string for each simulation
sel_base_a = "(not name H*) and protein"
sel_base_b = "(not name H*) and protein"
# Names of the output files
out_name_a = "traj/condition-a"
out_name_b = "traj/condition-b"

for subdir in ['traj', 'plots', 'vispdb', 'pca', 'clusters', 'results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# Extract the coordinates of the receptor from the trajectory
extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a + "_receptor", sel_base_a)
extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b + "_receptor", sel_base_b)

# Extract the features from the beginning (start_frame) of the trajectory
start_frame = 0
a_rec = get_structure_features(out_name_a + "_receptor.gro",
                               out_name_a + "_receptor.xtc",
                               start_frame)
a_rec_feat, a_rec_data = a_rec

b_rec = get_structure_features(out_name_b + "_receptor.gro",
                               out_name_b + "_receptor.xtc",
                               start_frame)
b_rec_feat, b_rec_data = b_rec

out_name_a = "condition-a"
out_name_b = "condition-b"

# Extract the multivariate torsion coordinates of each residue as a
# timeseries from the trajectory and write into subdirectory
# output = [[torsion 1 timeseries], [torsion 2 timeseries], ..., [torsion n timeseries]]
sc_multivar_res_feat_a, sc_multivar_res_data_a = get_multivar_res_timeseries(a_rec_feat, a_rec_data, 'sc-torsions', write=True, out_name=out_name_a)
sc_multivar_res_feat_b, sc_multivar_res_data_b = get_multivar_res_timeseries(b_rec_feat, b_rec_data, 'sc-torsions', write=True, out_name=out_name_b)

discrete_states_ab = get_discrete_states(sc_multivar_res_data_a['sc-torsions'], sc_multivar_res_data_b['sc-torsions'],
                                         discretize='gaussian', pbc=True)

# We can calculate the State Specific Information (SSI) shared between the
# ensemble switch and the combined ensemble residue conformations. As the ensemble
# is a binary change, SSI can exist within the range [0, 1] units=bits.
# 0 bits = no information, 1 bits = maximum information, i.e. you can predict the state of the ensemble with
# certainty from the state of the residue.
# Set write_plots = True to generate a folder with all the clustered states for each residue.
data_names, data_ssi = ssi_ensemble_analysis(sc_multivar_res_feat_a['sc-torsions'], sc_multivar_res_feat_b['sc-torsions'],
                                             sc_multivar_res_data_a['sc-torsions'], sc_multivar_res_data_b['sc-torsions'],
                                             discrete_states_ab,
                                             verbose=True, write_plots=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # Featurizing waters and calculating SSI

# First we preprocess the trajectories to extract coordinates for protein and waters.
root_dir = './mor-data'
# Simulation A
ref_file_a = root_dir + '/11427_dyn_151.psf'
pdb_file_a = root_dir + '/11426_dyn_151.pdb'
trj_file_a = [root_dir + '/11423_trj_151.xtc',
              root_dir + '/11424_trj_151.xtc',
              root_dir + '/11425_trj_151.xtc']
# Simulation B
ref_file_b = root_dir + '/11580_dyn_169.psf'
pdb_file_b = root_dir + '/11579_dyn_169.pdb'
trj_file_b = [root_dir + '/11576_trj_169.xtc',
              root_dir + '/11577_trj_169.xtc',
              root_dir + '/11578_trj_169.xtc']
# Base for the selection string for each simulation protein and all waters
sel_base_a = "protein or byres name OH2"
sel_base_b = "protein or byres name OH2"
# Names of the output files
out_name_a = "traj/cond-a_water"
out_name_b = "traj/cond-b_water"

for subdir in ['traj', 'plots', 'vispdb', 'pca', 'clusters', 'results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# Extract the coordinates of the receptor from the trajectory
extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a, sel_base_a)
extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b, sel_base_b)

# Extract the coordinates of the ensemble a aligned to ensemble b
extract_aligned_coords(out_name_a + ".gro", out_name_a + ".xtc",
                       out_name_b + ".gro", out_name_b + ".xtc")

# Extract the combined density of the waters in both ensembles a and b
extract_combined_grid(out_name_a + ".gro", "dens/cond-a_wateraligned.xtc",
                      out_name_b + ".gro", out_name_b + ".xtc",
                      atomgroup="OH2",
                      write_grid_as="TIP3P",
                      out_name="ab_grid_",
                      use_memmap=True)

grid_combined = "dens/ab_grid_OH2_density.dx"

# Then we featurize the waters common to both simulations
# We can do the same analysis for ions using the get_atom_features featurizer.
water_feat_a, water_data_a = get_water_features(structure_input=out_name_a + ".gro",
                                                xtc_input="dens/cond-a_wateraligned.xtc",
                                                top_waters=2,
                                                atomgroup="OH2",
                                                grid_input=grid_combined,
                                                write=True,
                                                out_name="cond_a")

water_feat_b, water_data_b = get_water_features(structure_input=out_name_b + ".gro",
                                                xtc_input=out_name_b + ".xtc",
                                                top_waters=2,
                                                atomgroup="OH2",
                                                grid_input=grid_combined,
                                                write=True,
                                                out_name="cond_b")

# Calculating SSI is then exactly the same as for residues
discrete_states_ab1 = get_discrete_states(water_data_a['WaterPocket_Distr'],
                                          water_data_b['WaterPocket_Distr'],
                                          discretize='gaussian', pbc=True)

# SSI shared between waters and the switch between ensemble conditions
data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'],
                                             water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
                                             discrete_states_ab1,
                                             verbose=True)

# Alternatively we can see if the pocket occupancy (the presence/absence of water at the site) shares SSI
# Currently this is only enabled with ssi_ensemble_analysis. We need to turn off the periodic boundary conditions
# as the distributions are no longer periodic.
discrete_states_ab2 = get_discrete_states(water_data_a['WaterPocket_OccupDistr'],
                                          water_data_b['WaterPocket_OccupDistr'],
                                          discretize='partition_values', pbc=False)

data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_OccupDistr'], water_feat_b['WaterPocket_OccupDistr'],
                                             water_data_a['WaterPocket_OccupDistr'], water_data_b['WaterPocket_OccupDistr'],
                                             discrete_states_ab2, pbc=False, verbose=True)

# In this example we can see that the state of water 01 shares ~0.25 bits of
# information with the ensembles, but the occupancy of water 1 pocket shares ~0.07 bits,
# revealing that the polarisation of this water is more functionally important with respect
# to what these ensembles are investigating.

# We can calculate the State Specific Information (SSI) shared between the
# water pockets across both ensembles. This quantity is no longer bound by 1 bit.
data_names, data_ssi = ssi_feature_analysis(water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'],
                                            water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
                                            discrete_states_ab1,
                                            verbose=True)

# The ssi_feature_analysis() tells us that these two water pockets are conformationally to some extent.

# An equivalent interpretation of co-SSI is how much the switch between ensembles
# is involved in strengthening (positive co-SSI) / weakening (negative co-SSI) the conformational coupling between two features.
feat_names, data_ssi, data_cossi = cossi_featens_analysis(water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'],
                                                          water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'],
                                                          water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
                                                          water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
                                                          discrete_states_ab1, discrete_states_ab1,
                                                          verbose=True)
