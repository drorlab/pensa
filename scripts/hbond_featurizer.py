from pensa.features import hbond_features, atom_features

# Featurize water cavity hydrogen bonds

root_dir = './mor-data'

ref_file_a = root_dir + '/11427_dyn_151.psf'
pdb_file_a = root_dir + '/11426_dyn_151.pdb'
trj_file_a = root_dir + '/11423_trj_151.xtc'

names, data = hbond_features.read_water_site_h_bonds_quickly(
    ref_file_a, trj_file_a,
    atomgroups=['OH2', 'H1', 'H2'],
    site_IDs=[1, 2],
    grid_input=None,
    write_grid_as='TIP3P',
    out_name='11423_trj_151'
)

# Featurize ligand-protein hydrogen bonds

ref_file_a = root_dir + '/11580_dyn_169.psf'
pdb_file_a = root_dir + '/11579_dyn_169.pdb'
trj_file_a = root_dir + '/11578_trj_169.xtc'

names, data = hbond_features.read_h_bonds(
    ref_file_a, trj_file_a,
    fixed_group='resname 4VO',
    dyn_group='protein',
    out_name='4VO_hbonds'
)
