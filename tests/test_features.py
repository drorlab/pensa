import pytest
import os
import importlib
from pensa.features import *

# Location of the data used in the tests
test_data_path = './tests/test_data'

# Simulation A (MOR-apo)
root_dir_a = test_data_path+'/MOR-apo'
ref_file_a =  root_dir_a+'/mor-apo.psf'
pdb_file_a =  root_dir_a+'/mor-apo.pdb'
trj_file_a = [root_dir_a+'/mor-apo-1.xtc', 
              root_dir_a+'/mor-apo-2.xtc', 
              root_dir_a+'/mor-apo-3.xtc']

# Simulation B (MOR-BU72)
root_dir_b = test_data_path+'/MOR-BU72'
ref_file_b =  root_dir_b+'/mor-bu72.psf'
pdb_file_b =  root_dir_b+'/mor-bu72.pdb'
trj_file_b = [root_dir_b+'/mor-bu72-1.xtc', 
              root_dir_b+'/mor-bu72-2.xtc', 
              root_dir_b+'/mor-bu72-3.xtc']

# Simulation O (DNA-opt)
root_dir_o = test_data_path+'/DNA-opt'
ref_file_o = root_dir_o+'/DNA-nw.gro'
trj_file_o = root_dir_o+'/trajfit-nw_step100.xtc'

# Simulation S (DNA-std)
root_dir_s = test_data_path+'/DNA-std'

# Test protein backbone torsions
def test_get_protein_backbone_torsions():
    # Feature Loaders
    bb_torsions_a = get_protein_backbone_torsions(
        pdb_file_a, trj_file_a[0], selection='all',
        first_frame=0, last_frame=None, step=1, 
        naming='segindex', include_omega=False
        )
    bb_torsions_b = get_protein_backbone_torsions(
        pdb_file_b, trj_file_b[0], selection='all',
        first_frame=0, last_frame=None, step=1, 
        naming='segindex', include_omega=False
        )
    assert len(bb_torsions_a) == len(bb_torsions_b)

# Test protein side-chain torsions
def test_get_protein_sidechain_torsions():
    # Feature Loaders
    sc_torsions_a = get_protein_sidechain_torsions(
        pdb_file_a, trj_file_a[0], selection='all',
        first_frame=0, last_frame=None, step=1, 
        naming='segindex', include_omega=False
        )
    sc_torsions_b = get_protein_sidechain_torsions(
        pdb_file_b, trj_file_b[0], selection='all',
        first_frame=0, last_frame=None, step=1, 
        naming='segindex', include_omega=False
        )
    assert len(sc_torsions_a) == len(sc_torsions_b)

# Tests
def test_get_features():
    assert 4 == 4
    pass

