from pensa.features import \
    read_protein_backbone_torsions, \
    read_protein_sidechain_torsions, \
    read_nucleicacid_backbone_torsions, \
    read_nucleicacid_pseudotorsions

# Location of the data used in the tests
test_data_path = './tests/test_data'

# Simulation A (MOR-apo)
root_dir_a = test_data_path + '/MOR-apo'
ref_file_a = root_dir_a + '/mor-apo.psf'
pdb_file_a = root_dir_a + '/mor-apo.pdb'
trj_file_a = [root_dir_a + '/mor-apo-1.xtc',
              root_dir_a + '/mor-apo-2.xtc',
              root_dir_a + '/mor-apo-3.xtc']

# Simulation B (MOR-BU72)
root_dir_b = test_data_path + '/MOR-BU72'
ref_file_b = root_dir_b + '/mor-bu72.psf'
pdb_file_b = root_dir_b + '/mor-bu72.pdb'
trj_file_b = [root_dir_b + '/mor-bu72-1.xtc',
              root_dir_b + '/mor-bu72-2.xtc',
              root_dir_b + '/mor-bu72-3.xtc']

# Simulation O (DNA-opt)
root_dir_o = test_data_path + '/DNA-opt'
gro_file_o = root_dir_o + '/DNA-nw.gro'
trj_file_o = root_dir_o + '/trajfit-nw_step100.xtc'

# Simulation S (DNA-std)
root_dir_s = test_data_path + '/DNA-std'
gro_file_s = root_dir_s + '/DNA-nw.gro'
trj_file_s = root_dir_s + '/trajfit-nw_step100.xtc'


# Test protein backbone torsions
def test_read_protein_backbone_torsions():
    # Feature Loaders
    bb_torsions_a = read_protein_backbone_torsions(
        pdb_file_a, trj_file_a[0], selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True,
        include_omega=False
    )
    bb_torsions_b = read_protein_backbone_torsions(
        pdb_file_b, trj_file_b[0], selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True,
        include_omega=False
    )
    assert len(bb_torsions_a) == len(bb_torsions_b)
    assert bb_torsions_a[0][0] == bb_torsions_b[0][0] == 'PHI 0 VAL 66'
    assert bb_torsions_a[0][-1] == bb_torsions_b[0][-1] == 'PSI 0 CYS 351'
    assert '%1.4f %1.4f' % (bb_torsions_a[1][0][0], bb_torsions_a[1][0][-1]) == '-1.0394 -0.7756'
    assert '%1.4f %1.4f' % (bb_torsions_a[1][-1][0], bb_torsions_a[1][-1][-1]) == '-1.0648 -0.5028'


# Test protein side-chain torsions
def test_read_protein_sidechain_torsions():
    # Feature Loaders
    sc_torsions_a = read_protein_sidechain_torsions(
        pdb_file_a, trj_file_a[0], selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    sc_torsions_b = read_protein_sidechain_torsions(
        pdb_file_b, trj_file_b[0], selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    assert len(sc_torsions_a) == len(sc_torsions_b)
    assert sc_torsions_a[0][0] == sc_torsions_b[0][0] == 'CHI1 0 MET 65'
    assert sc_torsions_a[0][-1] == sc_torsions_b[0][-1] == 'CHI5 0 ARG 348'
    assert '%1.4f %1.4f' % (sc_torsions_a[1][0][0], sc_torsions_a[1][0][-1]) == '1.0151 -0.2732'
    assert '%1.4f %1.4f' % (sc_torsions_a[1][-1][0], sc_torsions_a[1][-1][-1]) == '-0.7536 0.1536'


# Test nucleic acid backbone torsions
def test_read_nucleicacid_backbone_torsions():
    bb_torsions_o = read_nucleicacid_backbone_torsions(
        gro_file_o, trj_file_o, selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    bb_torsions_s = read_nucleicacid_backbone_torsions(
        gro_file_s, trj_file_s, selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    assert bb_torsions_o[0] == bb_torsions_s[0]


# Test nucleic acid pseudo-torsions
def test_read_nucleicacid_pseudotorsions():
    pseudotorsions_o = read_nucleicacid_pseudotorsions(
        gro_file_o, trj_file_o, selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    pseudotorsions_s = read_nucleicacid_pseudotorsions(
        gro_file_s, trj_file_s, selection='all',
        first_frame=0, last_frame=None, step=1,
        naming='segindex', radians=True
    )
    pseudotorsions_o[0] == pseudotorsions_s[0]
