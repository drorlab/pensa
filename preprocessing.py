import MDAnalysis as mda
import pyemma
import numpy as np



# -- Loading the Features --


def get_features(pdb,xtc,start_frame):
    """http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html"""
    
    feature_names = {}
    features_data = {}

    bbtorsions_feat = pyemma.coordinates.featurizer(pdb)
    bbtorsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    bbtorsions_data = pyemma.coordinates.load(xtc, features=bbtorsions_feat)[start_frame:]
    feature_names['bb-torsions'] = bbtorsions_feat
    features_data['bb-torsions'] = bbtorsions_data.T
    
    sctorsions_feat = pyemma.coordinates.featurizer(pdb)
    sctorsions_feat.add_sidechain_torsions(cossin=True, periodic=False)
    sctorsions_data = pyemma.coordinates.load(xtc, features=sctorsions_feat)[start_frame:]
    feature_names['sc-torsions'] = sctorsions_feat
    features_data['sc-torsions'] = sctorsions_data.T

    bbdistances_feat = pyemma.coordinates.featurizer(pdb)
    bbdistances_feat.add_distances(bbdistances_feat.pairs(bbdistances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
    bbdistances_data = pyemma.coordinates.load(xtc, features=bbdistances_feat)[start_frame:]
    feature_names['bb-distances'] = bbdistances_feat
    features_data['bb-distances'] = bbdistances_data.T
    
    return feature_names, features_data



# -- Preprocessing Trajectories --


def range_to_string(a, b):
    """Converts a tuple to a string with all integers in between"""
    
    r = np.arange(a, b+1)
    string = ''
    for ri in r:
        string += str(ri)
        string += ' '

    return string


def load_selection(sel_file, sel_base=''):
    """Load a selection from a selection string"""
    
    sel_string = sel_base+'resid '
    for r in np.loadtxt(sel_file, dtype=int):
        sel_string += range_to_string(*r)
    
    return sel_string


def extract_coordinates(ref, pdb, trj, out_name, sel_string):
    """Extract selected coordinates from a trajectory file"""
    
    u = mda.Universe(ref,pdb)
    selection = u.select_atoms(sel_string)
    print(selection.ids)

    selection.write(out_name+'.pdb')
    selection.write(out_name+'.gro')

    u = mda.Universe(ref,trj)
    selection = u.select_atoms(sel_string)

    with mda.Writer(out_name+'.xtc', selection.n_atoms) as W:
        for ts in u.trajectory:
            W.write(selection)
            
    return


def extract_coordinates_combined(ref, trj, sel_string, out_name, start_frame=0):
    """Extract selected coordinates from several trajectory files"""
        
    # Determine number of atoms from first trajectory
    u = mda.Universe(ref[0], trj[0])
    selection = u.select_atoms(sel_string[0])
    num_at = selection.n_atoms
                                          
    # Go through trajectories and write selections
    with mda.Writer(out_name+'.xtc', num_at) as W:
        for r, t, s in zip(ref, trj, sel_string):
            print(r, t)
            print(s)
            u = mda.Universe(r, t)
            selection = u.select_atoms(s)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
            
    return

