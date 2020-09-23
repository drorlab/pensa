import MDAnalysis as mda
import pyemma
import numpy as np



# -- Loading the Features --


def get_features(pdb,xtc,start_frame):
    """http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html"""
    
    labels = []
    features = []
    data = []
    
    torsions_feat  = pyemma.coordinates.featurizer(pdb)
    torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    torsions_data = pyemma.coordinates.load(xtc, features=torsions_feat)[start_frame:]
    labels   = ['backbone\ntorsions']
    features = [torsions_feat]
    data     = [torsions_data]
    
    distances_feat = pyemma.coordinates.featurizer(pdb)
    distances_feat.add_distances(distances_feat.pairs(distances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
    distances_data = pyemma.coordinates.load(xtc, features=distances_feat)[start_frame:]
    labels   += ['backbone atom\ndistances']
    features += [distances_feat]
    data     += [distances_data]
    
    sidechains_feat  = pyemma.coordinates.featurizer(pdb)
    sidechains_feat.add_sidechain_torsions(cossin=True, periodic=False)
    sidechains_data = pyemma.coordinates.load(xtc, features=sidechains_feat)[start_frame:]
    labels   += ['sidechains\ntorsions']
    features += [sidechains_feat]
    data     += [sidechains_data]
    
    return labels, features, data



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

