import os
import numpy as np
import glob
import mdtraj as md

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def get_fns(dir_name,pattern):
    return np.sort(glob.glob(os.path.join(dir_name, pattern)))

def load_traj_coords_dir(dir_name,pattern,top):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        t = md.load(fn, top=top)
        d = t.xyz.reshape((len(t), 3*top.n_atoms))
        all_d.append(d)
    all_d = np.vstack(all_d)
    return all_d

def load_npy_dir(dir_name,pattern):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        d = np.load(fn)
        all_d.append(d)
    if len(d.shape) == 1:
        all_d = np.hstack(all_d)
    else:
        all_d = np.vstack(all_d)
    return all_d
