import numpy as np
import MDAnalysis as mda
import MDAnalysis.lib.distances as ld
import gpcrmining.gpcrdb as db


def read_atom_group_distances(pdb, xtc, sel_a='protein', sel_b='resname LIG',
                              first_frame=0, last_frame=None, step=1,
                              naming='plain'):
    """
    Load distances between all atom pairs between two selected groups.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    sel_a : str, default='protein'
        Selection string to choose atoms for the first group.
    sel_b : str, default='resname LIG'
        Selection string to choose atoms for the second group.
    first_frame : int, default=0
        First frame to return of the features. Zero-based.
    last_frame : int, default=None
        Last frame to return of the features. Zero-based.
    step : int, default=1
        Subsampling step width when reading the frames.
    naming : str, default='plain'
        Naming scheme for each atom in the feature names.
        plain: neither chain nor segment ID included
        chainid: include chain ID (only works if chains are defined)
        segid: include segment ID (only works if segments are defined)

    Returns
    -------
    feature_names : list of str
        Names of all distances
    features_data : numpy array
        Data for all distances [Å]

    """

    u = mda.Universe(pdb, xtc)
    a = u.select_atoms(sel_a)
    b = u.select_atoms(sel_b)
    num_at_a = len(a)
    num_at_b = len(b)

    # Name the atoms
    if naming == 'chainid':
        at_labels_a = ['%s %s %s %s' % (atom.chainID, atom.residue.resname, atom.resid, atom.name) for atom in a]
        at_labels_b = ['%s %s %s %s' % (atom.chainID, atom.residue.resname, atom.resid, atom.name) for atom in b]
    elif naming == 'segid':
        at_labels_a = ['%s %s %s %s' % (atom.segid, atom.residue.resname, atom.resid, atom.name) for atom in a]
        at_labels_b = ['%s %s %s %s' % (atom.segid, atom.residue.resname, atom.resid, atom.name) for atom in b]
    else:
        at_labels_a = ['%s %s %s' % (atom.residue.resname, atom.resid, atom.name) for atom in a]
        at_labels_b = ['%s %s %s' % (atom.residue.resname, atom.resid, atom.name) for atom in b]

    # Name the distance labels
    d_labels = []
    k = -1
    for i in range(num_at_a):
        for j in range(num_at_b):
            k += 1
            _dl = 'DIST: %s - %s' % (at_labels_a[i], at_labels_b[j])
            d_labels.append(_dl)

    # Calculate the distances
    num_at = len(a)
    num_dist = int(num_at * (num_at - 1) / 2)
    len_traj = len(u.trajectory[first_frame:last_frame:step])
    template = np.zeros([num_dist, ])
    data_arr = np.zeros([len_traj, num_dist])
    frame = 0
    for ts in u.trajectory[first_frame:last_frame:step]:
        data_arr[frame] = ld.self_distance_array(a.positions, result=template)
        frame += 1

    return d_labels, data_arr


def read_atom_self_distances(pdb, xtc, selection='all', first_frame=0, last_frame=None, step=1, naming='plain'):
    """
    Load distances between all selected atoms.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    selection : str
        Selection string to choose which atoms to include. Default: all.
    first_frame : int, default=0
        First frame to return of the features. Zero-based.
    last_frame : int, default=None
        Last frame to return of the features. Zero-based.
    step : int, default=1
        Subsampling step width when reading the frames.
    naming : str, default='plain'
        Naming scheme for each atom in the feature names.
        plain: neither chain nor segment ID included
        chainid: include chain ID (only works if chains are defined)
        segid: include segment ID (only works if segments are defined)

    Returns
    -------
    feature_names : list of str
        Names of all distances
    features_data : numpy array
        Data for all distances [Å]

    """

    u = mda.Universe(pdb, xtc)
    a = u.select_atoms(selection)
    num_at = len(a)

    # Name the atoms
    if naming == 'chainid':
        at_labels = ['%s %s %s %s' % (atom.chainID, atom.residue.resname, atom.resid, atom.name) for atom in a]
    elif naming == 'segid':
        at_labels = ['%s %s %s %s' % (atom.segid, atom.residue.resname, atom.resid, atom.name) for atom in a]
    else:
        at_labels = ['%s %s %s' % (atom.residue.resname, atom.resid, atom.name) for atom in a]

    # Name the distance labels
    d_labels = []
    k = -1
    for i in range(num_at):
        for j in range(i + 1, num_at):
            k += 1
            _dl = 'DIST: %s - %s' % (at_labels[i], at_labels[j])
            d_labels.append(_dl)

    # Calculate the distances
    num_at = len(a)
    num_dist = int(num_at * (num_at - 1) / 2)
    len_traj = len(u.trajectory[first_frame:last_frame:step])
    template = np.zeros([num_dist, ])
    data_arr = np.zeros([len_traj, num_dist])
    frame = 0
    for ts in u.trajectory[first_frame:last_frame:step]:
        data_arr[frame] = ld.self_distance_array(a.positions, result=template)
        frame += 1

    return d_labels, data_arr


def read_calpha_distances(pdb, xtc, first_frame=0, last_frame=None, step=1):
    """
    Load distances between all C-alpha atoms.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    first_frame : int, default=0
        First frame to return of the features. Zero-based.
    last_frame : int, default=None
        Last frame to return of the features. Zero-based.
    step : int, default=1
        Subsampling step width when reading the frames.

    Returns
    -------
    feature_names : list of str
        Names of all C-alpha distances
    features_data : numpy array
        Data for all C-alpha distances [Å]

    """
    names, data = read_atom_self_distances(
        pdb, xtc,
        selection='name CA',
        first_frame=first_frame,
        last_frame=last_frame,
        step=step
    )
    return names, data


def select_gpcr_residues(gpcr_name, res_dbnum):
    """
    Gets sequential residue numbers for residues provided as GPCRdb numbers.

    Parameters
    ----------
    gpcr_name : str
        Name of the GPCR as in the GPCRdb.
    res_dbnum : list of str
        Relative GPCR residue numbers.

    Returns
    -------
    sel_resnum : list of int
        Sequential residue numbers.
    sel_labels : list of str
        Labels containing GPCRdb numbering of the residues.

    """
    res_array = db.get_residue_info(gpcr_name)
    sel_array = db.select_by_gpcrdbnum(res_array, res_dbnum)
    sel_resnum = [item[1] for item in sel_array]
    sel_labels = [item[3] for item in sel_array]
    return sel_resnum, sel_labels


def read_gpcr_calpha_distances(pdb, xtc, gpcr_name, res_dbnum,
                               first_frame=0, last_frame=None, step=1):
    """
    Load distances between all selected atoms.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    gpcr_name : str
        Name of the GPCR as in the GPCRdb.
    res_dbnum : list
        Relative GPCR residue numbers.
    first_frame : int, default=0
        First frame to return of the features. Zero-based.
    last_frame : int, default=None
        Last frame to return of the features. Zero-based.
    step : int, default=1
        Subsampling step width when reading the frames.

    Returns
    -------
    feature_names : list of str
        Names of all C-alpha distances.
    feature_labels : list of str
        Labels containing GPCRdb numbering of the residues.
    features_data : numpy array
        Data for all C-alpha distances [Å].

    """
    # Select residues from relative residue numbers
    resnums, reslabels = select_gpcr_residues(gpcr_name, res_dbnum)
    # Create the selection string
    selection = 'name CA and resid'
    for rn in resnums:
        selection += ' %i' % rn
    # Create the GPCRdb distance labels
    distlabels = []
    k = -1
    for i in range(len(reslabels)):
        for j in range(i + 1, len(reslabels)):
            k += 1
            _dl = 'CA DIST: %s - %s' % (reslabels[i], reslabels[j])
            distlabels.append(_dl)
    # Calculate the distances and get the sequential names
    names, data = read_atom_self_distances(
        pdb, xtc,
        selection=selection,
        first_frame=first_frame,
        last_frame=last_frame,
        step=step
    )
    return names, distlabels, data
