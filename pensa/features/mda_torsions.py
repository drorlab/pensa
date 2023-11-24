import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral


def read_torsions(pdb, xtc, sel=[[0, 1, 2, 3], [1, 2, 3, 4]], first_frame=0, last_frame=None, step=1, naming=None):
    """
    Load distances between all atom pairs between two selected groups.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    sel : list, default=[[0, 1, 2, 3]]
        List of quadruplets with selection indices to choose atoms for the torsions.
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
        segindex: include segment index (only works if segments are defined)

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]
    """

    for selection in sel:
        assert len(selection) == 4

    # Read the dihedral angles
    u = mda.Universe(pdb, xtc)
    torsion_atoms = [u.atoms[selection] for selection in sel]
    dihedrals = Dihedral(torsion_atoms).run()
    dihedral_angles = dihedrals.angles

    # Generate the labels
    torsion_labels = []
    for ta in torsion_atoms:
        # Name the atoms
        if naming == 'chainid':
            at_labels = ['%s %s %s %s' % (atom.chainID, atom.residue.resname, atom.resid, atom.name) for atom in ta]
        elif naming == 'segid':
            at_labels = ['%s %s %s %s' % (atom.segid, atom.residue.resname, atom.resid, atom.name) for atom in ta]
        elif naming == 'segindex':
            at_labels = ['%s %s %s %s' % (atom.segindex, atom.residue.resname, atom.resid, atom.name) for atom in ta]
        else:
            at_labels = ['%s %s %s' % (atom.residue.resname, atom.resid, atom.name) for atom in ta]
        # Name the torsion labels
        _tl = 'TORS: %s - %s - %s - %s' % (at_labels[0], at_labels[1], at_labels[2], at_labels[3])
        torsion_labels.append(_tl)

    return torsion_labels, dihedral_angles[first_frame:last_frame:step]


def find_atom_by_name(res, at_name):
    """
    Find the index of the first atom of a certain name in a residue.

    Parameters
    ----------
    res : Residue
        MDAnalysis residue object.
    at_name : str
        Name of the requested atom.

    Returns
    -------
    index : int
        Index of the first atom with name at_name or -1 (if none of the atoms has this name)

    """
    for atom in res.atoms:
        if atom.name == at_name:
            return atom.index
    return -1


def list_depth(a_list):
    if isinstance(a_list, list):
        return 1 + max(list_depth(item) for item in a_list)
    else:
        return 0


def find_atom_indices_per_residue(pdb, at_names=["C4'", "P", "C4'", "P"], rel_res=[-1, 0, 0, 1],
                                  selection='all', verbose=False):
    """
    Find the indices of atoms with a certain name for each residue (and its neighbors).

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    at_names : list of str or list of list of str
        Names of the requested atoms or list of sets of names of requested atoms.
        If a list of lists is passed, all sub-lists must have the same length.
    rel_res : list of int, default=[-1, 0, 0, 1]]
        Residue number of each atom's residue relative to the current residue.
    selection : str, default = 'all'
        MDAnalysis selection string
    verbose : bool, default = False
        Print info for all residues.

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]
    """

    # If only one name list is given, create a list with this list
    if list_depth(at_names) <= 1:
        at_names = [at_names]
    # Make sure all name lists have the right length
    for an in at_names:
        assert len(an) == len(rel_res)
    # Make sure the relative residue numbers are indices
    rel_res = np.array(rel_res, dtype=int)

    u = mda.Universe(pdb)
    a = u.select_atoms(selection)
    r = a.residues

    indices_list = []

    # In each residue ..
    for i, res in enumerate(r):

        sel_resnums = res.resnum + rel_res
        sel_resinds = i + rel_res

        # Check whether all residue indices are present.
        if not np.all(sel_resinds < len(r)):
            continue

        # Check whether consecutive indices correspond to consecutive residue numbers
        # (we don't want to calculate torsions between non-connected residues)
        if not np.all(sel_resnums == r[sel_resinds].resnums):
            continue

        # Try each set of selection names ...
        num_indices = 0
        num_at_sets = 0
        while num_indices < len(rel_res) and num_at_sets < len(at_names):
            # For each selection name ...
            indices = -np.ones(len(rel_res), dtype=int)
            for j, at_name in enumerate(at_names[num_at_sets]):
                # ... check each atom.
                indices[j] = find_atom_by_name(r[i + rel_res[j]], at_name)
            num_indices = np.sum(indices >= 0)
            num_at_sets += 1

        if num_indices == len(rel_res):
            if verbose:
                print(indices, a[indices].names, a[indices].resids)
            indices_list.append(indices)

    return indices_list


def read_nucleicacid_backbone_torsions(pdb, xtc, selection='all',
                                      first_frame=0, last_frame=None, step=1,
                                      naming='segindex', radians=False):
    """
    Load nucleic acid backbone torsions

    ALPHA (α):   O3'(i-1)-P(i)-O5'(i)-C5'(i)
    BETA (β):    P(i)-O5'(i)-C5'(i)-C4'(i)
    GAMMA (γ):   O5'(i)-C5'(i)-C4'(i)-C3'(i)
    DELTA (δ):   C5'(i)-C4'(i)-C3'(i)-O3'(i)
    EPSILON (ε): C4'(i)-C3'(i)-O3'(i)-P(i + 1)
    ZETA (ζ):    C3'(i)-O3'(i)-P(i + 1)-O5'(i + 1)
    CHI (χ):     O4'(i)-C1'(i)-N9(i)-C4(i) for purines
    or           O4'(i)-C1'(i)-N1(i)-C2(i) for pyridines

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    selection : list, default='all'
        List of quadruplets with selection indices to choose atoms for the torsions.
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
        segindex: include segment index (only works if segments are defined)
    radians : bool, default=False
        Return torsions in radians instead of degrees.

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]

    """
    # Find indices of torsion atoms for each residue
    # ALPHA (α):   O3'(i-1)-P(i)-O5'(i)-C5'(i)
    indices_alpha = find_atom_indices_per_residue(
        pdb,
        at_names=["O3'", "P", "O5'", "C5'"],
        rel_res=[-1, 0, 0, 0],
        selection=selection
    )
    # BETA (β):    P(i)-O5'(i)-C5'(i)-C4'(i)
    indices_beta = find_atom_indices_per_residue(
        pdb,
        at_names=["P", "O5'", "C5'", "C4'"],
        rel_res=[0, 0, 0, 0],
        selection=selection
    )
    # GAMMA (γ):   O5'(i)-C5'(i)-C4'(i)-C3'(i)
    indices_gamma = find_atom_indices_per_residue(
        pdb,
        at_names=["O5'", "C5'", "C4'", "C3'"],
        rel_res=[0, 0, 0, 0],
        selection=selection
    )
    # DELTA (δ):   C5'(i)-C4'(i)-C3'(i)-O3'(i)
    indices_delta = find_atom_indices_per_residue(
        pdb,
        at_names=["C5'", "C4'", "C3'", "O3'"],
        rel_res=[0, 0, 0, 0],
        selection=selection
    )
    # EPSILON (ε): C4'(i)-C3'(i)-O3'(i)-P(i + 1)
    indices_epsilon = find_atom_indices_per_residue(
        pdb,
        at_names=["C4'", "C3'", "O3'", "P"],
        rel_res=[0, 0, 0, 1],
        selection=selection
    )
    # ZETA (ζ):    C3'(i)-O3'(i)-P(i + 1)-O5'(i + 1)
    indices_zeta = find_atom_indices_per_residue(
        pdb,
        at_names=["C3'", "O3'", "P", "O5'"],
        rel_res=[0, 0, 1, 1],
        selection=selection
    )
    # CHI (χ):     O4'(i)-C1'(i)-N9(i)-C4(i) for purines
    #              O4'(i)-C1'(i)-N1(i)-C2(i) for pyridines
    indices_chi = find_atom_indices_per_residue(
        pdb,
        at_names=[["O4'", "C1'", "N9", "C4"], ["O4'", "C1'", "N1", "C2"]],
        rel_res=[0, 0, 0, 0],
        selection=selection
    )

    # Define angle names for labels
    angles = []
    angles += ['ALPHA'] * len(indices_alpha)
    angles += ['BETA'] * len(indices_beta)
    angles += ['GAMMA'] * len(indices_gamma)
    angles += ['DELTA'] * len(indices_delta)
    angles += ['EPSILON'] * len(indices_epsilon)
    angles += ['ZETA'] * len(indices_zeta)
    angles += ['CHI'] * len(indices_chi)
    # Calculate the torsions
    all_indices = indices_alpha + indices_beta + indices_gamma + indices_delta \
        + indices_epsilon + indices_zeta + indices_chi
    torsions = read_torsions(
        pdb, xtc, sel=all_indices, naming=naming,
        first_frame=first_frame, last_frame=last_frame, step=step
    )
    # Extract the residue info
    nums = [pti.split(' - ')[1].split(' ')[-2] for pti in torsions[0]]
    names = [pti.split(' - ')[1].split(' ')[-3] for pti in torsions[0]]
    if naming == 'chainid' or naming == 'segid' or naming == 'segindex':
        seg = [pti.split(' - ')[1].split(' ')[-4] for pti in torsions[0]]
    else:
        seg = ['0'] * len(angles)
    # Construct label names
    labels = [ang + ' ' + seg[i] + ' ' + names[i] + ' ' + nums[i] for i, ang in enumerate(angles)]
    # Convert to radians if so desired
    if radians:
        values = torsions[1] * np.pi / 180
    else:
        values = torsions[1]
    return labels, values


def read_nucleicacid_pseudotorsions(pdb, xtc, selection='all',
                                   first_frame=0, last_frame=None, step=1,
                                   naming='segindex', radians=False):
    """
    Load nucleic acid pseudotorsions

    ETA (η):   C4'(i-1)-P(i)-C4'(i)-P(i + 1)
    THETA (θ): P(i)-C4'(i)-P(i + 1)-C4'(i + 1)

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    selection : list, default='all'
        List of quadruplets with selection indices to choose atoms for the torsions.
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
        segindex: include segment index (only works if segments are defined)
    radians : bool, default=False
        Return torsions in radians instead of degrees.

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]

    """
    # Find indices of torsion atoms for each residue
    indices_eta = find_atom_indices_per_residue(
        pdb,
        at_names=["C4'", "P", "C4'", "P"],
        rel_res=[-1, 0, 0, 1],
        selection=selection
    )
    indices_theta = find_atom_indices_per_residue(
        pdb,
        at_names=["P", "C4'", "P", "C4'"],
        rel_res=[0, 0, 1, 1],
        selection=selection
    )
    # Define angle names for labels
    angles = ['ETA'] * len(indices_eta) + ['THETA'] * len(indices_theta)
    # Calculate the torsions
    torsions = read_torsions(
        pdb, xtc, sel=indices_eta + indices_theta, naming=naming,
        first_frame=first_frame, last_frame=last_frame, step=step
    )
    # Extract the residue info
    nums = [pti.split(' - ')[1].split(' ')[-2] for pti in torsions[0]]
    names = [pti.split(' - ')[1].split(' ')[-3] for pti in torsions[0]]
    if naming == 'chainid' or naming == 'segid' or naming == 'segindex':
        seg = [pti.split(' - ')[1].split(' ')[-4] for pti in torsions[0]]
    else:
        seg = ['0'] * len(angles)
    # Construct label names
    labels = [ang + ' ' + seg[i] + ' ' + names[i] + ' ' + nums[i] for i, ang in enumerate(angles)]
    # Convert to radians if so desired
    if radians:
        values = torsions[1] * np.pi / 180
    else:
        values = torsions[1]
    return labels, values


def read_protein_backbone_torsions(pdb, xtc, selection='all',
                                   first_frame=0, last_frame=None, step=1,
                                   naming='segindex', radians=False,
                                   include_omega=False):
    """
    Load protein backbone torsions

    PHI (φ):   C(i-1)-N(i)-CA(i)-C(i)
    PSI (ψ):   N(i)-CA(i)-C(i)-N(i + 1)
    OMEGA (ω): CA(i)-C(i)-N(i + 1)-CA(i + 1)

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    selection : list, default='all'
        List of quadruplets with selection indices to choose atoms for the torsions.
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
        segindex: include segment index (only works if segments are defined)
    radians : bool, default=False
        Return torsions in radians instead of degrees.

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]

    """
    # Find indices of torsion atoms for each residue
    #   PHI (φ):   C(i-1)-N(i)-CA(i)-C(i)
    indices_phi = find_atom_indices_per_residue(
        pdb,
        at_names=["C", "N", "CA", "C"],
        rel_res=[-1, 0, 0, 0],
        selection=selection
    )
    #   PSI (ψ):   N(i)-CA(i)-C(i)-N(i + 1)
    indices_psi = find_atom_indices_per_residue(
        pdb,
        at_names=["N", "CA", "C", "N"],
        rel_res=[0, 0, 0, 1],
        selection=selection
    )
    #   OMEGA (ω): CA(i)-C(i)-N(i + 1)-CA(i + 1)
    indices_omega = find_atom_indices_per_residue(
        pdb,
        at_names=["CA", "C", "N", "CA"],
        rel_res=[0, 0, 1, 1],
        selection=selection
    )
    # Define angle names for labels
    angles = ['PHI'] * len(indices_phi) + ['PSI'] * len(indices_psi)
    torsion_selection = indices_phi + indices_psi
    if include_omega:
        angles += ['OMEGA'] * len(indices_omega)
        torsion_selection += indices_omega
    # Calculate the torsions
    torsions = read_torsions(
        pdb, xtc, sel=torsion_selection, naming=naming,
        first_frame=first_frame, last_frame=last_frame, step=step
    )
    # Extract the residue info
    nums = [pti.split(' - ')[1].split(' ')[-2] for pti in torsions[0]]
    names = [pti.split(' - ')[1].split(' ')[-3] for pti in torsions[0]]
    if naming == 'chainid' or naming == 'segid' or naming == 'segindex':
        seg = [pti.split(' - ')[1].split(' ')[-4] for pti in torsions[0]]
    else:
        seg = ['0'] * len(angles)
    # Construct label names
    labels = [ang + ' ' + seg[i] + ' ' + names[i] + ' ' + nums[i] for i, ang in enumerate(angles)]
    # Convert to radians if so desired
    if radians:
        values = torsions[1] * np.pi / 180
    else:
        values = torsions[1]
    return labels, values


at_names_chi1 = [["N", "CA", "CB", "CG"],
                 ["N", "CA", "CB", "CG1"],
                 ["N", "CA", "CB", "SG"],
                 ["N", "CA", "CB", "OG"],
                 ["N", "CA", "CB", "OG1"]]

at_names_chi2 = [["CA", "CB", "CG", "CD"],
                 ["CA", "CB", "CG", "CD1"],
                 ["CA", "CB", "CG1", "CD"],
                 ["CA", "CB", "CG1", "CD1"],
                 ["CA", "CB", "CG", "OD1"],
                 ["CA", "CB", "CG", "ND1"],
                 ["CA", "CB", "CG", "SD"]]

at_names_chi3 = [["CB", "CG", "CD", "NE"],
                 ["CB", "CG", "CD", "CE"],
                 ["CB", "CG", "CD", "OE1"],
                 ["CB", "CG", "SD", "CE"]]

at_names_chi4 = [["CG", "CD", "NE", "CZ"],
                 ["CG", "CD", "CE", "NZ"]]

at_names_chi5 = [["CD", "NE", "CZ", "NH1"]]


def read_protein_sidechain_torsions(pdb, xtc, selection='all',
                                    first_frame=0, last_frame=None, step=1,
                                    naming='segindex', radians=False):
    """
    Load protein sidechain torsions.

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    selection : list, default='all'
        List of quadruplets with selection indices to choose atoms for the torsions.
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
        segindex: include segment index (only works if segments are defined)
    radians : bool, default=False
        Return torsions in radians instead of degrees.

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]

    """
    # Find indices of torsion atoms for each residue
    indices_chi1 = find_atom_indices_per_residue(
        pdb, at_names_chi1, [0, 0, 0, 0], selection
    )
    indices_chi2 = find_atom_indices_per_residue(
        pdb, at_names_chi2, [0, 0, 0, 0], selection
    )
    indices_chi3 = find_atom_indices_per_residue(
        pdb, at_names_chi3, [0, 0, 0, 0], selection
    )
    indices_chi4 = find_atom_indices_per_residue(
        pdb, at_names_chi4, [0, 0, 0, 0], selection
    )
    indices_chi5 = find_atom_indices_per_residue(
        pdb, at_names_chi5, [0, 0, 0, 0], selection
    )
    # Define angle names for labels
    angles = []
    angles += ['CHI1'] * len(indices_chi1)
    angles += ['CHI2'] * len(indices_chi2)
    angles += ['CHI3'] * len(indices_chi3)
    angles += ['CHI4'] * len(indices_chi4)
    angles += ['CHI5'] * len(indices_chi5)
    # Define torsion selections
    torsion_selection = []
    torsion_selection += indices_chi1
    torsion_selection += indices_chi2
    torsion_selection += indices_chi3
    torsion_selection += indices_chi4
    torsion_selection += indices_chi5
    # Calculate the torsions
    torsions = read_torsions(
        pdb, xtc, sel=torsion_selection, naming=naming,
        first_frame=first_frame, last_frame=last_frame, step=step
    )
    # Extract the residue info
    nums = [pti.split(' - ')[1].split(' ')[-2] for pti in torsions[0]]
    names = [pti.split(' - ')[1].split(' ')[-3] for pti in torsions[0]]
    if naming == 'chainid' or naming == 'segid' or naming == 'segindex':
        seg = [pti.split(' - ')[1].split(' ')[-4] for pti in torsions[0]]
    else:
        seg = ['0'] * len(angles)
    # Construct label names
    labels = [ang + ' ' + seg[i] + ' ' + names[i] + ' ' + nums[i] for i, ang in enumerate(angles)]
    # Convert to radians if so desired
    if radians:
        values = torsions[1] * np.pi / 180
    else:
        values = torsions[1]
    return labels, values
