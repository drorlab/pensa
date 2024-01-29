import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (HydrogenBondAnalysis as HBA)
import numpy as np
from gridData import Grid
from tqdm import tqdm
import os
from pensa.preprocessing.density import generate_grid, local_maxima_3D, write_atom_to_pdb



# -------------------
#  Helper Functions
# -------------------


def name_atom_features(u, atom_ids, feature_type='H-DON', naming='plain'):
    atom_names = []
    for id in atom_ids:
        at = u.select_atoms('index %i' % id)[0]
        if naming == 'chainid':
            at_label = '%s %s %s %s' % (at.chainID, at.residue.resname, at.resid, at.name)
        elif naming == 'segid':
            at_label = '%s %s %s %s' % (at.segid, at.residue.resname, at.resid, at.name)
        else:
            at_label = '%s %s %s' % (at.residue.resname, at.resid, at.name)
        atom_names.append('%s: %s' % (feature_type, at_label))
    return atom_names


def name_pairs(u, all_pairs, pair_type='HBOND', naming='plain'):
    pair_names = []
    for pair in all_pairs:
        don = u.select_atoms('index ' + str(pair[0]))[0]
        acc = u.select_atoms('index ' + str(pair[1]))[0]
        if naming == 'chainid':
            at_label_don = '%s %s %s %s' % (don.chainID, don.residue.resname, don.resid, don.name)
            at_label_acc = '%s %s %s %s' % (acc.chainID, acc.residue.resname, acc.resid, acc.name)
        elif naming == 'segid':
            at_label_don = '%s %s %s %s' % (don.segid, don.residue.resname, don.resid, don.name)
            at_label_acc = '%s %s %s %s' % (acc.segid, acc.residue.resname, acc.resid, acc.name)
        else:
            at_label_don = '%s %s %s' % (don.residue.resname, don.resid, don.name)
            at_label_acc = '%s %s %s' % (acc.residue.resname, acc.resid, acc.name)
        pair_names.append('%s: %s - %s' % (pair_type, at_label_don, at_label_acc))
    return pair_names



# ----------------------------------------------------
#  Functions based on a distance-and-angle criterion
# ----------------------------------------------------


def read_h_bonds(structure_input, xtc_input, selection1, selection2, naming='plain'):
    """
    Read hydrogen bonds between two atm groups.

    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    selection1 : str
        Atom group selection to find bonding partners for.
    selection2: str
        Atom group selection to find bonding partners within.
    naming : str, default='plain'
        Naming scheme for each atom in the feature names.
        plain: neither chain nor segment ID included
        chainid: include chain ID (only works if chains are defined)
        segid: include segment ID (only works if segments are defined)

    Returns
    -------
        feature_names : list of str
            Names of all bonds
        features_data : numpy array
            Data for all bonds

    """

    # Find hydrogen bonds between the groups
    u = mda.Universe(structure_input, xtc_input)
    hbond = HBA(universe=u, between=[selection1, selection2])
    hbonds_mda = hbond.run()
    hb = hbonds_mda.results.hbonds

    # Determine all donor-acceptor pairs
    all_pairs = np.array(np.unique(hb[:,1:4:2], axis=0), dtype=int).tolist()
    # Determine the corresponding names
    feature_names = name_pairs(u, all_pairs, pair_type='HBOND', naming=naming)
    
    # Initialize the arrays for the occupation distributions (binary)
    features_data = np.zeros((len(all_pairs), len(u.trajectory)), dtype=int)
    # Go through all bonds and set the corresponding values to one
    for bond in hb:
        frame = int(bond[0])
        donor_id = bond[1]
        hydrogen_id = bond[2]
        acceptor_id = bond[3] 
        pair = [donor_id, acceptor_id]
        pair_id = np.argwhere([p == pair for p in all_pairs])[0][0]
        features_data[pair_id, frame] = 1

    return feature_names, features_data


def read_h_bond_satisfaction(structure_input, xtc_input, fixed_group, dyn_group='all', naming='plain'):
    """
    Find whether hydrogen-bond donors and acceptors in atom group 1 (fixed) are satisfied by partners in atom group 2 (dynamic).

    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    fixed_group : str
        Atomgroup selection to find bonding partners for.
    dyn_group: str
        Atomgroup selection to find bonding partners within.
    naming : str, default='plain'
        Naming scheme for each atom in the feature names.
        plain: neither chain nor segment ID included
        chainid: include chain ID (only works if chains are defined)
        segid: include segment ID (only works if segments are defined)

    Returns
    -------
        feature_names : list of str
            Names of all H-bond donors and acceptors
        features_data : numpy array
            Binary satisfaction data for all donors and acceptors 

    """

    # Find hydrogen bonds between the groups
    u = mda.Universe(structure_input, xtc_input)
    hbond = HBA(universe=u, between=[fixed_group, dyn_group])
    hbonds_mda = hbond.run()
    hb = hbonds_mda.results.hbonds

    # Find all atoms in the fixed group
    fixed_group_ids = hbond.between_ags[0][0].indices
    # Determine all donors in the fixed group
    all_donors = np.array(np.unique(hb[:,1]), dtype=int)
    donor_in_fixed = [i in fixed_group_ids for i in all_donors]
    fixed_donors = all_donors[donor_in_fixed].tolist()
    # Determine all acceptors in the fixed group
    all_acceptors = np.array(np.unique(hb[:,3]), dtype=int)
    acceptor_in_fixed = [i in fixed_group_ids for i in all_acceptors]
    fixed_acceptors = all_acceptors[acceptor_in_fixed].tolist()
    
    # Give the features the corresponding names
    donor_names = name_atom_features(u, fixed_donors, feature_type='H-DON', naming=naming)
    acceptor_names = name_atom_features(u, fixed_acceptors, feature_type='H-ACC', naming=naming)
    
    # Initialize the arrays for the occupation distributions (binary)
    donor_data = np.zeros((len(fixed_donors), len(u.trajectory)), dtype=int)
    acceptor_data = np.zeros((len(fixed_acceptors), len(u.trajectory)), dtype=int)
    
    # Go through all bonds and set the corresponding values to one
    for bond in hb:
        frame = int(bond[0])
        donor_id = bond[1]
        hydrogen_id = bond[2]
        acceptor_id = bond[3] 
        if donor_id in fixed_donors:
            donor_num = np.argwhere([donor_id == id for id in fixed_donors])[0][0]
            donor_data[donor_num, frame] = 1
        if acceptor_id in fixed_acceptors:
            acceptor_num = np.argwhere([acceptor_id == id for id in fixed_acceptors])[0][0]
            acceptor_data[acceptor_num, frame] = 1

    # Construct the output dictionaries.
    feature_names = donor_names + acceptor_names
    features_data = np.concatenate([donor_data, acceptor_data])

    return feature_names, features_data


def read_water_site_h_bonds(structure_input, xtc_input, water_o_atom_name, biomol_sel='protein', 
                            site_IDs=None, grid_input=None, write_grid_as=None, out_name=None):
    """
    Find hydrogen bonds between waters occupying cavities and protein.

    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    water_o_atom_name : str
        Atom name to calculate the density for (usually water oxygen).
    biomol_sel : str
        Selection string for the biomolecule that forms the cavity/site. The default is 'protein'
    site_IDs : list, optional
        List of indexes for the sites desired to investigate. If none is provided, all sites will be analyzed
    grid_input : str, optional
        File name for the density grid input. The default is None, and a grid is automatically generated.
    write : bool, optional
        If true, the following data will be written out: reference pdb with occupancies,
        water distributions, water data summary. The default is None.
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model
        to convert the density into. The default is None. Options are suggested if default.
    out_name : str, optional
        Prefix for all written filenames. The default is None.

    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features

    """

    # Create the universe
    u = mda.Universe(structure_input, xtc_input)

    # Create or load the density grid
    if grid_input is None:
        g = generate_grid(u, water_o_atom_name, write_grid_as, out_name)
    else:
        g = Grid(grid_input)        

    # Write the water binding sites
    if out_name is not None:
        p = u.select_atoms(biomol_sel)
        pdb_outname = out_name + "_WaterSites.pdb"
        p_avg = np.zeros_like(p.positions)
        # do a quick average of the protein (in reality you probably want to remove PBC and RMSD-superpose)
        for ts in u.trajectory:
            p_avg += p.positions
        p_avg /= len(u.trajectory)
        # temporarily replace positions with the average
        p.positions = p_avg
        # write average protein coordinates
        p.write(pdb_outname)
        # just make sure that we have clean original coordinates again (start at the beginning)
        u.trajectory.rewind()

    # Get the positions and values of the water sites
    xyz, val = local_maxima_3D(g.grid)
    # Negate the array to get probabilities in descending order
    val_sort = np.argsort(-1 * val.copy())
    coords = [xyz[max_val] for max_val in val_sort]

    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
    # Determine the water sites to analyze
    if site_IDs is None:
        site_IDs = np.arange(len(coords)) + 1

    # Go through all selected water sites
    for site_no in site_IDs:
        # Site numbers are one-based
        ID_to_idx = site_no - 1
        # Print the site number
        print('\nSite no: ', site_no, '\n')
        # Shifting the coordinates of the maxima by the grid origin to match
        # the simulation box coordinates
        shifted_coords = coords[ID_to_idx] + g.origin
        point_str = str(shifted_coords)[1:-1]
        # Write the site atom to the PDB
        if out_name is not None:
            write_atom_to_pdb(pdb_outname, shifted_coords, 'W' + str(site_no), water_o_atom_name)
        # Define the atom group for the water binding site
        # (all water atoms within 3.5 Angstroms of density maxima)
        site_water = 'byres (name ' + water_o_atom_name + ' and point ' + point_str + ' 3.5)'
        # Read H-bond donors and acceptors of the binding site that are satisfied by any water in the pocket
        hb_names, hb_data = read_h_bond_satisfaction(structure_input, xtc_input, biomol_sel, site_water)
        # append the feature names and timeseries data
        feature_names['W' + str(site_no)] = hb_names
        features_data['W' + str(site_no)] = hb_data

    return feature_names, features_data



# -----------------------------------------------
#  Functions based on a distance-only criterion
# -----------------------------------------------


def atg_to_names(atg):
    idxes = [7, 10, 2]
    all_atgs = []
    print(atg)
    for line in range(len(atg)):
        stringdex = [str(atg[line]).split(' ')[idx] for idx in idxes]
        all_atgs.append(stringdex[-1][:-1] + " " + stringdex[1] + " " + stringdex[2])
    return all_atgs


def _unique_bonding_pairs(lst):
    return ([list(i) for i in {* [tuple(sorted(i)) for i in lst]}])


def read_h_bonds_quickly(structure_input, xtc_input, fixed_group, dyn_group):
    """
    Find hydrogen bonding partners for atomgroup1 in atomgroup2.

    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    fixed_group : str
        Atomgroup selection to find bonding partners for.
    dyn_group: str
        Atomgroup selection to find bonding partners within.

    Returns
    -------
    feature_names : list of str
        Names of all bonds
    features_data : numpy array
        Data for all bonds

    """

    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}

    u = mda.Universe(structure_input, xtc_input)

    # First locate all potential bonding sites
    interacting_atoms1 = fixed_group
    # locate all potential bonding sites for atomgroups
    hbond = HBA(universe=u)
    atomgroup_donors1 = hbond.guess_hydrogens(interacting_atoms1)
    atomgroup_acceptors1 = hbond.guess_acceptors(interacting_atoms1)

    interacting_atoms2 = dyn_group + " and around 3.5 " + fixed_group
    interacting_atoms2_idx = u.select_atoms(interacting_atoms2, updating=True).indices
    int2group = 'index ' + ' or index '.join([str(ind) for ind in interacting_atoms2_idx])
    # locate all potential bonding sites for atomgroups
    hbond = HBA(universe=u)
    atomgroup_donors2 = hbond.guess_hydrogens(int2group)
    atomgroup_acceptors2 = hbond.guess_acceptors(int2group)
    # print(atomgroup_donors2)

    donor1_idcs = u.select_atoms(atomgroup_donors1).indices
    acceptor1_idcs = u.select_atoms(atomgroup_acceptors1).indices
    donor2_idcs = u.select_atoms(atomgroup_donors2).indices
    acceptor2_idcs = u.select_atoms(atomgroup_acceptors2).indices

    # First locate all potential bonding sites for atomgroups
    # bonds for [[atomgroup1 donors] , [atomgroup1 acceptors]]
    all_bonds = [[], []]
    for frame_no in tqdm(range(len(u.trajectory))):
        # find the frame
        u.trajectory[frame_no]

        # obtain indices for all donor and acceptor atoms
        frame_bonds = []
        for donor1_idx in donor1_idcs:
            idx_bonds = []        # find donor positions
            donor1_pos = np.array(u.select_atoms("index " + str(donor1_idx)).positions)
            for acceptor2_idx in acceptor2_idcs:
                # find acceptor positions
                acceptor2_pos = np.array(u.select_atoms("index " + str(acceptor2_idx)).positions)
                # if distance between atoms less than 3.5 angstrom then count as bond
                if np.linalg.norm(donor1_pos - acceptor2_pos) < 3.5:
                    idx_bonds.append([donor1_idx, acceptor2_idx])
            # print(idx_bonds)
            frame_bonds.append(idx_bonds)
            # print(frame_bonds)
            all_bonds[0].append(frame_bonds)

        frame_bonds = []
        for donor2_idx in donor2_idcs:
            idx_bonds = []
            # find donor positions
            donor2_pos = np.array(u.select_atoms("index " + str(donor2_idx)).positions)
            for acceptor1_idx in acceptor1_idcs:
                # find acceptor positions
                acceptor1_pos = np.array(u.select_atoms("index " + str(acceptor1_idx)).positions)
                # if distance between atoms less than 3.5 angstrom then count as bond
                if np.linalg.norm(donor2_pos - acceptor1_pos) < 3.5:
                    idx_bonds.append([donor2_idx, acceptor1_idx])
            # print(idx_bonds)
            frame_bonds.append(idx_bonds)
            # print(frame_bonds)
            all_bonds[1].append(frame_bonds)

    all_donor_pairs = _unique_bonding_pairs([y for subl in [x for sub in all_bonds[0] for x in sub] for y in subl])
    all_acceptor_pairs = _unique_bonding_pairs([y for subl in [x for sub in all_bonds[1] for x in sub] for y in subl])

    all_donor_pair_names = [[atg_to_names(u.select_atoms('index ' + str(i[0])))[0],
                             atg_to_names(u.select_atoms('index ' + str(i[1])))[0]] for i in all_donor_pairs]
    all_acceptor_pair_names = [[atg_to_names(u.select_atoms('index ' + str(i[0])))[0],
                                atg_to_names(u.select_atoms('index ' + str(i[1])))[0]] for i in all_acceptor_pairs]

    donor_dist = np.zeros((len(all_donor_pairs), len(u.trajectory)))
    acceptor_dist = np.zeros((len(all_acceptor_pairs), len(u.trajectory)))

    for frame in tqdm(range(len(u.trajectory))):
        for pair in range(len(all_donor_pairs)):
            if list(reversed(all_donor_pairs[pair])) in [flat for sub in all_bonds[0][frame_no] for flat in sub]:
                donor_dist[pair][frame_no] = 1
        for pair in range(len(all_acceptor_pairs)):
            if list(reversed(all_acceptor_pairs[pair])) in [flat for sub in all_bonds[1][frame_no] for flat in sub]:
                acceptor_dist[pair][frame_no] = 1

    feature_names['donor_names'] = np.array(all_donor_pair_names)
    feature_names['acceptor_names'] = np.array(all_acceptor_pair_names)
    features_data['donor_data'] = np.array(donor_dist)
    features_data['acceptor_data'] = np.array(acceptor_dist)

    return feature_names, features_data


def read_water_site_h_bonds_quickly(structure_input, xtc_input, atomgroups, site_IDs,
                                    grid_input=None, write=None, write_grid_as=None, out_name=None):
    """
    Find hydrogen bonds between waters occupying cavities and protein.

    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    site_IDs : list
        List of indexes for the sites desired to investigate.
    grid_input : str, optional
        File name for the density grid input. The default is None, and a grid is automatically generated.
    write : bool, optional
        If true, the following data will be written out: reference pdb with occupancies,
        water distributions, water data summary. The default is None.
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model
        to convert the density into. The default is None. Options are suggested if default.
    out_name : str, optional
        Prefix for all written filenames. The default is None.

    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features

    """

    if write is not None:
        if out_name is None:
            print('WARNING: You are writing results without providing out_name.')

    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}

    u = mda.Universe(structure_input, xtc_input)
    if write is True:
        if not os.path.exists('h2o_hbonds/'):
            os.makedirs('h2o_hbonds/')
        p = u.select_atoms("protein")
        pdb_outname = 'h2o_hbonds/' + out_name + "_Sites.pdb"
        p_avg = np.zeros_like(p.positions)
        # do a quick average of the protein (in reality you probably want to remove PBC and RMSD-superpose)
        for ts in u.trajectory:
            p_avg += p.positions
        p_avg /= len(u.trajectory)
        # temporarily replace positions with the average
        p.positions = p_avg
        # write average protein coordinates
        p.write(pdb_outname)
        # just make sure that we have clean original coordinates again (start at the beginning)
        u.trajectory.rewind()

        if grid_input is None:
            g = generate_grid(u, atomgroups[0], write_grid_as, out_name)
        else:
            g = Grid(grid_input)
    elif grid_input is None:
        g = generate_grid(u, atomgroups)
    else:
        g = Grid(grid_input)

    xyz, val = local_maxima_3D(g.grid)
    # Negate the array to get probabilities in descending order
    val_sort = np.argsort(-1 * val.copy())
    coords = [xyz[max_val] for max_val in val_sort]
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    site_information = []
    O_hbonds_all_site = []
    H_hbonds_all_site = []

    for site_no in site_IDs:

        feature_names['W' + str(site_no)] = {}
        features_data['W' + str(site_no)] = {}

        ID_to_idx = site_no - 1
        print('\n')
        print('Site no: ', site_no)
        print('\n')
        O_hbonds = []
        H_hbonds = []
        # Find all water atoms within 3.5 Angstroms of density maxima
        # Shifting the coordinates of the maxima by the grid origin to match
        # the simulation box coordinates
        shifted_coords = coords[ID_to_idx] + g.origin
        point_str = str(shifted_coords)[1:-1]
        counting = []

        if write is True:
            write_atom_to_pdb(pdb_outname, shifted_coords, 'W' + str(site_no), atomgroups[0])

        for frame_no in tqdm(range(len(u.trajectory))):
            u.trajectory[frame_no]
            radius = ' 3.5'
            at_sel = 'byres (name ' + atomgroups[0] + ' and point ' + point_str + radius + ')'
            atomgroup_IDS = list(u.select_atoms(at_sel).indices)[::3]
            counting.append(list(set(atomgroup_IDS)))

        # Water atom indices that appear in the water site
        flat_list = [item for sublist in counting for item in sublist]

        # Extract water orientation timeseries
        for frame_no in tqdm(range(len(u.trajectory))):

            u.trajectory[frame_no]
            site_resid = counting[frame_no]
            # print(site_resid)

            if len(site_resid) == 1:
                # (x, y, z) positions for the water oxygen at trajectory frame_no
                resid = str(site_resid[0])
                proxprotatg = 'protein and around 3.5 byres index ' + resid
                O_site = 'index ' + resid
                H_site = '((byres index ' + resid + ') and (name ' + atomgroups[1] + ' or name ' + atomgroups[2] + '))'

                hbond = HBA(universe=u)
                protein_hydrogens_sel = hbond.guess_hydrogens(proxprotatg)
                protein_acceptors_sel = hbond.guess_acceptors(proxprotatg)
                # bonds formed by the water oxygen
                if len(protein_hydrogens_sel) != 0:
                    O_bonds = '( ' + protein_hydrogens_sel + ' ) and around 3.5 ' + O_site
                else:
                    O_bonds = ''
                # bonds formed by the water hydrogens
                if len(protein_acceptors_sel) != 0:
                    H_bonds = '( ' + protein_acceptors_sel + ' ) and around 3.5 ' + H_site
                else:
                    H_bonds = ''

                H_hbond_group = u.select_atoms(H_bonds)
                H_hbonds.append(H_hbond_group)
                O_hbond_group = u.select_atoms(O_bonds)
                O_hbonds.append(O_hbond_group)

            # Featurize water with highest pocket occupation (if multiple waters in pocket)
            elif len(site_resid) > 1:
                freq_count = []
                for ID in site_resid:
                    freq_count.append([flat_list.count(ID), ID])
                freq_count.sort(key=lambda x: x[0])

                fcstr = str(freq_count[-1][1])
                proxprotatg = 'protein and around 3.5 byres index ' + fcstr
                O_site = 'index ' + fcstr
                H_site = '((byres index ' + fcstr + ') and (name ' + atomgroups[1] + ' or name ' + atomgroups[2] + '))'

                hbond = HBA(universe=u)
                protein_hydrogens_sel = hbond.guess_hydrogens(proxprotatg)
                protein_acceptors_sel = hbond.guess_acceptors(proxprotatg)
                # bonds formed by the water oxygen
                if len(protein_hydrogens_sel) != 0:
                    O_bonds = '( ' + protein_hydrogens_sel + ' ) and around 3.5 ' + O_site
                else:
                    O_bonds = ''
                # bonds formed by the water hydrogens
                if len(protein_acceptors_sel) != 0:
                    H_bonds = '( ' + protein_acceptors_sel + ' ) and around 3.5 ' + H_site
                else:
                    H_bonds = ''

                H_hbond_group = u.select_atoms(H_bonds)
                H_hbonds.append(H_hbond_group)
                O_hbond_group = u.select_atoms(O_bonds)
                O_hbonds.append(O_hbond_group)

            # 10000.0 = no waters bound
            elif len(site_resid) < 1:
                O_hbonds.append("unocc")
                H_hbonds.append("unocc")

        bondouts = []
        for bondtype in [O_hbonds, H_hbonds]:
            resids = []
            for line in bondtype:
                if type(line) is str:
                    resids.append([line])
                else:
                    idxes = [8, 10, 2]
                    all_atgs = []
                    for atg in range(len(line)):
                        # print(line[atg])
                        stringdex = [str(line[atg]).split(' ')[idx] for idx in idxes]
                        all_atgs.append(stringdex[0][:-1] + " " + stringdex[1] + " " + stringdex[2])
                    resids.append(all_atgs)

            names = list(set([flat for sub in resids for flat in sub]))
            if names.count('unocc') > 0:
                names.remove('unocc')
            dist = np.zeros((len(names), len(u.trajectory)))
            for bondsite in range(len(names)):
                for frame in range(len(resids)):
                    if resids[frame].count(names[bondsite]) > 0:
                        dist[bondsite][frame] = 1
            bondouts.append([names, dist])

        O_site_pdb_id = "O" + str(site_no)
        H_site_pdb_id = "H" + str(site_no)
        # Write data out and visualize water sites in pdb
        # "FIX OUTPUT UNIFORMITY, SINGLE BONDS NOT OUTPUT WITH ANY ARRAY DIMENSION"
        if write is True:
            np.savetxt(
                'h2o_hbonds/' + out_name + O_site_pdb_id + '_names.txt',
                np.array(bondouts[0][0], dtype=object), fmt='%s'
            )
            np.savetxt(
                'h2o_hbonds/' + out_name + O_site_pdb_id + '_data.txt',
                np.array(bondouts[0][1], dtype=object), fmt='%s'
            )
            np.savetxt(
                'h2o_hbonds/' + out_name + H_site_pdb_id + '_names.txt',
                np.array(bondouts[1][0], dtype=object), fmt='%s'
            )
            np.savetxt(
                'h2o_hbonds/' + out_name + H_site_pdb_id + '_data.txt',
                np.array(bondouts[1][1], dtype=object), fmt='%s'
            )

        feature_names['W' + str(site_no)]['acceptor_names'] = \
            np.array(bondouts[0][0], dtype=object)
        feature_names['W' + str(site_no)]['donor_names'] = \
            np.array(bondouts[1][0], dtype=object)
        features_data['W' + str(site_no)]['acceptor_timeseries'] = \
            np.array(bondouts[0][1], dtype=object)
        features_data['W' + str(site_no)]['donor_timeseries'] = \
            np.array(bondouts[1][1], dtype=object)
        features_data['W' + str(site_no)]['acceptor_frequencies'] = \
            np.sum(np.array(bondouts[0][1], dtype=object), axis=1) / len(u.trajectory)
        features_data['W' + str(site_no)]['donor_frequencies'] = \
            np.sum(np.array(bondouts[1][1], dtype=object), axis=1) / len(u.trajectory)

    return feature_names, features_data
