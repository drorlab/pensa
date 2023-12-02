Tracing Information Flow
========================

Here we trace infromation flow throughout the receptor using SSI analysis tools. 
The SSI resolved in the ensemble comparison concludes which features couple their
torsion angles to the ensemble conditions, thereby receiving information of the ensemble switch. 
However, to trace information flow we need to ensure that features couple to 
each other as well as to the ensemble condition.

We can trace information flow by calculating the feature-feature SSI using 
the function ``ssi_feature_analysis()``.

The Co-SSI statistic, which tells how much the communication between two features 
depends on the ensemble condition, can be calculated using ``cossi_featens_analysis()``.

Features and States
-------------------

.. code:: python

    import os
    import numpy as np
    from pensa.features import \
        read_structure_features, \
        read_water_features, \
        get_multivar_res, \
        sort_features
    from pensa.statesinfo import \
        get_discrete_states
    from pensa.comparison import \
        ssi_ensemble_analysis, \
        ssi_feature_analysis, \
        cossi_featens_analysis

First, load the structural features as described in the previous tutorial:

.. code:: python

    sim_a_rec = read_structure_features(
        "traj/condition-a_receptor.gro",
        "traj/condition-a_receptor.xtc"
    )
    sim_b_rec = read_structure_features(
        "traj/condition-b_receptor.gro",
        "traj/condition-b_receptor.xtc"
    )
    sim_a_rec_feat, sim_a_rec_data = sim_a_rec
    sim_b_rec_feat, sim_b_rec_data = sim_b_rec


Then combine all backbone torsions from one residue to one multivariate feature. 
You can also analyze sidechain torsions, just replace ``bb`` by ``sc``. Analysis of 
sidechain torsions takes much more time to run though so for this tutorial, we look
only at backbone torsions.

.. code:: python

    bbtors_res_feat_a, bbtors_res_data_a = get_multivar_res(
        sim_a_rec_feat['bb-torsions'], sim_a_rec_data['bb-torsions'] 
    )
    bbtors_res_feat_b, bbtors_res_data_b = get_multivar_res(
        sim_b_rec_feat['bb-torsions'], sim_b_rec_data['bb-torsions']
    )

From those residue-based multivariate features, we now determine the state boundaries.

.. code:: python

    bbtors_states = get_discrete_states(
        bbtors_res_data_a, bbtors_res_data_b
    )

Let's also featurize the water cavities in the system, using the grid we generated in
the preprocessing tutorial.

.. code:: python

    grid = "traj/water_grid_ab_OH2_density.dx"
    water_feat_a, water_data_a = read_water_features(
        "traj/condition-a_water.gro", "traj/condition-a_water_aligned.xtc",
        top_waters = 5, atomgroup = "OH2", grid_input = grid
    )
    water_feat_b, water_data_b = read_water_features(
        "traj/condition-b_water.gro", "traj/condition-b_water.xtc",
        top_waters = 5, atomgroup = "OH2", grid_input = grid
    )

Just like we did for torsions, we can now determine the discrete states for the orientation
of the water molecules in the pockets.

.. code:: python

    water_states = get_discrete_states(
        water_data_a['WaterPocket_Distr'],
        water_data_b['WaterPocket_Distr'],
        discretize='gaussian', pbc=True
    )

Water occupancy (is water in the pocket or not?) is described as a binary feature with the 
values 0 or 1. We can thus define the state boundaries manually.

.. code:: python

    water_occup_states = [[[-0.1, 0.5, 1.1]]] * len(water_states)


Water Pockets
-------------

We start by comparing the occupancy between the two conditions, similar to what we did in the
comparison tutorial:

.. code:: python

    water_names, water_ssi = ssi_ensemble_analysis(
        water_feat_a['WaterPocket_OccupDistr'], water_feat_b['WaterPocket_OccupDistr'], 
        water_data_a['WaterPocket_OccupDistr'], water_data_b['WaterPocket_OccupDistr'],
        water_occup_states, verbose=True, h2o=False, pbc=False
    )

Except for water site O2, there are not many differences in occupancy, so let's have a closer 
look and compare the orientation of the water molecules between the two conditions: 

.. code:: python

    water_names, water_ssi = ssi_ensemble_analysis(
        water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'], 
        water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
        water_states, verbose=True, h2o=True
    )

Beyond comparing distributions, we can also quantify the amount of feature-feature communication
between the water sites. The corresponding function looks similar to the ensemble comarison above,
however it calculates the amount of shared information between each pair of residues.

.. code:: python

    water_pairs_names, water_pairs_ssi = ssi_feature_analysis(
        water_feat_a['WaterPocket_Distr'], water_feat_b['WaterPocket_Distr'], 
        water_data_a['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'],
        water_states, verbose=True, h2o=True
    )

This function produces an array for the SSI between all features. This array is two-dimensional,
so we visualize its values using a heat map:

.. code:: python

    from pensa.comparison import resnum_heatmap, pair_features_heatmap

    pair_features_heatmap(
        water_pairs_names, water_pairs_ssi,
        "plots/water-pairs_ssi.pdf",
        vmin = 0.0, vmax = 1.,
        cbar_label='SSI',
        separator=' & '
    )


Water and Torsions Combined
---------------------------

Now we want to investigate information flow between more than one type of features.
To do so, we combine the water sites and the torsions:

.. code:: python

    all_feat_a = water_feat_a['WaterPocket_Distr'] + bbtors_res_feat_a
    all_feat_b = water_feat_b['WaterPocket_Distr'] + bbtors_res_feat_b
    all_data_a = np.array(list(water_data_a['WaterPocket_Distr']) + list(bbtors_res_data_a), dtype=object)
    all_data_b = np.array(list(water_data_b['WaterPocket_Distr']) + list(bbtors_res_data_b), dtype=object)
    all_states = water_states + bbtors_states

Note that we only use backbone torsions here to minimize computational effort.
Analysis of sidechain torsions (or of both combined) can often deliver more scientific insigts.

As we did for the water sites above, we now calculate the SSI for all combined feature-feature pairs.

.. code:: python

    all_pairs_names, all_pairs_ssi = ssi_feature_analysis(
        all_feat_a, all_feat_b, 
        all_data_a, all_data_b,
        all_states, verbose=True
    )

The number of pairs for an entire protein is enormous so we determine those with 
the highest SSI. To alleviate the computational effort for the sort function, we first 
filter the pairs by a threshold SSI of 0.5:

.. code:: python

    relevant = np.abs(all_pairs_ssi) > 0.5 
    not_self = np.array([name.split(' & ')[0] != name.split(' & ')[1] for name in all_pairs_names])
    relevant *= not_self
    argrelev = np.argwhere(relevant).flatten()
    all_relevant_pairs_names = [all_pairs_names[i] for i in argrelev]
    all_relevant_pairs_ssi = all_pairs_ssi[relevant]

Then we run the actual sorting by SSI.

.. code:: python

    sort_features(all_relevant_pairs_names, all_relevant_pairs_ssi)


The Co-SSI feature-feature-ensemble analysis is done in the same manner. 

.. code:: python

    all_new_pairs_names, all_new_pairs_ssi, all_new_pairs_cossi = cossi_featens_analysis(
        all_feat_a, all_feat_b, all_feat_a, all_feat_b,
        all_data_a, all_data_b, all_data_a, all_data_b,
        all_states, all_states, verbose=True
    )
                                             
The output of ``cossi_featens_analysis()`` produces an array for the SSI and the Co-SSI
between all features. These results can again be visualized in a 2D representation.
