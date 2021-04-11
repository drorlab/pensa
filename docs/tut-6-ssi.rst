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

Feature - Feature Communication
-------------------------------
First, load the structural features as described in the previous tutorial:

.. code:: python

    sim_a_rec = get_structure_features("traj/condition-a_receptor.gro", 
                                       "traj/condition-a_receptor.xtc")
    sim_b_rec = get_structure_features("traj/condition-b_receptor.gro",
                                       "traj/condition-b_receptor.xtc")
    sim_a_rec_feat, sim_a_rec_data = sim_a_rec
    sim_b_rec_feat, sim_b_rec_data = sim_b_rec


Then run the SSI feature-feature analysis in the same manner the other statistics.


.. code:: python

    names_bbtors, ssi_featfeat_bbtors = ssi_feature_analysis(sim_a_rec_feat, sim_b_rec_feat,
                                                             sim_a_rec_data, sim_b_rec_data,
                                                             torsions='bb', verbose=True)
                                             
The output of ``ssi_feature_analysis()`` produces an array for the SSI between
all features, with the names_bbtors referring to both feature names. This result 
can be visualized in a two-dimensional representation similar to the distances 
using ``distances_visualization()``

.. code:: python

    distances_visualization(names_bbtors, ssi_featfeat_bbtors,
                            "plots/receptor_ssi-bbdist.pdf",
                            vmin = 0.0, vmax = max(ssi_featfeat_bbtors),
                            cbar_label='SSI')

The Co-SSI feature-feature-ensemble analysis is done in the same manner. 

.. code:: python

    names_bbtors, ssi_featfeat_bbtors, cossi_bbtors = cossi_featens_analysis(sim_a_rec_feat, sim_b_rec_feat,
                                                                             sim_a_rec_data, sim_b_rec_data,
                                                                             torsions='bb', verbose=True)
                                             
The output of ``cossi_featens_analysis()`` produces an array for the SSI and the 
Co-SSI between all features, with the names_bbtors producing the same output as 
the feature-feature SSI. These results can again be visualized in a 2D representation. 
It is worth noting, as ``cossi_featens_analysis()`` also computes SSI, you need not 
run ``ssi_feature_analysis()`` in addition.

Water pocket information transfer
---------------------------------

Internal water pockets and ion/atom pockets can also receive information about 
the ensemble condition. The State Specific Information (SSI) analysis can be 
applied to investigate these distributions in the same manner. First we need 
to featurize the water pockets using the combined ensemble water density explained 
in the featurization tutorial. 

.. code:: python

    struc = "traj/cond-a_water.gro"
    xtc = "dens/cond-a_wateraligned.xtc"
    grid = "dens/ab_grid_OH2_density.xtc"
    water_feat_a, water_data_a = get_water_features(structure_input = struc, 
                                                xtc_input = xtc,
                                                top_waters = 5,
                                                atomgroup = "OH2",
                                                grid_input = grid)

    struc = "traj/cond-b_water.gro"
    xtc = "traj/cond-b_water.xtc"
    grid = "dens/ab_grid_OH2_density.xtc"
    water_feat_b, water_data_b = get_water_features(structure_input = struc, 
                                                xtc_input = xtc,
                                                top_waters = 5,
                                                atomgroup = "OH2",
                                                grid_input = grid)
                                                
    
Information shared between water pockets and the ensemble condition is then 
quantified using ``ssi_ensemble_analysis()``. We set ``torsions = None`` for 
waters. 

.. code:: python

    data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_Distr'],water_feat_b['WaterPocket_Distr'],
                                                 water_data_a['WaterPocket_Distr'],water_data_b['WaterPocket_Distr'], 
                                                 torsions = None,
                                                 verbose=True)                                                
 

Additionally, we can see if the pocket occupancy (i.e. the presence/absence of 
water at the site) shares SSI. Currently this is only enabled with 
``ssi_ensemble_analysis``. We need to turn off the periodic boundary conditions
as the distributions are no longer periodic.

.. code:: python

    data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_OccupDistr'],water_feat_b['WaterPocket_OccupDistr'],
                                                 water_data_a['WaterPocket_OccupDistr'],water_data_b['WaterPocket_OccupDistr'],
                                                 wat_occupancy=True, pbc=False, verbose=True)











