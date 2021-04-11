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
Co-SSI between all features , with the names_bbtors producing the same output as 
the feature-feature SSI. These results can again be visualized in a 2D representation.

