import unittest
import gc
import os
import matplotlib.pyplot as plt
import numpy as np

from pensa.clusters import \
    obtain_clusters, wss_over_number_of_clusters, \
    obtain_combined_clusters, wss_over_number_of_combined_clusters, \
    write_cluster_traj

from pensa.statesinfo import \
    get_discrete_states

from pensa.comparison import \
    relative_entropy_analysis, relen_block_analysis, relen_sem_analysis, \
    ssi_ensemble_analysis, ssi_block_analysis, ssi_sem_analysis, \
    residue_visualization, distances_visualization, \
    pca_feature_correlation, tica_feature_correlation

from pensa.features import \
    read_structure_features, \
    get_multivar_res_timeseries, \
    sort_features

from pensa.dimensionality import \
    calculate_pca, pca_eigenvalues_plot, \
    calculate_tica, tica_eigenvalues_plot, \
    sort_traj_along_pc, sort_trajs_along_common_pc, \
    sort_traj_along_tic, sort_trajs_along_common_tic, \
    compare_projections

from pensa.preprocessing import \
    load_selection, extract_coordinates, extract_coordinates_combined


# Location of the data used in the tests
test_data_path = './tests/test_data'

# Locations for the output files
for subdir in ['traj', 'plots', 'vispdb', 'pca', 'tica', 'clusters', 'results']:
    if not os.path.exists(test_data_path + '/' + subdir):
        os.makedirs(test_data_path + '/' + subdir)


# ** CLASS WITH ALL TEST FUNCTIONS ** #

class Test_pensa(unittest.TestCase):

    # EXAMPLE WORKFLOW WITH REUSABLE DATA

    def setUp(self):

        # - PREPROCESSING

        # Set root directory for each simulation
        root_dir_a = test_data_path + '/MOR-apo'
        root_dir_b = test_data_path + '/MOR-BU72'
        # Simulation A
        self.ref_file_a = root_dir_a + '/mor-apo.psf'
        self.pdb_file_a = root_dir_a + '/mor-apo.pdb'
        self.trj_file_a = [root_dir_a + '/mor-apo-1.xtc',
                           root_dir_a + '/mor-apo-2.xtc',
                           root_dir_a + '/mor-apo-3.xtc']
        # Simulation B
        self.ref_file_b = root_dir_b + '/mor-bu72.psf'
        self.pdb_file_b = root_dir_b + '/mor-bu72.pdb'
        self.trj_file_b = [root_dir_b + '/mor-bu72-1.xtc',
                           root_dir_b + '/mor-bu72-2.xtc',
                           root_dir_b + '/mor-bu72-3.xtc']

        # Names of the output files
        self.trj_name_a = test_data_path + "/traj/condition-a"
        self.trj_name_b = test_data_path + "/traj/condition-b"

        # First test case: entire receptor.

        # Base for the selection string for each simulation
        sel_base_a = "(not name H*) and protein"
        sel_base_b = "(not name H*) and protein"

        # Extract the coordinates of the entire receptors
        self.file_extr_a = extract_coordinates(
            self.ref_file_a, self.pdb_file_a, self.trj_file_a,
            self.trj_name_a + "_receptor", sel_base_a
        )
        self.file_extr_b = extract_coordinates(
            self.ref_file_b, self.pdb_file_b, self.trj_file_b,
            self.trj_name_b + "_receptor", sel_base_b
        )

        # Second test case: only transmembrane (TM) region.

        # Generate selection strings from the file
        sel_string_a = load_selection(
            test_data_path + "/mor_tm.txt", sel_base_a + " and "
        )
        sel_string_b = load_selection(
            test_data_path + "/mor_tm.txt", sel_base_b + " and "
        )

        # Extract the coordinates of the TM region from the trajectory
        self.file_tm_single_a = extract_coordinates(
            self.ref_file_a, self.pdb_file_a, [self.trj_file_a],
            self.trj_name_a + "_tm", sel_string_a
        )
        self.file_tm_single_b = extract_coordinates(
            self.ref_file_b, self.pdb_file_b, [self.trj_file_b],
            self.trj_name_b + "_tm", sel_string_b
        )
        self.file_tm_combined = extract_coordinates_combined(
            [self.ref_file_a] * 3 + [self.ref_file_b] * 3,
            self.trj_file_a + self.trj_file_b,
            [sel_string_a] * 3 + [sel_string_b] * 3,
            test_data_path + '/traj/combined_tm',
            start_frame=0
        )

        # - FEATURES -

        self.start_frame = 10

        # -- Receptor features --
        sim_a_rec = read_structure_features(
            test_data_path + "/traj/condition-a_receptor.gro",
            test_data_path + "/traj/condition-a_receptor.xtc",
            start_frame=self.start_frame
        )
        sim_b_rec = read_structure_features(
            test_data_path + "/traj/condition-b_receptor.gro",
            test_data_path + "/traj/condition-b_receptor.xtc",
            start_frame=self.start_frame
        )
        self.sim_a_rec_feat, self.sim_a_rec_data = sim_a_rec
        self.sim_b_rec_feat, self.sim_b_rec_data = sim_b_rec

        # -- TMR features --
        sim_a_tmr = read_structure_features(
            test_data_path + "/traj/condition-a_tm.gro",
            test_data_path + "/traj/condition-a_tm.xtc",
            start_frame=self.start_frame
        )
        sim_b_tmr = read_structure_features(
            test_data_path + "/traj/condition-b_tm.gro",
            test_data_path + "/traj/condition-b_tm.xtc",
            start_frame=self.start_frame
        )
        self.sim_a_tmr_feat, self.sim_a_tmr_data = sim_a_tmr
        self.sim_b_tmr_feat, self.sim_b_tmr_data = sim_b_tmr

        # -- Discrete States --

        # --- Multivariate Features
        out_name_a = "condition-a"
        out_name_b = "condition-b"
        self.sim_a_rec_multivar_feat, self.sim_a_rec_multivar_data = get_multivar_res_timeseries(
            self.sim_a_rec_feat, self.sim_a_rec_data, 'sc-torsions', write=True, out_name=out_name_a
        )
        self.sim_b_rec_multivar_feat, self.sim_b_rec_multivar_data = get_multivar_res_timeseries(
            self.sim_b_rec_feat, self.sim_b_rec_data, 'sc-torsions', write=True, out_name=out_name_b
        )

        # --- Gaussian Discretization
        self.discrete_states_ab = get_discrete_states(
            self.sim_a_rec_multivar_data['sc-torsions'], self.sim_b_rec_multivar_data['sc-torsions'],
            discretize='gaussian', pbc=True
        )

        # -- Relative Entropy --

        # --- BB Torsions
        self.relen_bbtor = relative_entropy_analysis(
            self.sim_a_rec_feat['bb-torsions'], self.sim_b_rec_feat['bb-torsions'],
            self.sim_a_rec_data['bb-torsions'], self.sim_b_rec_data['bb-torsions'],
            bin_num=10, verbose=False
        )
        self.names_bbtors, self.jsd_bbtors, self.kld_ab_bbtors, kld_ba_bbtors = self.relen_bbtor

        # --- SC Torsions
        self.relen_sctor = relative_entropy_analysis(
            self.sim_a_rec_feat['sc-torsions'], self.sim_b_rec_feat['sc-torsions'],
            self.sim_a_rec_data['sc-torsions'], self.sim_b_rec_data['sc-torsions'],
            bin_num=10, verbose=False
        )
        self.names_sctors, self.jsd_sctors, self.kld_ab_sctors, kld_ba_sctors = self.relen_sctor

        # --- Distance
        self.relen_dist = relative_entropy_analysis(
            self.sim_a_rec_feat['bb-distances'], self.sim_b_rec_feat['bb-distances'],
            self.sim_a_rec_data['bb-distances'], self.sim_b_rec_data['bb-distances'],
            bin_num=10, verbose=False
        )
        self.names_bbdist, self.jsd_bbdist, self.kld_ab_bbdist, self.kld_ba_bbdist = self.relen_dist

        # - PCA AND TICA -

        bbt_a = self.sim_a_tmr_data['bb-torsions']
        bbt_b = self.sim_b_tmr_data['bb-torsions']
        combined_data_tors = np.concatenate([bbt_a, bbt_b], 0)

        self.pca_combined = calculate_pca(combined_data_tors)
        self.tica_bbt_a = calculate_tica(bbt_a)
        self.tica_bbt_b = calculate_tica(bbt_b)

        # -- PCA features
        self.graph, self.corr = pca_feature_correlation(
            self.sim_a_tmr_feat['bb-torsions'], combined_data_tors,
            pca=self.pca_combined, num=3, threshold=0.4,
            plot_file=test_data_path + "/plots/pca-features_bbtors_a.pdf"
        )
        plt.close()

        # -- Sort trajectory pc
        pca_a = calculate_pca(self.sim_a_tmr_data['bb-torsions'])
        plt.close()
        self.all_sort, _, _ = sort_traj_along_pc(
            self.sim_a_tmr_data['bb-torsions'],
            test_data_path + "/traj/condition-a_receptor.gro",
            test_data_path + "/traj/condition-a_receptor.xtc",
            test_data_path + "/pca/condition-a_receptor_by_tmr",
            pca=pca_a, num_pc=3
        )

        # -- Compare projections
        self.val = compare_projections(
            self.sim_a_tmr_data['bb-torsions'],
            self.sim_b_tmr_data['bb-torsions'],
            self.pca_combined,
            label_a='A', label_b='B'
        )
        plt.close()

        # - CLUSTERING -

        self.cidx, self.cond, self.oidx, self.wss, self.centroids = obtain_combined_clusters(
            self.sim_a_tmr_data['bb-torsions'], self.sim_b_tmr_data['bb-torsions'],
            label_a='A', label_b='B', start_frame=self.start_frame,
            algorithm='kmeans', max_iter=100, num_clusters=3, min_dist=12,
            saveas=test_data_path + '/plots/combined_clust_bbtors.pdf'
        )

        # -- Write clusters as trajectory
        name = "condition-a_tm"
        self.atom_group = write_cluster_traj(
            self.cidx[self.cond == 0],
            test_data_path + "/traj/" + name + ".gro", test_data_path + "/traj/" + name + ".xtc",
            test_data_path + "/clusters/" + "combined_clust_bbtors_" + name,
            self.start_frame
        )

    # ** COORDINATES AND FEATURES **

    # -- extract_coordinates() and extract_coordinates_combined()
    def test_01_extract_coordinates(self):
        # Number of atoms from selection in first test case
        self.assertEqual(self.file_extr_a, 2322)
        self.assertEqual(self.file_extr_b, 2322)
        # Number of Atom from the selction
        self.assertEqual(self.file_tm_single_a, 1877)
        self.assertEqual(self.file_tm_single_b, 1877)
        # Number of Atom from the selection - combined
        self.assertEqual(self.file_tm_combined, 1877)

    # -- load_selection()
    def test_02_load_selection(self):
        sel_base_a = "(not name H*) and protein"
        sel_base_b = "(not name H*) and protein"
        sel_string_a = load_selection(test_data_path + "/mor_tm.txt", sel_base_a + " and ")
        sel_string_b = load_selection(test_data_path + "/mor_tm.txt", sel_base_b + " and ")
        self.assertEqual(sel_string_a, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')
        self.assertEqual(sel_string_b, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')

    # -- get_features()
    def test_03_get_feature(self):
        # Simulation A, full receptor
        self.assertEqual(len(self.sim_a_rec_feat['bb-torsions']), 574)
        self.assertEqual(len(self.sim_a_rec_feat['sc-torsions']), 527)
        self.assertEqual(self.sim_a_rec_data['bb-torsions'].shape, (20, 574))
        self.assertEqual(self.sim_a_rec_data['sc-torsions'].shape, (20, 527))
        # Simulation B, full receptor
        self.assertEqual(len(self.sim_b_rec_feat['bb-torsions']), 574)
        self.assertEqual(len(self.sim_b_rec_feat['sc-torsions']), 527)
        self.assertEqual(self.sim_b_rec_data['bb-torsions'].shape, (20, 574))
        self.assertEqual(self.sim_b_rec_data['sc-torsions'].shape, (20, 527))
        # Simulation A, transmembrane region
        self.assertEqual(len(self.sim_a_tmr_feat['bb-torsions']), 448)
        self.assertEqual(len(self.sim_a_tmr_feat['sc-torsions']), 423)
        self.assertEqual(self.sim_a_tmr_data['bb-torsions'].shape, (20, 448))
        self.assertEqual(self.sim_a_tmr_data['sc-torsions'].shape, (20, 423))
        # Simulation B, transmembrane region
        self.assertEqual(len(self.sim_b_tmr_feat['bb-torsions']), 448)
        self.assertEqual(len(self.sim_b_tmr_feat['sc-torsions']), 423)
        self.assertEqual(self.sim_b_tmr_data['bb-torsions'].shape, (20, 448))
        self.assertEqual(self.sim_b_tmr_data['sc-torsions'].shape, (20, 423))

    # ** ENSEMBLE COMPARISON **

    # -- relative_entropy_analysis()
    def test_04_relative_entropy_analysis(self):
        self.assertEqual(len(self.relen_bbtor[1]), 574)
        self.assertEqual(len(self.relen_bbtor[0]), 574)

    # -- relen_sem_analysis()
    def test_05_relen_sem_analysis(self):
        # --- Uncertainty for torsions
        self.relen_bbtor_blocks = relen_block_analysis(
            self.sim_a_rec_feat['bb-torsions'], self.sim_b_rec_feat['bb-torsions'],
            self.sim_a_rec_data['bb-torsions'], self.sim_b_rec_data['bb-torsions'],
            blockanlen=10, cumdist=False, verbose=True
        )
        self.relen_bbtor_sem = relen_sem_analysis(self.relen_bbtor_blocks, write_plot=False)
        # --- Uncertainty for distances
        self.relen_dist_blocks = relen_block_analysis(
            self.sim_a_rec_feat['bb-distances'], self.sim_b_rec_feat['bb-distances'],
            self.sim_a_rec_data['bb-distances'], self.sim_b_rec_data['bb-distances'],
            blockanlen=10, cumdist=False, verbose=True
        )
        self.relen_dist_sem = relen_sem_analysis(self.relen_dist_blocks, write_plot=False)
        print('SEM FOR TORSIONS')
        resrelenvals, avresrelenvals, avsemvals = self.relen_bbtor_sem
        print('resrelenvals:')
        print(resrelenvals)
        print('avresrelenvals:')
        print(avresrelenvals)
        print('avsemvals:')
        print(avsemvals)
        resrelenvals, avresrelenvals, avsemvals = self.relen_dist_sem
        print('SEM FOR DISTANCES')
        print('resrelenvals:')
        print(resrelenvals)
        print('avresrelenvals:')
        print(avresrelenvals)
        print('avsemvals:')
        print(avsemvals)

    # -- ssi_ensemble_analysis()
    def test_06_ssi_ensemble_analysis(self):
        # --- Torsions
        ssi_sctor = ssi_ensemble_analysis(
            self.sim_a_rec_multivar_feat['sc-torsions'], self.sim_b_rec_multivar_feat['sc-torsions'],
            self.sim_a_rec_multivar_data['sc-torsions'], self.sim_b_rec_multivar_data['sc-torsions'],
            self.discrete_states_ab, verbose=False
        )
        names_sctors, ssi_sctors = ssi_sctor

    # -- ssi_sem_analysis()
    def test_07_ssi_sem_analysis(self):
        # --- Uncertainty for torsions
        ssi_names, ssi_sctor_blocks = ssi_block_analysis(
            self.sim_a_rec_feat['sc-torsions'], self.sim_b_rec_feat['sc-torsions'],
            self.sim_a_rec_data['sc-torsions'], self.sim_b_rec_data['sc-torsions'],
            blockanlen=100, pbc=True, discretize='gaussian', group_feat=True, cumdist=False, verbose=True
        )
        ssi_sctor_sem = ssi_sem_analysis(
            ssi_names, ssi_sctor_blocks, write_plot=False
        )

    # -- sort_features()
    def test_08_sort_features(self):
        sf = sort_features(self.names_bbtors, self.jsd_bbtors)
        self.assertEqual(len(sf), 574)

    # -- residue_visualization()
    def test_09_residue_visualization(self):
        ref_filename = test_data_path + "/traj/condition-a_receptor.gro"
        out_filename = "receptor_bbtors-deviations_tremd"
        vis = residue_visualization(
            self.names_bbtors, self.jsd_bbtors, ref_filename,
            test_data_path + "/plots/" + out_filename + "_jsd.pdf",
            test_data_path + "/vispdb/" + out_filename + "_jsd.pdb",
            y_label='max. JS dist. of BB torsions'
        )
        self.assertEqual(len(vis), 2)
        self.assertEqual(len(vis[0]), 288)
        self.assertEqual(len(vis[1]), 288)
        plt.close()
        del vis

    # -- distances_visualization()
    def test_10_distances_visualization(self):
        matrix = distances_visualization(
            self.names_bbdist, self.jsd_bbdist,
            test_data_path + "/plots/receptor_jsd-bbdist.pdf",
            vmin=0.0, vmax=1.0
        )
        self.assertEqual(len(matrix), 288)
        for i in range(len(matrix)):
            self.assertEqual(len(matrix[i]), 288)
            self.assertEqual(matrix[i][i], 0)
        plt.close()
        del matrix

    # ** DIMENSIONALITY **

    # -- calculate_pca()
    def test_11_calculate_pca(self):
        self.assertEqual(len(self.pca_combined.mean_), 448)
        self.assertEqual(self.pca_combined.get_covariance().shape[0], 448)

    # -- calculate_tica
    def test_12_calculate_tica(self):
        self.assertEqual(self.tica_bbt_a.koopman_matrix.size, 361)
        self.assertEqual(self.tica_bbt_b.koopman_matrix.size, 361)

    # -- pca_eigenvalues_plot()
    def test_13_pca_eigenvalues_plot(self):
        arr = pca_eigenvalues_plot(
            self.pca_combined, num=12,
            plot_file=test_data_path + '/plots/combined_tmr_pca_ev.pdf'
        )
        self.assertEqual(len(arr[0]), 12)
        self.assertEqual(len(arr[1]), 12)
        plt.close()
        del arr

    # -- tica_eigenvalues_plot()
    def test_14_tica_eigenvalues_plot(self):
        arr_1, arr_2 = tica_eigenvalues_plot(
            self.tica_bbt_a, num=12,
            plot_file=test_data_path + '/plots/combined_tmr_tica_bbt_a_ev.pdf'
        )
        self.assertEqual(len(arr_1), 12)
        self.assertEqual(len(arr_2), 12)

    # -- pca_features()
    def test_15_pca_features(self):
        self.assertEqual(len(self.graph), 3)
        plt.close()
        # -- Graph
        for i in range(len(self.graph)):
            self.assertEqual(len(self.graph[i]), 448)
        # -- Corr
        self.assertEqual(len(self.corr), 81)

    # -- tica_features()
    def test_16_tica_features(self):
        test_feature = tica_feature_correlation(
            self.sim_a_tmr_feat['bb-torsions'], self.sim_a_tmr_data['bb-torsions'], 
            tica=self.tica_bbt_a, num=3, threshold=0.4
        )
        self.assertEqual(len(test_feature), 448)

    # -- sort_trajs_along_common_pc() + sort_traj_along_pc() + project_on_pc()
    def test_17_sort_trajs_along_pc(self):
        sort_common_traj = sort_trajs_along_common_pc(
            self.sim_a_tmr_data['bb-torsions'],
            self.sim_b_tmr_data['bb-torsions'],
            test_data_path + "/traj/condition-a_receptor.gro",
            test_data_path + "/traj/condition-b_receptor.gro",
            test_data_path + "/traj/condition-a_receptor.xtc",
            test_data_path + "/traj/condition-b_receptor.xtc",
            test_data_path + "/pca/receptor_by_tmr",
            num_pc=3, start_frame=self.start_frame
        )
        plt.close()
        for ele in sort_common_traj:
            self.assertEqual(len(ele), 3)
        self.assertEqual(len(self.all_sort), 3)

    # -- sort_trajs_along_common_tic()
    def test_18_sort_trajs_along_common_tic(self):
        sproj, sidx_data, sidx_traj = sort_trajs_along_common_tic(
            self.sim_a_tmr_data['bb-torsions'],
            self.sim_b_tmr_data['bb-torsions'],
            test_data_path + "/traj/condition-a_receptor.gro",
            test_data_path + "/traj/condition-b_receptor.gro",
            test_data_path + "/traj/condition-a_receptor.xtc",
            test_data_path + "/traj/condition-b_receptor.xtc",
            test_data_path + "/tica/receptor_by_tmr",
            num_ic=3
        )
        self.assertEqual(len(sproj[0]), 40)
        self.assertEqual(len(sidx_data[0]), 40)

    # -- sort_traj_along_tic()
    def test_19_sort_traj_along_tic(self):
        all_sort, _, _ = sort_traj_along_tic(
            self.sim_a_tmr_data['bb-torsions'],
            test_data_path + "/traj/condition-a_receptor.gro",
            test_data_path + "/traj/condition-a_receptor.xtc",
            test_data_path + "/pca/condition-a_receptor_by_tmr",
            tica=self.tica_bbt_a,
            num_ic=3
        )
        self.assertEqual(len(all_sort), 3)

    # -- compare_projections()
    def test_20_compare_projections(self):
        self.assertEqual(len(self.val), 3)
        self.assertEqual(len(self.val[0]), 2)
        self.assertEqual(len(self.val[1]), 2)
        self.assertEqual(len(self.val[2]), 2)
        self.assertEqual(len(self.val[0][0]), 20)
        self.assertEqual(len(self.val[0][1]), 20)
        self.assertEqual(len(self.val[1][0]), 20)
        self.assertEqual(len(self.val[1][1]), 20)
        self.assertEqual(len(self.val[2][0]), 20)
        self.assertEqual(len(self.val[2][1]), 20)

    # ** CLUSTERING **

    # -- obtain_combined_clusters()
    def test_21_obtain_combined_clusters(self):
        self.assertEqual(len(self.cidx), 40)
        self.assertEqual(len(self.cond), 40)
        test_oidx = [
#            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29
        ]
        self.assertEqual(len(self.centroids), 3)
        self.assertEqual(len(self.centroids[0]), 448)
        self.assertEqual(len(self.centroids[1]), 448)
        self.assertEqual(len(self.centroids[2]), 448)
        for i in range(len(self.oidx)):
            self.assertEqual(self.oidx[i], test_oidx[i])

    # -- write_cluster_traj()
    def test_22_write_cluster_traj(self):
        self.assertEqual(len(self.atom_group), 20)
        for i in range(len(self.atom_group)):
            self.assertEqual(self.atom_group[i].n_atoms, 1877)

    # --  wss_over_number_of_combined_clusters()
    def test_23_wss_over_number_of_combined_clusters(self):
        # -- wss over number of combined
        wss_combined_avg, wss_combined_std = wss_over_number_of_combined_clusters(
            self.sim_a_tmr_data['bb-torsions'], self.sim_b_tmr_data['bb-torsions'],
            label_a='A', label_b='B', start_frame=0,
            algorithm='kmeans', max_num_clusters=12,
            max_iter=100, num_repeats=5,
            plot_file=None
        )
        plt.close()
        # -- wss over number of clusters
        wss_avg, wss_std = wss_over_number_of_clusters(
            self.sim_a_tmr_data['bb-torsions'],
            algorithm='kmeans', max_num_clusters=12,
            max_iter=100, num_repeats=5,
            plot_file=None
        )
        plt.close()
        self.assertEqual(len(wss_avg), 11)
        self.assertEqual(len(wss_std), 11)
        self.assertLess(wss_std[0], 1.0e-12)
        self.assertEqual(len(wss_combined_avg), 11)
        self.assertEqual(len(wss_combined_std), 11)
        self.assertLess(wss_combined_std[0], 1.0e-12)

    # -- obtain_clusters()
    def test_24_obtain_clusters(self):
        # -- obtain combined clusters
        _ci, _wss, _centroids = obtain_clusters(self.sim_a_tmr_data['bb-torsions'], num_clusters=5)
        plt.close()
        unq_ci = list(set(_ci))
        for i in range(len(unq_ci)):
            self.assertEqual(unq_ci[i], i)

        self.assertEqual(len(_centroids), 5)
        for i in range(len(_centroids)):
            self.assertEqual(len(_centroids[i]), 448)


# ** RUN THE TESTS ** #
unittest.main(argv=['ignored', '-v'], exit=False)
gc.collect()
