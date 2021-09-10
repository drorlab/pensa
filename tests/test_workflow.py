import unittest, gc, mdshare, pyemma, os, requests, scipy.spatial, scipy.spatial.distance, pytest, importlib, scipy.stats, sys, os
import scipy as sp
import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np

from pyemma.util.contexts import settings
from pensa.clusters import *
from pensa.comparison import *
from pensa.features import *
from pensa.dimensionality import *
from pensa.preprocessing import *


#py_file_location = '/Users/sangtruong_2021/Documents/GitHub/pensa/./test_data/'
#sys.path.append(os.path.abspath(py_file_location))

test_data_path = './tests/test_data'
for subdir in ['traj','plots','vispdb','pca','tica','clusters','results']:
    if not os.path.exists(test_data_path+'/'+subdir):
        os.makedirs(test_data_path+'/'+subdir)

# -- Testing some functions which have return values
class Test_pensa(unittest.TestCase):

  # -- Reuse variable and functions
  
  def setUp(self):
    print("done set up 1 ======================================")
    start_frame = 0
    # -- Rec
    sim_a_rec = get_structure_features(test_data_path + "/traj/condition-a_receptor.gro",
                                       test_data_path + "/traj/condition-a_receptor.xtc",
                                       start_frame)
    sim_b_rec = get_structure_features(test_data_path + "/traj/condition-b_receptor.gro",
                                       test_data_path + "/traj/condition-b_receptor.xtc",
                                       start_frame)
    self.sim_a_rec_feat, self.sim_a_rec_data = sim_a_rec
    self.sim_b_rec_feat, self.sim_b_rec_data = sim_b_rec

    # -- TMR
    sim_a_tmr = get_structure_features(test_data_path + "/traj/condition-a_tm.gro",
                                       test_data_path + "/traj/condition-a_tm.xtc",
                                       start_frame)
    sim_b_tmr = get_structure_features(test_data_path + "/traj/condition-b_tm.gro",
                                       test_data_path + "/traj/condition-b_tm.xtc",
                                       start_frame)
    self.sim_a_tmr_feat, self.sim_a_tmr_data = sim_a_tmr
    self.sim_b_tmr_feat, self.sim_b_tmr_data = sim_b_tmr

    # -- Torsion
    self.relen_tor = relative_entropy_analysis(self.sim_a_rec_feat['bb-torsions'],
                                               self.sim_b_rec_feat['bb-torsions'],
                                               self.sim_a_rec_data['bb-torsions'],
                                               self.sim_b_rec_data['bb-torsions'],
                                               bin_num=10, verbose=False)
    self.names_bbtors, self.jsd_bbtors, self.kld_ab_bbtors, kld_ba_bbtors = self.relen_tor

    # -- Distance
    self.relen_dist = relative_entropy_analysis(self.sim_a_rec_feat['bb-distances'],
                                                self.sim_b_rec_feat['bb-distances'],
                                                self.sim_a_rec_data['bb-distances'],
                                                self.sim_b_rec_data['bb-distances'],
                                                bin_num=10, verbose=False)
    self.names_bbdist, self.jsd_bbdist, self.kld_ab_bbdist, self.kld_ba_bbdist = self.relen_dist

    combined_data_tors = np.concatenate([self.sim_a_tmr_data['bb-torsions'],self.sim_b_tmr_data['bb-torsions']],0)
    self.pca_combined = calculate_pca(combined_data_tors)
    self.tica_combined = calculate_tica(combined_data_tors)
    
    # -- obtain combined clusters
    self.cidx, self.cond, self.oidx, self.wss, self.centroids = obtain_combined_clusters(self.sim_a_tmr_data['bb-torsions'],self.sim_b_tmr_data['bb-torsions'],
                                                                                        label_a='A', label_b='B', start_frame=0,
                                                                                        algorithm='kmeans', max_iter=100, num_clusters=3, min_dist=12,
                                                                                        saveas=test_data_path + '/plots/combined_clust_bbtors.pdf')

    # -- PCA features
    self.graph, self.corr = pca_features(self.pca_combined, self.sim_a_tmr_feat['bb-torsions'], 3, 0.4)
    plt.close()

    # -- Sort trajectory along common pc
    self.sort_common_traj = sort_trajs_along_common_pc(self.sim_a_tmr_data['bb-torsions'],
                                                      self.sim_b_tmr_data['bb-torsions'],
                                                      test_data_path + "/traj/condition-a_receptor.gro",
                                                      test_data_path + "/traj/condition-b_receptor.gro",
                                                      test_data_path + "/traj/condition-a_receptor.xtc",
                                                      test_data_path + "/traj/condition-b_receptor.xtc",
                                                      test_data_path + "/pca/receptor_by_tmr",
                                                      num_pc=3, start_frame = start_frame)
    plt.close()

    # -- Sort trajectory pc
    pca_a = calculate_pca(self.sim_a_tmr_data['bb-torsions'])
    pca_features(pca_a, self.sim_a_tmr_feat['bb-torsions'], 3, 0.4)
    plt.close()
    self.all_sort, _, _ = sort_traj_along_pc(self.sim_a_tmr_data['bb-torsions'],
                                             test_data_path + "/traj/condition-a_receptor.gro",
                                             test_data_path + "/traj/condition-a_receptor.xtc",
                                             test_data_path + "/pca/condition-a_receptor_by_tmr", 
                                             pca=pca_a, num_pc=3)

    # -- Compare projections
    self.val = compare_projections(self.sim_a_tmr_data['bb-torsions'],
                                    self.sim_b_tmr_data['bb-torsions'],
                                    self.pca_combined,
                                    label_a='A',
                                    label_b='B')
    plt.close()

    # -- Write cluster trajectory
    name = "condition-a_tm"
    self.atom_group = write_cluster_traj(self.cidx[self.cond==0], test_data_path + "/traj/"+name+".gro",test_data_path + "/traj/"+name+".xtc",
                                        test_data_path + "/clusters/"+"combined_clust_bbtors_"+name, start_frame )


  # -- extract_coordinates() and extract_coordinates_combined()
  def test_extract_coordinates(self):
    root_dir_a = test_data_path+'/MOR-apo'
    root_dir_b = test_data_path+'/MOR-BU72'

    # Simulation A
    ref_file_a =  root_dir_a+'/mor-apo.psf'
    pdb_file_a =  root_dir_a+'/mor-apo.pdb'
    trj_file_a = [root_dir_a+'/mor-apo-1.xtc',
                  root_dir_a+'/mor-apo-2.xtc',
                  root_dir_a+'/mor-apo-3.xtc']
    # Simulation B
    ref_file_b =  root_dir_b+'/mor-bu72.psf'
    pdb_file_b =  root_dir_b+'/mor-bu72.pdb'
    trj_file_b = [root_dir_b+'/mor-bu72-1.xtc',
                  root_dir_b+'/mor-bu72-2.xtc',
                  root_dir_b+'/mor-bu72-3.xtc']
    # Base for the selection string for each simulation
    sel_base_a = "(not name H*) and protein"
    sel_base_b = "(not name H*) and protein"
    # Names of the output files
    out_name_a = test_data_path + "/traj/condition-a"
    out_name_b = test_data_path + "/traj/condition-b"
    out_name_combined=test_data_path + "/traj/combined"

    sel_string_a = load_selection(test_data_path + "/mor_tm.txt", sel_base_a+" and ")
    sel_string_b = load_selection(test_data_path + "/mor_tm.txt", sel_base_b+" and ")

    file_a = extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
    file_b = extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)

    # First test case
    # Number of Atom from selection
    self.assertEqual(file_a, 2322)
    self.assertEqual(file_b, 2322)

    # Second test case
    # Extract the coordinates of the transmembrane region from the trajectory
    file_a = extract_coordinates(ref_file_a, pdb_file_a, [trj_file_a], out_name_a+"_tm", sel_string_a)
    file_b = extract_coordinates(ref_file_b, pdb_file_b, [trj_file_b], out_name_b+"_tm", sel_string_b)
    file_combine = extract_coordinates_combined([ref_file_a]*3 + [ref_file_b]*3,
                             trj_file_a + trj_file_b,
                             [sel_string_a]*3 + [sel_string_b]*3,
                             test_data_path + '/traj/combined_tm.xtc',
                             start_frame=0)

    # Number of Atom from the selction
    self.assertEqual(file_a, 1877)
    self.assertEqual(file_b, 1877)
    # Number of Atom from the selection - combined
    self.assertEqual(file_a, 1877)

  # -- load_selection()
  def test_load_selection(self):
    sel_base_a = "(not name H*) and protein"
    sel_base_b = "(not name H*) and protein"
    sel_string_a = load_selection(test_data_path + "/mor_tm.txt", sel_base_a+" and ")
    sel_string_b = load_selection(test_data_path + "/mor_tm.txt", sel_base_b+" and ")

    self.assertEqual(sel_string_a, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')
    self.assertEqual(sel_string_b, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')

  # -- get_features()
  def test_get_feature(self):
    self.assertEqual(self.sim_a_rec_data['bb-torsions'].shape, (30, 574))
    self.assertEqual(self.sim_a_rec_data['sc-torsions'].shape, (30, 527))

    self.assertEqual(self.sim_b_rec_data['bb-torsions'].shape, (30, 574))
    self.assertEqual(self.sim_b_rec_data['sc-torsions'].shape, (30, 527))
    
  # -- relative_entropy_analysis()
  def test_relative_entropy_analysis(self):
    self.assertEqual(len(self.relen_tor[1]), 574)
    self.assertEqual(len(self.relen_tor[0]), 574)

  # -- sort_features()
  def test_sort_features(self):
    sf = sort_features(self.names_bbtors, self.jsd_bbtors)
    self.assertEqual(len(sf), 574)

  # -- residue_visualization()
  def test_residue_visualization(self):
    ref_filename = test_data_path + "/traj/condition-a_receptor.gro"
    out_filename = "receptor_bbtors-deviations_tremd"
    vis = residue_visualization(self.names_bbtors, self.jsd_bbtors, ref_filename,
                                test_data_path + "/plots/"+out_filename+"_jsd.pdf",
                                test_data_path + "/vispdb/"+out_filename+"_jsd.pdb",
                                y_label='max. JS dist. of BB torsions')

    self.assertEqual(len(vis), 2)
    self.assertEqual(len(vis[0]), 288)
    self.assertEqual(len(vis[1]), 288)
    plt.close()
    del vis

  # -- distances_visualization()
  def test_distances_visualization(self):
    matrix = distances_visualization(self.names_bbdist, self.jsd_bbdist,
                                      test_data_path + "/plots/receptor_jsd-bbdist.pdf",
                                      vmin = 0.0, vmax = 1.0)
    self.assertEqual(len(matrix), 288)
    for i in range(len(matrix)):
      self.assertEqual(len(matrix[i]), 288)
      self.assertEqual(matrix[i][i], 0)
    plt.close()
    del matrix


  # ** DIMENSIONALITY **

  # -- calculate_pca()
  def test_calculate_pca(self):
    self.assertEqual(len(self.pca_combined.mean), 460)
    self.assertEqual(self.pca_combined.dim, -1)
    self.assertEqual(self.pca_combined.skip, 0)

    self.assertEqual(self.pca_combined.stride, 1)
    self.assertEqual(self.pca_combined.var_cutoff, 0.95)

  # -- calculate_tica
  def test_tica_combined(self):
    self.assertEqual(self.tica_combined.lag, 10)
    self.assertEqual(self.tica_combined.kinetic_map, True)


  # -- pca_eigenvalues_plot()
  def test_pca_eigenvalues_plot(self):
    arr = pca_eigenvalues_plot(self.pca_combined, num=12, plot_file=test_data_path+'/plots/combined_tmr_eigenvalues.pdf')
    self.assertEqual(len(arr[0]), 12)
    self.assertEqual(len(arr[1]), 12)
    plt.close()
    del arr

  # -- tica_eigenvalues_plot()
  def test_tica_eigenvalues_plot(self):
    arr_1, arr_2 = tica_eigenvalues_plot(self.tica_combined, num=12, plot_file=test_data_path+'/plots/combined_tmr_eigenvalues.pdf')
    self.assertEqual(len(arr_1), 12)
    self.assertEqual(len(arr_2), 12)


  #-- pca_features()
  def test_pca_features(self):
    self.assertEqual(len(self.graph), 3)
    plt.close()

    # -- Graph
    for i in range(len(self.graph)):
      self.assertEqual(len(self.graph[i]), 460)

    # -- Corr
    self.assertEqual(len(self.corr), 51)

  # -- tica_features()
  def test_tica_features(self):
    test_feature = tica_features(self.tica_combined,self.sim_a_tmr_feat['bb-torsions'], 3, 0.4)
    self.assertEqual(len(test_feature), 460)


  # -- sort_trajs_along_common_pc() + sort_traj_along_pc() + project_on_pc()
  def test_sort_trajs_along_pc(self):
    for ele in self.sort_common_traj:
      self.assertEqual(len(ele), 3)

    self.assertEqual(len(self.all_sort), 3)

  # -- sort_trajs_along_common_tic()
  def test_sort_trajs_along_common_tic(self):
    sproj, sidx_data, sidx_traj = sort_trajs_along_common_tic(self.sim_a_tmr_data['bb-torsions'],
                            self.sim_b_tmr_data['bb-torsions'],
                            test_data_path + "/traj/condition-a_receptor.gro",
                            test_data_path + "/traj/condition-b_receptor.gro",
                            test_data_path + "/traj/condition-a_receptor.xtc",
                            test_data_path + "/traj/condition-b_receptor.xtc",
                            test_data_path + "/tica/receptor_by_tmr",
                            num_ic=3)
    self.assertEqual(len(sproj[0]), 60)
    self.assertEqual(len(sidx_data[0]), 60)

  # -- sort_traj_along_tic()
  def test_sort_traj_along_tic(self):
    all_sort, _, _ = sort_traj_along_tic(self.sim_a_tmr_data['bb-torsions'], 
                                         test_data_path + "/traj/condition-a_receptor.gro",
                                         test_data_path + "/traj/condition-a_receptor.xtc",
                                         test_data_path + "/pca/condition-a_receptor_by_tmr", 
                                         tica = self.tica_combined, num_ic=3)
    self.assertEqual(len(all_sort), 3)


  # -- compare_projections()
  def test_compare_projections(self):
    self.assertEqual(len(self.val), 3)
    self.assertEqual(len(self.val[0]), 2)
    self.assertEqual(len(self.val[1]), 2)
    self.assertEqual(len(self.val[2]), 2)

    self.assertEqual(len(self.val[0][0]), 30)
    self.assertEqual(len(self.val[0][1]), 30)
    self.assertEqual(len(self.val[1][0]), 30)
    self.assertEqual(len(self.val[1][1]), 30)
    self.assertEqual(len(self.val[2][0]), 30)
    self.assertEqual(len(self.val[2][1]), 30)



  # ** CLUSTERING **

  # -- obtain_combined_clusters()
  def test_obtain_combined_clusters(self):
    self.assertEqual(len(self.cidx), 60)
    self.assertEqual(len(self.cond), 60)
    test_oidx = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  0,  1,  2,  3,
                  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29]
    self.assertEqual(len(self.centroids), 3)
    self.assertEqual(len(self.centroids[0]), 460)
    self.assertEqual(len(self.centroids[1]), 460)
    self.assertEqual(len(self.centroids[2]), 460)
    for i in range(len(self.oidx)):
      self.assertEqual(self.oidx[i], test_oidx[i])

  # -- write_cluster_traj()
  def test_write_cluster_traj(self):
    self.assertEqual(len(self.atom_group), 30)
    for i in range(len(self.atom_group)):
      self.assertEqual(self.atom_group[i].n_atoms, 1877)

  # --  wss_over_number_of_combined_clusters()
  def test_wss_over_number_of_combined_clusters(self):
    # -- wss over number of combined
    wss_combined_avg, wss_combined_std = wss_over_number_of_combined_clusters(self.sim_a_tmr_data['bb-torsions'],
                                                                              self.sim_b_tmr_data['bb-torsions'],
                                                                              label_a='A', label_b='B',
                                                                              start_frame=0,
                                                                              algorithm='kmeans',
                                                                              max_iter=100, num_repeats = 5,
                                                                              max_num_clusters = 12,
                                                                              plot_file = None)
    plt.close()

    # -- wss over number of clusters
    wss_avg, wss_std = wss_over_number_of_clusters(self.sim_a_tmr_data['bb-torsions'],
                                                    algorithm='kmeans',
                                                    max_iter=100, num_repeats = 5,
                                                    max_num_clusters = 12,
                                                    plot_file = None)
    plt.close()
    self.assertEqual(len(wss_avg), 11)
    self.assertEqual(len(wss_std), 11)
    self.assertEqual(wss_std[0], 0)
    self.assertEqual(len(wss_combined_avg), 11)
    self.assertEqual(len(wss_combined_std), 11)
    self.assertEqual(wss_combined_std[0], 0)

  # -- obtain_clusters()
  def test_obtain_clusters(self):
    # -- obtain combined clusters
    _ci, _wss, _centroids = obtain_clusters(self.sim_a_tmr_data['bb-torsions'], num_clusters=5 )
    plt.close()
    unq_ci = list(set(_ci))
    for i in range(len(unq_ci)):
      self.assertEqual(unq_ci[i], i)

    self.assertEqual(len(_centroids), 5)
    for i in range(len(_centroids)):
      self.assertEqual(len(_centroids[i]), 460)



# -- Extract coordinate

root_dir_a = test_data_path + '/MOR-apo'
root_dir_b = test_data_path + '/MOR-BU72'

# Simulation A
ref_file_a =  root_dir_a+'/mor-apo.psf'
pdb_file_a =  root_dir_a+'/mor-apo.pdb'
trj_file_a = [root_dir_a+'/mor-apo-1.xtc',
			  root_dir_a+'/mor-apo-2.xtc',
			  root_dir_a+'/mor-apo-3.xtc']
# Simulation B
ref_file_b =  root_dir_b+'/mor-bu72.psf'
pdb_file_b =  root_dir_b+'/mor-bu72.pdb'
trj_file_b = [root_dir_b+'/mor-bu72-1.xtc',
			  root_dir_b+'/mor-bu72-2.xtc',
			  root_dir_b+'/mor-bu72-3.xtc']
# Base for the selection string for each simulation
sel_base_a = "(not name H*) and protein"
sel_base_b = "(not name H*) and protein"
# Names of the output files
out_name_a = test_data_path + "/traj/condition-a"
out_name_b = test_data_path + "/traj/condition-b"
out_name_combined=test_data_path + "/traj/combined"


sel_string_a = load_selection(test_data_path + "/mor_tm.txt", sel_base_a+" and ")
sel_string_b = load_selection(test_data_path + "/mor_tm.txt", sel_base_b+" and ")

file_a = extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
file_b = extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)

file_a = extract_coordinates(ref_file_a, pdb_file_a, [trj_file_a], out_name_a+"_tm", sel_string_a)
file_b = extract_coordinates(ref_file_b, pdb_file_b, [trj_file_b], out_name_b+"_tm", sel_string_b)
file_combine = extract_coordinates_combined([ref_file_a]*3 + [ref_file_b]*3,
											 trj_file_a + trj_file_b,
											 [sel_string_a]*3 + [sel_string_b]*3,
											 test_data_path + '/traj/combined_tm.xtc',
											 start_frame=0)
# Run the test file
unittest.main(argv=['ignored', '-v'], exit=False)
#unittest.main()

gc.collect()
