import unittest, gc, mdshare, pyemma, os, requests, scipy.spatial, scipy.spatial.distance, pytest, importlib, scipy.stats, sys, os
import scipy as sp
import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np

from pyemma.util.contexts import settings
from clusters import *
from comparison import *
from features import *
from pca import *
from preprocessing import *
from tica import *

py_file_location = r"C:\Users\A49139\Documents\GitHub\pensa\pensa"
sys.path.append(os.path.abspath(py_file_location))

# Testing some functions which have return values
class Test_pensa(unittest.TestCase):
  
  # -- Reuse variable and functions
  def setUp(self):
    self.maxDiff = None
    start_frame = 0

    # -- Rec
    sim_a_rec = get_features("test_data/traj/condition-a_receptor.gro", 
                              "test_data/traj/condition-a_receptor.xtc", 
                              start_frame)
    sim_b_rec = get_features("test_data/traj/condition-b_receptor.gro",
                              "test_data/traj/condition-b_receptor.xtc", 
                              start_frame)
    self.sim_a_rec_feat, self.sim_a_rec_data = sim_a_rec
    self.sim_b_rec_feat, self.sim_b_rec_data = sim_b_rec

    # -- TMR 
    sim_a_tmr = get_features("test_data/traj/condition-a_tm.gro", 
                              "test_data/traj/condition-a_tm.xtc", 
                              start_frame)
    sim_b_tmr = get_features("test_data/traj/condition-b_tm.gro", 
                                "test_data/traj/condition-b_tm.xtc", 
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

    # -- obtain combined clusters
    self.cidx, self.cond, self.oidx, self.wss, self.centroids = obtain_combined_clusters(self.sim_a_tmr_data['bb-torsions'],self.sim_b_tmr_data['bb-torsions'],
                                                                                        label_a='A', label_b='B', start_frame=0,
                                                                                        algorithm='kmeans', max_iter=100, num_clusters=3, min_dist=12,
                                                                                        saveas='test_data/plots/combined_clust_bbtors.pdf')
    
    # -- PCA features
    self.graph, self.corr = pca_features(self.pca_combined, self.sim_a_tmr_feat['bb-torsions'], 3, 0.4)
    plt.close()
    
    # -- Sort trajectory along common pc
    self.sort_common_traj = sort_trajs_along_common_pc(self.sim_a_tmr_data['bb-torsions'],
                                                      self.sim_b_tmr_data['bb-torsions'],
                                                      start_frame,
                                                      "test_data/traj/condition-a_receptor.gro",
                                                      "test_data/traj/condition-b_receptor.gro",
                                                      "test_data/traj/condition-a_receptor.xtc",
                                                      "test_data/traj/condition-b_receptor.xtc",
                                                      "test_data/pca/receptor_by_tmr",
                                                      num_pc=3)
    plt.close()

    # -- Sort trajectory pc
    pca_a = calculate_pca(self.sim_a_tmr_data['bb-torsions'])
    pca_features(pca_a, self.sim_a_tmr_feat['bb-torsions'], 3, 0.4)
    plt.close()
    self.sort_traj, self.all_proj = sort_traj_along_pc(self.sim_a_tmr_data['bb-torsions'], pca_a, 0, 
                                                      "test_data/traj/condition-a_receptor.gro", 
                                                      "test_data/traj/condition-a_receptor.xtc", 
                                                      "test_data/pca/condition-a_receptor_by_tmr", num_pc=3)
    
    # -- Compare projections
    self.val = compare_projections(self.sim_a_tmr_data['bb-torsions'],
                                    self.sim_b_tmr_data['bb-torsions'],
                                    self.pca_combined,
                                    label_a='A', 
                                    label_b='B')
    plt.close()

    # -- Write cluster trajectory
    name = "condition-a_tm"
    self.atom_group = write_cluster_traj(self.cidx[self.cond==0], "test_data/traj/"+name+".gro","test_data/traj/"+name+".xtc", 
                                        "test_data/clusters/"+"combined_clust_bbtors_"+name, start_frame )

  # -- No clue
  def tearDown(self):
    pass
  
  # -- extract_coordinates() and extract_coordinates_combined()
  def test_extract_coordinates(self):
    root_dir_a = './test_data/MOR-apo'
    root_dir_b = './test_data/MOR-BU72'

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
                  root_dir_b+'/mor-bu72-2.xtc']
    # Base for the selection string for each simulation
    sel_base_a = "(not name H*) and protein"
    sel_base_b = "(not name H*) and protein"
    # Names of the output files
    out_name_a = "test_data/traj/condition-a"
    out_name_b = "test_data/traj/condition-b"
    out_name_combined="test_data/traj/combined"

    sel_string_a = load_selection("test_data/mor_tm.txt", sel_base_a+" and ")
    sel_string_b = load_selection("test_data/mor_tm.txt", sel_base_b+" and ")

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
                             'test_data/traj/combined_tm.xtc', 
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
    sel_string_a = load_selection("test_data/mor_tm.txt", sel_base_a+" and ")
    sel_string_b = load_selection("test_data/mor_tm.txt", sel_base_b+" and ")

    self.assertEqual(sel_string_a, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')
    self.assertEqual(sel_string_b, '(not name H*) and protein and resid 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 ')

  # -- get_features()
  def test_get_feature(self):
    self.assertEqual(self.sim_a_rec_data['bb-torsions'].shape, (30, 1148))
    self.assertEqual(self.sim_a_rec_data['sc-torsions'].shape, (30, 1054))
    self.assertEqual(self.sim_a_rec_data['bb-distances'].shape, (30, 41328))
        
    self.assertEqual(self.sim_b_rec_data['bb-torsions'].shape, (30, 1148))
    self.assertEqual(self.sim_b_rec_data['sc-torsions'].shape, (30, 1054))
    self.assertEqual(self.sim_b_rec_data['bb-distances'].shape, (30, 41328))
  
  # -- relative_entropy_analysis()
  def test_relative_entropy_analysis(self):    
    self.assertEqual(len(self.relen_tor[1]), 1148)
    self.assertEqual(len(self.relen_tor[0]), 1148)
    self.assertEqual(self.relen_tor[0][0], 'COS(PHI 0 VAL 66)')
    self.assertEqual(self.relen_tor[0][1], 'SIN(PHI 0 VAL 66)')
    self.assertEqual(self.relen_tor[0][2], 'COS(PSI 0 MET 65)')

  # -- sort_features()
  def test_sort_features(self):
    sf = sort_features(self.names_bbtors, self.jsd_bbtors) 
    test = ['COS(PSI 0 MET 99)','COS(PSI 0 THR 97)', 'SIN(PSI 0 THR 97)', 'COS(PSI 0 TYR 96)',
            'SIN(PSI 0 TYR 96)', 'COS(PSI 0 LYS 100)', 'SIN(PSI 0 MET 99)', 'COS(PHI 0 ASN 104)',
            'SIN(PSI 0 ASN 104)', 'COS(PSI 0 ASN 104)', 'SIN(PHI 0 THR 101)', 'COS(PSI 0 LEU 110)']
    sub_test = [1.0, 1.0, 1.0, 1.0,
                0.9230067758371652, 0.778412814913686, 0.7777156193332305, 0.7602417430261599,
                0.7565262841288898, 0.7512022647055374, 0.7417138509912806, 0.740226274002333]
    for i in range(12):
      self.assertEqual(sf[i][0], test[i])
      self.assertEqual(round(float(sf[i][1]), 2), round(sub_test[i], 2))

  # -- residue_visualization()
  def test_residue_visualization(self):
    ref_filename = "test_data/traj/condition-a_receptor.gro"
    out_filename = "receptor_bbtors-deviations_tremd"
    vis = residue_visualization(self.names_bbtors, self.jsd_bbtors, ref_filename, 
                                "test_data/plots/"+out_filename+"_jsd.pdf", 
                                "test_data/vispdb/"+out_filename+"_jsd.pdb",
                                y_label='max. JS dist. of BB torsions')
    
    self.assertEqual(len(vis), 2)
    self.assertEqual(len(vis[0]), 288)
    self.assertEqual(len(vis[1]), 288)
    self.assertEqual(vis[0][0], 65)
    self.assertEqual(vis[0][144], 209)
    self.assertEqual(vis[0][-1], 352)
    plt.close()
    del vis
  
  # -- distances_visualization()
  def test_distances_visualization(self):
    matrix = distances_visualization(self.names_bbdist, self.jsd_bbdist, 
                                      "test_data/plots/receptor_jsd-bbdist.pdf",
                                      vmin = 0.0, vmax = 1.0)
    self.assertEqual(len(matrix), 288)
    for i in range(len(matrix)):
      self.assertEqual(len(matrix[i]), 288)
      self.assertEqual(matrix[i][i], 0)
    plt.close()
    del matrix
  
  # -- calculate_pca()
  def test_calculate_pca(self):
    self.assertEqual(len(self.pca_combined.mean), 920)
    self.assertEqual(self.pca_combined.dim, -1)
    self.assertEqual(self.pca_combined.skip, 0)
    self.assertEqual(round(self.pca_combined.mean[-1],2), -0.23)
    self.assertEqual(self.pca_combined.stride, 1)
    self.assertEqual(self.pca_combined.var_cutoff, 0.95)
  
  # -- pca_eigenvalues_plot()
  def test_pca_eigenvalues_plot(self):
    test_x = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    test_y = [3.31234071, 2.53478601, 1.41273752, 1.01946236, 0.84863457,
              0.66441008, 0.58999452, 0.49132666, 0.46619091, 0.42758194,
              0.41925754, 0.38218951]
    arr = pca_eigenvalues_plot(self.pca_combined, num=12, plot_file='test_data/plots/combined_tmr_eigenvalues.pdf')
    for i in range(12):
      self.assertEqual(arr[0][i], test_x[i])
      self.assertEqual(round(arr[1][i], 2), round(test_y[i], 2)) 
    plt.close()
    del arr

  # -- obtain_combined_clusters()
  def test_obtain_combined_clusters(self):
    self.assertEqual(len(self.cidx), 60)
    self.assertEqual(len(self.cond), 60)
    test_oidx = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  0,  1,  2,  3,
                  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29]
    self.assertEqual(len(self.centroids), 3)
    self.assertEqual(len(self.centroids[0]), 920)
    self.assertEqual(len(self.centroids[1]), 920)
    self.assertEqual(len(self.centroids[2]), 920)
    for i in range(len(self.oidx)):
      self.assertEqual(self.oidx[i], test_oidx[i])
  
  #-- pca_features()
  def test_pca_features(self):
    self.assertEqual(len(self.graph), 3)
    plt.close()

    # -- Graph
    graph_val = [0.24,-0.04, 0.24]
    for i in range(len(self.graph)):
      self.assertEqual(len(self.graph[i]), 920)
      self.assertEqual(round(self.graph[i][0],2), graph_val[i])

    # -- Corr
    self.assertEqual(len(self.corr), 148)
    self.assertEqual(round(self.corr[0], 2), -0.42)
    self.assertEqual(round(self.corr[-1],2), 0.45)
  
  # -- sort_trajs_along_common_pc() + sort_traj_along_pc() + project_on_pc()
  def test_sort_trajs_along_pc(self):
    self.assertEqual(len(self.sort_common_traj), 180)
    for ele in self.sort_common_traj:
      self.assertEqual(ele.n_atoms, 2322)

    self.assertEqual(len(self.sort_traj), 90)
    for ele in self.sort_traj:
      self.assertEqual(ele.n_atoms, 2322)
    
    self.assertEqual(len(self.all_proj), 3)
    self.assertEqual(round(self.all_proj[0][0], 2), -0.26)
    self.assertEqual(round(self.all_proj[1][0], 2), 2.99)
    self.assertEqual(round(self.all_proj[2][0], 2), 1.51)

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
      self.assertEqual(len(_centroids[i]), 920)
        
# Run the test file
unittest.main(argv=['ignored', '-v'], exit=False)
gc.collect()