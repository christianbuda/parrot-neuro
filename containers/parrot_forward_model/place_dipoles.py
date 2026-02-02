from mesh_poisson_disk_sampling import poisson_disk_vertex_sampling, uniform_vertex_sampling
import pygeodesic.geodesic as geodesic 
import igl
import scipy
import sklearn
import trimesh
import nibabel as nib
import numpy as np
import os
import glob
import pathlib
import argparse

# estimated number of surface dipoles
# HIGH RES: 2 mm spacing, 40000 triplets, 120000 dipoles
# MID RES: 3 mm spacing, 17530 triplets, 53000 dipoles
# LOW RES: 4 mm spacing, 9700 triplets, 29000 dipoles

# dipoles per surface
# freesurfer_lh_middle # 13300 -> 6000 -> 3300
# freesurfer_rh_middle # 13300 -> 6000 -> 3300
# cereb_inner # 12000 -> 5300 -> 3000
# hippunfold_L_dentate_middle # 50 -> 25 -> 20
# hippunfold_R_dentate_middle # 50 -> 25 -> 20
# hippunfold_L_hipp_middle # 200 -> 90 -> 50
# hippunfold_R_hipp_middle # 200 -> 90 -> 50

def str_to_label(names):
    with open(add_subject_dir('atlas/atlas100_labels.txt'), 'r') as f:
        label_dict = f.readlines()
    label_dict = dict(map(lambda x: tuple(x.strip().split(',')[-1::-1]), label_dict))
    labels = []
    for name in names:
        labels.append(int(label_dict[name]))
    return labels

def add_subject_dir(*paths):
    if len(paths)==1:
        return os.path.join(subject_dir, paths[0])
    return tuple([add_subject_dir(x) for x in paths])

def load_npy(paths):
    if isinstance(paths, list):
        return [load_npy(x) for x in paths]
    return np.load(paths)

def save_npy(filename, arr):
    path = os.path.dirname(filename)
    os.makedirs(path, exist_ok=True)
    np.save(filename, arr)
    return

def compute_face_normals(vertices, faces, return_area = False):
    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]
    
    
    if not return_area:
        return np.cross(B-A, C-A)/np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
    else:
        norms = np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
        return np.cross(B-A, C-A)/norms, norms[:,0]/2
    
def compute_face_centers(vertices, faces):
    return vertices[faces].mean(axis = 1)

def compute_vertex_normals(vertices, faces, normalized = True):
    face_normals, face_areas = compute_face_normals(vertices, faces, return_area = True)

    face_weights = trimesh.Trimesh(vertices=vertices, faces=faces).faces_sparse.multiply(face_areas)

    vertex_normals = np.array(np.concatenate([face_weights.multiply(face_normals[:,0]).sum(axis = 1),
    face_weights.multiply(face_normals[:,1]).sum(axis = 1),
    face_weights.multiply(face_normals[:,2]).sum(axis = 1)], axis = 1))

    if normalized:
        return vertex_normals/np.linalg.norm(vertex_normals, axis = 1, keepdims = True)
    else:
        return vertex_normals

def find_best_label(labels, best_idx):
    # find most frequent label associated to each dipole
    
    row_data_pairs = np.stack([best_idx, labels], axis = 1)

    # Find unique (row, value) pairs and their counts
    unique_pairs, pair_counts = np.unique(row_data_pairs, axis=0, return_counts=True)
    
    unique_rows = unique_pairs[:, 0].astype(int)
    unique_labels = unique_pairs[:, 1]
    
    # We want to sort such that the "winner" for each row is at the top (first index).
    # Keys for lexsort (last is primary):
    # 1. Row ID (Ascending) -> Group by row
    # 2. Count (Descending) -> Highest frequency first
    # 3. Value (Ascending)  -> Tie-breaker: prefer smaller value
    sort_order = np.lexsort((unique_labels, -pair_counts, unique_rows))
    
    unique_rows = unique_rows[sort_order]
    unique_labels = unique_labels[sort_order]
    
    # unique with return_index gives the FIRST index of each unique element.
    # Since we sorted by Row, the first index for each row is our "winner" (max count).
    _, max_idx = np.unique(unique_rows, return_index=True)
    
    best_labels = unique_labels[max_idx]
    
    return(best_labels)

def find_closest_euclidean_source(mesh_vertices, source_indices):
        """
        Returns the closest source for every vertex based on straight-line 3D distance.
        """
        # 1. Extract the coordinates of the source vertices
        source_coords = mesh_vertices[source_indices]
        
        # 2. Build a KD-Tree of the SOURCE points
        tree = scipy.spatial.KDTree(source_coords)
        
        # 3. Query the tree with ALL mesh vertices
        # k=1 returns the single closest neighbor
        distances, positions_in_source_list = tree.query(mesh_vertices, k=1)
        
        return positions_in_source_list

def sample_surface(mesh, min_dist, vertex_label, vertex_thickness = None, vertex_volume = None, generator = None, verbose = True, test_mode = False):
    # input is a trimesh object and the minimum distance between sampled vertices (in mm)
    
    if generator is None:
        generator = np.random.default_rng()
    
    if vertex_volume is None:
        assert vertex_thickness is not None, 'ERROR: you must either provide a volume or a thickness associated to each vertex'
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    if verbose:
        print(f'Sampling dipoles for surface with {len(vertices)} vertices, with min_dist = {min_dist}{f', with precomputed vertex volumes.' if vertex_volume is not None else ''}')
    
    if verbose:
        print('Sampling dipoles...')
    
    if test_mode:
        # sample vertices uniformly at random over the entire mesh (very fast, useful for testing)
        _,_, sampled_vertices = uniform_vertex_sampling(vertices, faces, num_points = int(0.5*mesh.area/min_dist**2), generator = generator)
        best_idx = find_closest_euclidean_source(vertices, sampled_vertices)
    else:
        # sample vertices uniformly with minimum distance using Poisson disk strategy (better, but slower)
        _,_, sampled_vertices = poisson_disk_vertex_sampling(vertices, faces, min_dist = min_dist, generator = generator, verbose = verbose)
    
        if verbose:
            print('Computing geodesic distance between sampled dipoles and all other vertices.')
        
        # i retain this method to compute area of influence of each dipole because boundaries of regions are not well defined and regions are not very small
        # subcortical region follow a more sophisticated approach where each dipole can only have influence on the ones with the same label
        geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        _, best_idx = geoalg.geodesicDistances(source_indices=sampled_vertices, target_indices=np.arange(len(vertices)))
    
    # find best label for each dipole
    # deals with the possibility of having several labels per vertex
    if isinstance(vertex_label, list):
        dipole_labels = []
        for labels in vertex_label:
            dipole_labels.append(find_best_label(labels, best_idx))
    else:
        dipole_labels = find_best_label(vertex_label, best_idx)
    
    if vertex_volume is None:
        # Compute the Mass Matrix, which estimates the amount of area associated to each vertex in the mesh
        M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
        vertex_areas = M.diagonal()

        # estimate volume from thickness and area
        vertex_volume = vertex_areas*vertex_thickness
    
    dipole_volume = np.array(scipy.sparse.coo_matrix((vertex_volume, (np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), len(sampled_vertices))).sum(axis=0))[0]
    
    # compute normals
    all_normals = compute_vertex_normals(vertices, faces, normalized=False)
    dipole_normals = np.concatenate([scipy.sparse.coo_matrix((all_normals[:,0], (np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), len(sampled_vertices))).sum(axis=0), scipy.sparse.coo_matrix((all_normals[:,1],( np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), len(sampled_vertices))).sum(axis=0), scipy.sparse.coo_matrix((all_normals[:,2],( np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), len(sampled_vertices))).sum(axis=0)]).T
    dipole_normals = np.array(dipole_normals/np.linalg.norm(dipole_normals, axis = 1, keepdims = True))

    return sampled_vertices, vertices[sampled_vertices], dipole_labels, dipole_volume, dipole_normals

def sample_all_surfaces(all_meshes, dipole_spacing, generator = None, test_mode = False):
    surface_dipoles = []
    for mesh_dict in all_meshes:
        print(f'Sampling dipoles on mesh {mesh_dict["mesh"]}')
        sampled_vertices, dipole_positions, dipole_labels, dipole_volume, dipole_normals = sample_surface(mesh = trimesh.load_mesh(mesh_dict['mesh']), min_dist = dipole_spacing, vertex_label = load_npy(mesh_dict['labels']), vertex_thickness = load_npy(mesh_dict['thickness']), vertex_volume = (load_npy(mesh_dict['volume']) if 'volume' in mesh_dict.keys() else None), generator = generator, test_mode = test_mode)
        print('Done!\n')
        basename = pathlib.Path(mesh_dict['mesh']).stem
        output_dir = add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/surfaces/'+basename)

        output_dict = { 'sampled_vertices':os.path.join(output_dir, 'sampled_vertices.npy'),
                        'dipole_positions':os.path.join(output_dir, 'dipole_positions.npy'),
                        'dipole_volume':os.path.join(output_dir, 'dipole_volume.npy'),
                        'dipole_directions':os.path.join(output_dir, 'dipole_normals.npy')}

        save_npy(output_dict['sampled_vertices'], sampled_vertices)
        save_npy(output_dict['dipole_positions'], dipole_positions)
        save_npy(output_dict['dipole_volume'], dipole_volume)
        save_npy(output_dict['dipole_directions'], dipole_normals)
        
        if isinstance(dipole_labels, list):
            output_dict['dipole_labels'] = {}
            for idx,lab in enumerate(dipole_labels):
                atlas_name = os.path.basename(mesh_dict['labels'][idx])[3:-11]
                output_dict['dipole_labels'][atlas_name] = os.path.join(output_dir, atlas_name + '_dipole_labels'+'.npy')
                save_npy(output_dict['dipole_labels'][atlas_name], lab)
        else:
            output_dict['dipole_labels'] = os.path.join(output_dir, 'dipole_labels.npy')
            save_npy(output_dict['dipole_labels'], dipole_labels)
        
        surface_dipoles.append(output_dict)
    
    return surface_dipoles

def poisson_disk_subsampling(points, radius, generator = None):
    """
    Selects a subset of points in euclidean space such that no two points are closer than `radius`.
    """
    if generator is None:
        generator = np.random.default_rng()
    
    sampled_points = []
    all_idx = np.arange(len(points), dtype = int)
    selectable_mask = np.ones(len(points), dtype=bool)

    while np.count_nonzero(selectable_mask)>0:
        point = generator.choice(all_idx[selectable_mask])
        sampled_points.append(int(point))
        selectable_mask[np.linalg.norm(points-points[point], axis = 1)<radius] = False
    
    return sampled_points

def sample_spherical(npoints, generator = None):
    # sample npoints on a 3d sphere of unit radius
    if generator is None:
        generator = np.random.default_rng()
    vec = generator.standard_normal(size = 3*npoints).reshape((-1,3))
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec

def get_structure_size(positions):
    # returns an estimate of the size of a point cloud using PCA
    # it computes the span of the point cloud along the principal directions
    
    # handle edge cases
    if len(positions == 1):
        return np.inf
    if len(positions== 2):
        return np.linalg.norm(positions[0]-positions[1])
    
    def span(x, axis = None):
        return x.max(axis = axis)-x.min(axis=axis)
    pca = sklearn.decomposition.PCA().fit(positions)

    return span(pca.transform(positions), axis = 0).max()

def find_influence_area(dipoles, dipole_labels, voxels, voxel_labels):
    # finds closest dipole to each voxel grouping them by label
    # this implies that, if a label has no dipoles inside, the voxel is unassigned
    # in this case a value of -1 is placed
    dipole_indices = np.arange(len(dipole_labels))
    
    best_idx = -np.ones_like(voxel_labels, dtype = int)
    for label in np.unique(dipole_labels):
        which_dipoles = dipole_labels == label
        which_voxels = voxel_labels == label
        
        nn_lookup = scipy.spatial.KDTree(dipoles[which_dipoles])
        distances, indices = nn_lookup.query(voxels[which_voxels])
        
        indices = dipole_indices[which_dipoles][indices]
        
        best_idx[which_voxels] = indices
    
    return best_idx

def get_instruction_files(kind):
    if kind == 'volume':
        labels_to_exclude = []

        # hippocampus (surface based)
        labels_to_exclude += ['Left-subiculum', 'Left-CA1', 'Left-CA2', 'Left-CA3', 'Left-CA4', 'Left-dentate_gyrus', 'Right-subiculum', 'Right-CA1', 'Right-CA2', 'Right-CA3', 'Right-CA4', 'Right-dentate_gyrus', 'Right-SRLM', 'Right-cysts']
        # cerebellum (surface based)
        labels_to_exclude += ['cerebellum_L.GM.Lobule_I-II', 'cerebellum_L.GM.Lobule_III', 'cerebellum_L.GM.Lobule_IV', 'cerebellum_L.GM.Lobule_V', 'cerebellum_L.GM.Lobule_VI', 'cerebellum_L.GM.Crus_I', 'cerebellum_L.GM.Crus_II', 'cerebellum_L.GM.Lobule_VIIB', 'cerebellum_L.GM.Lobule_VIIIA', 'cerebellum_L.GM.Lobule_VIIIB', 'cerebellum_L.GM.Lobule_IX', 'cerebellum_L.GM.Lobule_X', 'cerebellum_R.GM.Lobule_I-II', 'cerebellum_R.GM.Lobule_III', 'cerebellum_R.GM.Lobule_IV', 'cerebellum_R.GM.Lobule_V', 'cerebellum_R.GM.Lobule_VI', 'cerebellum_R.GM.Crus_I', 'cerebellum_R.GM.Crus_II', 'cerebellum_R.GM.Lobule_VIIB', 'cerebellum_R.GM.Lobule_VIIIA', 'cerebellum_R.GM.Lobule_VIIIB', 'cerebellum_R.GM.Lobule_IX', 'cerebellum_R.GM.Lobule_X']
        # cortex, atlas100 (surface based)
        labels_to_exclude += ['17Networks_LH_VisCent_ExStr_1', '17Networks_LH_VisCent_ExStr_2', '17Networks_LH_VisCent_Striate_1', '17Networks_LH_VisCent_ExStr_3', '17Networks_LH_VisPeri_ExStrInf_1', '17Networks_LH_VisPeri_StriCal_1', '17Networks_LH_VisPeri_ExStrSup_1', '17Networks_LH_SomMotA_1', '17Networks_LH_SomMotA_2', '17Networks_LH_SomMotB_Aud_1', '17Networks_LH_SomMotB_S2_1', '17Networks_LH_SomMotB_S2_2', '17Networks_LH_SomMotB_Cent_1', '17Networks_LH_DorsAttnA_TempOcc_1', '17Networks_LH_DorsAttnA_ParOcc_1', '17Networks_LH_DorsAttnA_SPL_1', '17Networks_LH_DorsAttnB_PostC_1', '17Networks_LH_DorsAttnB_PostC_2', '17Networks_LH_DorsAttnB_PostC_3', '17Networks_LH_DorsAttnB_FEF_1', '17Networks_LH_SalVentAttnA_ParOper_1', '17Networks_LH_SalVentAttnA_Ins_1', '17Networks_LH_SalVentAttnA_Ins_2', '17Networks_LH_SalVentAttnA_ParMed_1', '17Networks_LH_SalVentAttnA_FrMed_1', '17Networks_LH_SalVentAttnB_PFCl_1', '17Networks_LH_SalVentAttnB_PFCmp_1', '17Networks_LH_LimbicB_OFC_1', '17Networks_LH_LimbicA_TempPole_1', '17Networks_LH_LimbicA_TempPole_2', '17Networks_LH_ContA_IPS_1', '17Networks_LH_ContA_PFCl_1', '17Networks_LH_ContA_PFCl_2', '17Networks_LH_ContB_PFClv_1', '17Networks_LH_ContC_pCun_1', '17Networks_LH_ContC_pCun_2', '17Networks_LH_ContC_Cingp_1', '17Networks_LH_DefaultA_PFCd_1', '17Networks_LH_DefaultA_pCunPCC_1', '17Networks_LH_DefaultA_PFCm_1', '17Networks_LH_DefaultB_Temp_1', '17Networks_LH_DefaultB_Temp_2', '17Networks_LH_DefaultB_IPL_1', '17Networks_LH_DefaultB_PFCd_1', '17Networks_LH_DefaultB_PFCl_1', '17Networks_LH_DefaultB_PFCv_1', '17Networks_LH_DefaultB_PFCv_2', '17Networks_LH_DefaultC_Rsp_1', '17Networks_LH_DefaultC_PHC_1', '17Networks_LH_TempPar_1', '17Networks_RH_VisCent_ExStr_1', '17Networks_RH_VisCent_ExStr_2', '17Networks_RH_VisCent_ExStr_3', '17Networks_RH_VisPeri_StriCal_1', '17Networks_RH_VisPeri_ExStrInf_1', '17Networks_RH_VisPeri_ExStrSup_1', '17Networks_RH_SomMotA_1', '17Networks_RH_SomMotA_2', '17Networks_RH_SomMotA_3', '17Networks_RH_SomMotA_4', '17Networks_RH_SomMotB_Aud_1', '17Networks_RH_SomMotB_S2_1', '17Networks_RH_SomMotB_S2_2', '17Networks_RH_SomMotB_Cent_1', '17Networks_RH_DorsAttnA_TempOcc_1', '17Networks_RH_DorsAttnA_ParOcc_1', '17Networks_RH_DorsAttnA_SPL_1', '17Networks_RH_DorsAttnB_PostC_1', '17Networks_RH_DorsAttnB_PostC_2', '17Networks_RH_DorsAttnB_FEF_1', '17Networks_RH_SalVentAttnA_ParOper_1', '17Networks_RH_SalVentAttnA_Ins_1', '17Networks_RH_SalVentAttnA_ParMed_1', '17Networks_RH_SalVentAttnA_FrMed_1', '17Networks_RH_SalVentAttnB_IPL_1', '17Networks_RH_SalVentAttnB_PFCl_1', '17Networks_RH_SalVentAttnB_PFCmp_1', '17Networks_RH_LimbicB_OFC_1', '17Networks_RH_LimbicA_TempPole_1', '17Networks_RH_ContA_IPS_1', '17Networks_RH_ContA_PFCl_1', '17Networks_RH_ContA_PFCl_2', '17Networks_RH_ContB_Temp_1', '17Networks_RH_ContB_IPL_1', '17Networks_RH_ContB_PFCld_1', '17Networks_RH_ContB_PFClv_1', '17Networks_RH_ContC_Cingp_1', '17Networks_RH_ContC_pCun_1', '17Networks_RH_DefaultA_IPL_1', '17Networks_RH_DefaultA_PFCd_1', '17Networks_RH_DefaultA_pCunPCC_1', '17Networks_RH_DefaultA_PFCm_1', '17Networks_RH_DefaultB_PFCd_1', '17Networks_RH_DefaultB_PFCv_1', '17Networks_RH_DefaultB_PFCv_2', '17Networks_RH_DefaultC_Rsp_1', '17Networks_RH_DefaultC_PHC_1', '17Networks_RH_TempPar_1', '17Networks_RH_TempPar_2', '17Networks_RH_TempPar_3']
        # brainstem
        labels_to_exclude += ['brainstem', 'DCG', 'Vermis', 'Midbrain', 'Pons', 'Medulla', 'Vermis-White-Matter', 'SCP', 'Floculus']
        # Corpus Callosum and other White Matter tracts
        labels_to_exclude += ['Fornix', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior', 'Ant-Commisure', 'R-Fornix', 'L-Fornix',]
        # Optic structures
        labels_to_exclude += ['R-N.opticus', 'L-N.opticus', 'R-Optic-tract', 'L-Optic-tract', 'R-Chiasma-Opticum', 'L-Chiasma-Opticum', 'Left-Lateral-nucleus-olfactory-tract', 'Right-Lateral-nucleus-olfactory-tract',]
        # Glandular structures
        labels_to_exclude += ['Epiphysis', 'Hypophysis']
        # Fluid cavities
        labels_to_exclude += ['Left-cysts', 'Right-cysts', 'Third-Ventricle', 'Infundibulum']
        # Transition zones
        labels_to_exclude += ['Left-SRLM', 'Right-SRLM', 'Left-Fusion-amygdala-HP-FAH', 'Right-Fusion-amygdala-HP-FAH', 'Left-Envelope-Amygdala', 'Right-Envelope-Amygdala', 'Left-Extranuclear-Amydala', 'Right-Extranuclear-Amydala']
        # VentralDC, Freesurfer catch-all label, mostly accounted for
        labels_to_exclude += ['Left-VentralDC', 'Right-VentralDC']

        # Use gradient as normal. Cortical Amygdala Group
        gradient_as_normal = [  {'labels_with_dipoles':['Left-Cortical-nucleus', 'Right-Cortical-nucleus', 'Left-Corticoamygdaloid-transitio', 'Right-Corticoamygdaloid-transitio', 'Left-Anterior-amygdaloid-area-AAA', 'Right-Anterior-amygdaloid-area-AAA', 'Left-Hippocampal-amygdala-transition-HATA', 'Right-Hippocampal-amygdala-transition-HATA', 'Left-Prepiriform-cortex', 'Right-Prepiriform-cortex', 'Left-Periamygdaloid-cortex', 'Right-Periamygdaloid-cortex'],
                                'all_labels':['Left-Lateral-nucleus', 'Left-Basolateral-nucleus', 'Left-Basal-nucleus', 'Left-Centromedial-nucleus', 'Left-Central-nucleus', 'Left-Medial-nucleus', 'Left-Cortical-nucleus', 'Left-Accessory-Basal-nucleus', 'Left-Corticoamygdaloid-transitio', 'Left-Anterior-amygdaloid-area-AAA', 'Left-Fusion-amygdala-HP-FAH', 'Left-Hippocampal-amygdala-transition-HATA', 'Left-Endopiriform-nucleus', 'Left-Lateral-nucleus-olfactory-tract', 'Left-Paralaminar-nucleus', 'Left-Intercalated-nucleus', 'Left-Prepiriform-cortex', 'Left-Periamygdaloid-cortex', 'Left-Envelope-Amygdala', 'Left-Extranuclear-Amydala', 'Right-Lateral-nucleus', 'Right-Basolateral-nucleus', 'Right-Basal-nucleus', 'Right-Centromedial-nucleus', 'Right-Central-nucleus', 'Right-Medial-nucleus', 'Right-Cortical-nucleus', 'Right-Accessory-Basal-nucleus', 'Right-Corticoamygdaloid-transitio', 'Right-Anterior-amygdaloid-area-AAA', 'Right-Fusion-amygdala-HP-FAH', 'Right-Hippocampal-amygdala-transition-HATA', 'Right-Endopiriform-nucleus', 'Right-Lateral-nucleus-olfactory-tract', 'Right-Paralaminar-nucleus', 'Right-Intercalated-nucleus', 'Right-Prepiriform-cortex', 'Right-Periamygdaloid-cortex', 'Right-Envelope-Amygdala', 'Right-Extranuclear-Amydala']}]

        # Orient along the longest dimension of the voxel cloud: Laminar Thalamus (LGN/MGN) & Reticular
        principal_axis_left = ['Left-Pallidum', 'Left-LGN', 'Left-MGN', 'Left-R', 'L-C.mammilare']
        principal_axis_right = ['Right-Pallidum', 'Right-LGN', 'Right-MGN', 'Right-R', 'R-C.mammilare']
        assert len(principal_axis_left) == len(principal_axis_right)

        # random directions: Thalamus (Main), Hypothalamus (Main), Striatum, Deep Amygdala
        # the value near the label is the size of a patch of tissue that fires coherently
        # its an heuristic and its based on the approximate size of the functional units of the structure and the size of the structure itself
        random_orientations = {'Left_Striatum':[5, ['Left-Caudate', 'Left-Putamen', 'Left-Accumbens-area']],
                            'Right-Striatum':[5, ['Right-Caudate', 'Right-Putamen', 'Right-Accumbens-area']],
                            'Left-Amygdala_Deep':[3.5, ['Left-Lateral-nucleus', 'Left-Basolateral-nucleus', 'Left-Basal-nucleus', 'Left-Centromedial-nucleus', 'Left-Central-nucleus', 'Left-Medial-nucleus', 'Left-Accessory-Basal-nucleus', 'Left-Endopiriform-nucleus', 'Left-Paralaminar-nucleus', 'Left-Intercalated-nucleus']],
                            'Right-Amygdala_Deep':[3.5, ['Right-Lateral-nucleus', 'Right-Basolateral-nucleus', 'Right-Basal-nucleus', 'Right-Centromedial-nucleus', 'Right-Central-nucleus', 'Right-Medial-nucleus', 'Right-Accessory-Basal-nucleus', 'Right-Endopiriform-nucleus', 'Right-Paralaminar-nucleus', 'Right-Intercalated-nucleus']],
                            'Left-Thalamus':[4, ['Left-AV', 'Left-CL', 'Left-CM', 'Left-CeM', 'Left-L-Sg', 'Left-LD', 'Left-LP', 'Left-MDl', 'Left-MDm', 'Left-MV(Re)', 'Left-PaV', 'Left-Pc', 'Left-Pf', 'Left-Pt', 'Left-PuA', 'Left-PuI', 'Left-PuL', 'Left-PuM', 'Left-PuMl', 'Left-PuMm', 'Left-VA', 'Left-VAmc', 'Left-VLa', 'Left-VLp', 'Left-VM', 'Left-VPL']],
                            'Right-Thalamus':[4, ['Right-AV', 'Right-CL', 'Right-CM', 'Right-CeM', 'Right-L-Sg', 'Right-LD', 'Right-LP', 'Right-MDl', 'Right-MDm', 'Right-MV(Re)', 'Right-PaV', 'Right-Pc', 'Right-Pf', 'Right-Pt', 'Right-PuA', 'Right-PuI', 'Right-PuL', 'Right-PuM', 'Right-PuMl', 'Right-PuMm', 'Right-VA', 'Right-VAmc', 'Right-VLa', 'Right-VLp', 'Right-VM', 'Right-VPL']],
                            'Left-Hypothalamus':[2.5, ['R-Lat-Hypothalamus', 'R-Med-Hypothalamus', 'R-Ant-Hypothalamus', 'R-Post-Hypothalamus',]],
                            'Right-Hypothalamus':[2.5, ['L-Lat-Hypothalamus', 'L-Med-Hypothalamus', 'L-Ant-Hypothalamus', 'L-Post-Hypothalamus']],
                            'Tuberal-Region':[2, ['Tuberal-Region']],}
        
        return labels_to_exclude, gradient_as_normal, principal_axis_left, principal_axis_right, random_orientations
    elif kind == 'surfaces':
        all_meshes = [  {'mesh':add_subject_dir('surfaces/freesurfer_lh_middle.stl'), 'thickness':add_subject_dir('surfaces/freesurfer_lh_middle_thickness.npy'), 'labels':glob.glob(add_subject_dir('atlas/freesurfer_surf/lh.*Parcels_labels.npy')), 'volume':add_subject_dir('surfaces/freesurfer_lh_middle_volume.npy')},
                        {'mesh':add_subject_dir('surfaces/freesurfer_rh_middle.stl'), 'thickness':add_subject_dir('surfaces/freesurfer_rh_middle_thickness.npy'), 'labels':glob.glob(add_subject_dir('atlas/freesurfer_surf/rh.*Parcels_labels.npy')), 'volume':add_subject_dir('surfaces/freesurfer_rh_middle_volume.npy')},
                        {'mesh':add_subject_dir('surfaces/cereb_inner.stl'), 'thickness':add_subject_dir('surfaces/cereb_inner_thickness.npy'), 'labels':add_subject_dir('atlas/cerebellum_surf/cereb_labels.npy')},
                        {'mesh':add_subject_dir('surfaces/hippunfold_L_dentate_middle.stl'), 'thickness':add_subject_dir('surfaces/hippunfold_L_dentate_middle_thickness.npy'), 'labels':add_subject_dir('atlas/hippunfold_surf/L_dent_labels.npy')},
                        {'mesh':add_subject_dir('surfaces/hippunfold_R_dentate_middle.stl'), 'thickness':add_subject_dir('surfaces/hippunfold_R_dentate_middle_thickness.npy'), 'labels':add_subject_dir('atlas/hippunfold_surf/R_dent_labels.npy')},
                        {'mesh':add_subject_dir('surfaces/hippunfold_L_hipp_middle.stl'), 'thickness':add_subject_dir('surfaces/hippunfold_L_hipp_middle_thickness.npy'), 'labels':add_subject_dir('atlas/hippunfold_surf/L_hipp_labels.npy')},
                        {'mesh':add_subject_dir('surfaces/hippunfold_R_hipp_middle.stl'), 'thickness':add_subject_dir('surfaces/hippunfold_R_hipp_middle_thickness.npy'), 'labels':add_subject_dir('atlas/hippunfold_surf/R_hipp_labels.npy')}]
        return all_meshes
    else:
        raise ValueError('kind must be one of ["volume", "surfaces"]')
    
def sample_volumetric(instruction_files, dipole_spacing, generator = None):
    # sample dipoles in volumetric areas
    
    if generator is None:
        generator = np.random.default_rng()
    
    ############# sample dipole position in interesting regions ###############
    
    # load dictionaries that contain various useful info about labels
    labels_to_exclude, gradient_as_normal, principal_axis_left, principal_axis_right, random_orientations = instruction_files

    # load volumetric atlas to select region to sample
    atlas = nib.load(add_subject_dir('atlas/atlas100.nii.gz'))
    assert atlas.header.get_xyzt_units()[0] == 'mm', 'Input atlas has spacing in units that are not mm, proceeding will break the code.'
    voxel_size = np.array(atlas.header.get_zooms())
    affine = atlas.affine
    atlas = atlas.get_fdata().astype(int)

    # remove all labels that dont need to have dipoles placed
    atlas[np.isin(atlas, str_to_label(labels_to_exclude))] = 0

    print(f'Sampling volume of {np.count_nonzero(atlas)} voxels at a spacing of {dipole_spacing} mm.')

    indices = np.where(atlas > 0)

    # center of interesting voxels in world space
    voxels = (affine@np.stack(list(indices)+[np.ones(len(indices[0]), dtype = int)]))[:-1]+voxel_size[:,np.newaxis]/2

    # # OLD APPROACH: intersect a regular grid of points with the interesting structure
    # # boundaries of a bounding box around interesting voxels
    # min_mm = voxels.min(axis=1)
    # max_mm = voxels.max(axis=1)+voxel_size

    # ranges = [np.arange(min_mm[i], max_mm[i], dipole_spacing) for i in range(3)]
    # xv, yv, zv = np.meshgrid(*ranges, indexing='ij')

    # # regular 3D grid, to be intersected with structures to obtain dipole positions
    # world_pos = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis = 0)

    # # uniform grid, in voxel space
    # vox_pos = np.linalg.inv(affine)@np.concatenate([world_pos, np.ones((1,world_pos.shape[1]), dtype = int)], axis = 0)
    # vox_pos = np.floor(vox_pos[:-1]).astype(int)

    # # which elements of the grid are in a region that needs dipoles
    # dipole_mask = atlas[*vox_pos]>0

    # # dipoles in voxel space
    # vox_pos = vox_pos[:,dipole_mask]

    # # dipoles in world space
    # world_pos = world_pos[:,dipole_mask]
    
    # NEW APPROACH: Poisson disk sampling on interesting voxels
    dipoles_idx = poisson_disk_subsampling(voxels.T, radius=dipole_spacing, generator = generator)

    # dipoles in voxel space
    vox_pos = np.stack(indices)[:,dipoles_idx]

    # dipoles in world space
    world_pos = voxels[:,dipoles_idx]
    
    # alter positions of dipoles in world space to put them at the center of their influence sphere
    # its like doing a step of kmeans

    # OLD: simply group voxels to dipoles by distance, and label assigned to dipole is most frequent label of closest voxels
    # nn_lookup = scipy.spatial.KDTree(world_pos.T)
    # distances, best_idx = nn_lookup.query(voxels.T)
    # dipole_labels = find_best_label(atlas[*indices], best_idx)

    # NEW: assign label to dipoles by position and group voxels by label, assigning voxels by distance to dipoles with same label
    dipole_labels = atlas[*vox_pos]
    best_idx = find_influence_area(world_pos.T, dipole_labels, voxels.T, atlas[*indices])
    good_voxels = best_idx >= 0
    best_idx = best_idx[good_voxels]

    dipole_centroids = np.concatenate([scipy.sparse.coo_matrix((voxels[0, good_voxels],( np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), world_pos.shape[1])).sum(axis=0), scipy.sparse.coo_matrix((voxels[1, good_voxels],( np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), world_pos.shape[1])).sum(axis=0), scipy.sparse.coo_matrix((voxels[2, good_voxels],( np.arange(len(best_idx)), best_idx)), shape = (len(best_idx), world_pos.shape[1])).sum(axis=0)]).T
    voxel_per_dipole = np.unique(best_idx, return_counts = True)[1]
    dipole_centroids/=voxel_per_dipole[:,np.newaxis]

    dipole_volume = voxel_per_dipole**np.prod(voxel_size)
    dipole_positions = np.array(dipole_centroids)
    ##########################################################

    # positions of dipoles in voxel space (in case they changed after the centroid step)
    vox_pos = np.linalg.inv(affine)@np.concatenate([dipole_positions.T, np.ones((1,dipole_positions.shape[0]), dtype = int)], axis = 0)
    vox_pos = np.floor(vox_pos[:-1]).astype(int)



    ################## compute preferential direction for each dipole #################
    dipole_preferential_direction = np.zeros_like(dipole_positions)
    
    ######## DIRECTION FROM GRADIENT
    for structure in gradient_as_normal:
        which_dipoles = np.isin(dipole_labels, str_to_label(structure['labels_with_dipoles']))
        
        whole_mask = np.isin(atlas, str_to_label(structure['all_labels'])).astype(float)
        smooth_mask = scipy.ndimage.gaussian_filter(whole_mask, sigma=2/voxel_size)
        grads = np.gradient(smooth_mask)
        grad_field = np.stack(grads, axis=-1)

        interesting_pos = vox_pos[:, which_dipoles]

        # the minus is to set positive direction outwards from the structure
        vectors = -grad_field[*interesting_pos]

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        norms[norms == 0] = 1.0
        orientations = vectors / norms
        
        dipole_preferential_direction[which_dipoles] = orientations
    
    ######## DIRECTION FROM PRINCIPAL AXIS
    for i in range(len(principal_axis_left)):
        left = str_to_label(principal_axis_left)[i]
        right = str_to_label(principal_axis_right)[i]
        
        indices_left = np.where(atlas == left)
        indices_right = np.where(atlas == right)

        if len(indices_left[0]) > 0:
            voxels = (affine@np.stack(list(indices_left)+[np.ones(len(indices_left[0]), dtype = int)]))[:-1]+voxel_size[:,np.newaxis]/2
            pca_left = sklearn.decomposition.PCA().fit(voxels.T)
            pca_left = pca_left.components_[0]
            
            # we make it so that the vectors have a positive y component, it is a convention to attempt enforcing consistency across different subjects
            if pca_left[1] < 0:
                pca_left *= -1
            
            dipole_preferential_direction[dipole_labels == left] = pca_left
            
        if len(indices_right[0]) > 0:
            voxels = (affine@np.stack(list(indices_right)+[np.ones(len(indices_right[0]), dtype = int)]))[:-1]+voxel_size[:,np.newaxis]/2
            pca_right = sklearn.decomposition.PCA().fit(voxels.T)
            pca_right = pca_right.components_[0]
            
            # we make it so that the vectors have a positive y component, it is a convention to attempt enforcing consistency across different subjects
            if pca_right[1] < 0:
                pca_right *= -1
            
            dipole_preferential_direction[dipole_labels == right] = pca_right

            if len(indices_left[0]) > 0:
                # flip the right vector to mantain simmetry
                if np.dot(pca_left, pca_right*np.array([-1,1,1])) < 0:
                    dipole_preferential_direction[dipole_labels == right] *= -1
    
    ######## DIRECTION FROM SMOOTH RANDOM FIELD
    for coherence_radius_mm, labels in random_orientations.values():
        which_dipoles = np.isin(dipole_labels, str_to_label(labels))
        
        if np.count_nonzero(which_dipoles) == 0:
            continue
        
        positions = dipole_positions[which_dipoles]
        
        # if coherence radius is not small enough to create at least about 3 functional structures
        # if not we decrease it based on structure size
        coherence_radius_mm = min([coherence_radius_mm, 0.5*get_structure_size(positions)/3])

        # subsample dipoles
        anchors = []
        
        # Quick KDTree of the full set
        tree_full = scipy.spatial.KDTree(positions)
        
        # We maintain a 'forbidden' mask. 
        # Once a point is picked, all neighbors within coherence_radius are forbidden.
        valid_mask = np.ones(len(positions), dtype=bool)
        
        # Random shuffle to ensure random seed placement
        search_order = generator.permutation(len(positions))

        for idx in search_order:
            if valid_mask[idx]:
                # Pick this point as an anchor
                anchors.append(idx)
                
                # Find neighbors within radius and mark them as invalid (covered)
                neighbors = tree_full.query_ball_point(positions[idx], coherence_radius_mm)
                valid_mask[neighbors] = False

        # Assign Random Orientation to Anchors
        anchor_vecs = sample_spherical(len(anchors), generator=generator)

        # Interpolate to Full Cloud
        if len(anchors)==1:
            final_orientations = np.repeat(anchor_vecs, len(positions), axis = 0)
        else:
            tree_anchors = scipy.spatial.KDTree(positions[anchors])
            dists, indices = tree_anchors.query(positions, k=min([3, len(anchors)]))

            # Inverse Distance Weighting (IDW)
            # Add epsilon to avoid division by zero (if dipole is exactly on top of anchor)
            weights = 1.0 / (dists + np.isclose(dists,0)*0.1) 
            
            # Calculate weighted average of the anchor vectors
            # indices shape: (N_dipoles, 3)
            # anchor_vecs[indices] shape: (N_dipoles, 3, 3)
            vectors_sum = np.sum(anchor_vecs[indices] * weights[:, :, np.newaxis], axis=1)
            
            final_norms = np.linalg.norm(vectors_sum, axis=1, keepdims=True)
            final_norms[np.isclose(final_norms, 0)] = 1
            final_orientations = vectors_sum / final_norms

        dipole_preferential_direction[which_dipoles] = final_orientations
    ######################################################################
    
    ############# save dipoles just sampled #######################
    output_dir = add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/volumetric/')

    output_dict = { 'dipole_positions':os.path.join(output_dir, 'dipole_positions.npy'),
                    'dipole_volume':os.path.join(output_dir, 'dipole_volume.npy'),
                    'dipole_directions':os.path.join(output_dir, 'dipole_preferential_direction.npy'),
                    'dipole_labels':os.path.join(output_dir, 'dipole_labels.npy')}

    save_npy(output_dict['dipole_positions'], dipole_positions)
    save_npy(output_dict['dipole_volume'], dipole_volume)
    save_npy(output_dict['dipole_directions'], dipole_preferential_direction)
    save_npy(output_dict['dipole_labels'], dipole_labels)
    
    return output_dict

def aggregate_array_files(path_dicts, output_dir):
    all_keys = list(map(lambda x: set(x.keys()), path_dicts))
    all_keys = set.intersection(*all_keys)
    
    for attribute in all_keys:
        out = np.concatenate(load_npy(list(map(lambda x: x[attribute], path_dicts))), axis = 0)
        save_npy(os.path.join(output_dir,attribute+'.npy'), out)
    return

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def unify_attribute(all_arrays_dict, attribute):
    all_atlases = list(filter(lambda x: isinstance(x, dict), map(lambda x: x[attribute], all_arrays_dict)))
    all_atlases = list(map(lambda x: set(x.keys()), all_atlases))

    # safety check
    assert all_equal(all_atlases)

    all_atlases = set().union(*all_atlases)

    if len(all_atlases) > 0:
        for atlas_name in all_atlases:
            for i in range(len(all_arrays_dict)):
                if not isinstance(all_arrays_dict[i][attribute], dict):
                    all_arrays_dict[i][atlas_name+'_'+attribute] = all_arrays_dict[i][attribute]
                else:
                    all_arrays_dict[i][atlas_name+'_'+attribute] = all_arrays_dict[i][attribute][atlas_name]

        for i in range(len(all_arrays_dict)):
            all_arrays_dict[i].pop(attribute)

    return all_arrays_dict

def homogenize_dicts(all_dicts):
    all_keys = list(map(lambda x: set(x.keys()), all_dicts))
    all_keys = set.intersection(*all_keys)
    
    out = []
    for i in range(len(all_dicts)):
        out.append({key:all_dicts[i][key] for key in all_keys})
    return out

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Randomly samples positions in the head for dipoles to be placed at.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 1. Define the Subject Folder Argument
    parser.add_argument(
        '--subject_dir',
        type=str,
        required=False,
        default='/subject/', # to be used inside container
        help='Path to the subject folder (e.g., /SUBJECTS/<subjectname>/)'
    )
    
    # 1. Define the Subject Folder Argument
    parser.add_argument(
        '--dipole_spacing',
        type=float,
        required=True,
        help='Spacing between dipoles, in mm (typical values range from 1 to 10)'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the base directory and dipole spacing from the command line
    subject_dir = args.subject_dir
    dipole_spacing = args.dipole_spacing
    
    # make output directory if needed
    os.makedirs(add_subject_dir(f'dipoles/'), exist_ok=True)
    
    if os.path.isdir(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/')):
        print(f'WARNING: Dipoles at {dipole_spacing} mm spacing already detected in subject folder, skipping computation.')
    else:
        # load random number generator
        generator = np.random.default_rng()

        # sample dipoles
        surface_dipoles_dict = sample_all_surfaces(get_instruction_files('surfaces'), dipole_spacing, generator = generator)
        volumetric_dipoles_dict = sample_volumetric(get_instruction_files('volume'), dipole_spacing, generator = generator)

        # save dipoles to disk
        all_arrays_dict = unify_attribute([volumetric_dipoles_dict] + surface_dipoles_dict, 'dipole_labels')
        aggregate_array_files(all_arrays_dict, add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/'))
