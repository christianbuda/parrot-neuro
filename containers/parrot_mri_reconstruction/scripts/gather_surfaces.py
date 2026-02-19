import shutil
import trimesh
import nibabel as nib
import numpy as np
import argparse
import os
import pymeshfix


def read_vtk(input):
    with open(input, 'r') as f:
        mesh = f.readline()
        
    ver = int(mesh.strip()[-3])
    
    if ver == 2:
        return read_vtk2(input)
    elif ver == 3:
        return read_fsl_vtk(input)
    elif ver == 4:
        return read_vtk4(input)
    
    raise ValueError("Wrong file! This is not a general purpose reader, don't use it outside the scope of this script")

def read_vtk2(input):
    with open(input, 'r') as f:
        mesh = f.readlines()

    header = 5
    nverts = int(mesh[4].split()[1])
    nfaces = int(mesh[nverts+header].split()[1])

    # just a check
    assert mesh[0] == '# vtk DataFile Version 2.0\n'
    assert mesh[1] == 'Saved using mVTK\n'
    assert mesh[2] == 'ASCII\n'
    assert mesh[3] == 'DATASET POLYDATA\n'
    assert mesh[4].split()[0::2] == ['POINTS', 'float']
    assert mesh[nverts+header].split()[0] == 'POLYGONS'
    assert int(mesh[nverts+header].split()[1])*4 == int(mesh[nverts+header].split()[2])

    points = mesh[header:header+nverts]
    faces = mesh[header+nverts+1:header+nverts+1+nfaces]

    points = np.array(list(map(lambda x: x.split(), points))).astype(float)
    faces = np.array(list(map(lambda x: x.split()[1:], faces))).astype(int)

    return(points,faces)

def read_fsl_vtk(input):
    with open(input, 'r') as f:
        mesh = f.readlines()

    header = 5
    nverts = int(mesh[4].split()[1])
    nfaces = int(mesh[nverts+header].split()[1])
    
    # just a check
    assert mesh[0] == '# vtk DataFile Version 3.0\n'
    assert mesh[1] == 'this file was written using fslvtkio\n'
    assert mesh[2] == 'ASCII\n'
    assert mesh[3] == 'DATASET POLYDATA\n'
    assert mesh[4].split()[0::2] == ['POINTS', 'float']
    assert mesh[nverts+header].split()[0] == 'POLYGONS'
    assert int(mesh[nverts+header].split()[1])*4 == int(mesh[nverts+header].split()[2])

    points = mesh[header:header+nverts]
    faces = mesh[header+nverts+1:header+nverts+1+nfaces]

    points = np.array(list(map(lambda x: x.split(), points))).astype(float)
    faces = np.array(list(map(lambda x: x.split()[1:], faces))).astype(int)
    
    return(points,faces)

def read_vtk4(input):
    with open(input, 'r') as f:
        mesh = f.readlines()

    header = 5
    nverts = int(np.ceil(int(mesh[4].split()[1])/3))
    nstrips = int(mesh[nverts+header].split()[1])

    # just a check
    assert mesh[0] == '# vtk DataFile Version 4.0\n'
    assert mesh[1] == 'vtk output\n'
    assert mesh[2] == 'ASCII\n'
    assert mesh[3] == 'DATASET POLYDATA\n'
    assert mesh[4].split()[0::2] == ['POINTS', 'float']
    assert mesh[nverts+header].split()[0] == 'TRIANGLE_STRIPS'

    points = mesh[header:header+nverts]
    points = np.concatenate(list(map(lambda x: np.array(x.split()).astype(float).reshape((-1,3)), points)))

    strips = mesh[header+nverts+1:header+nverts+1+nstrips]
    faces = np.concatenate(list(map(lambda x: np.lib.stride_tricks.sliding_window_view(np.array(x.split()[1:]).astype(int), window_shape=(3)), strips)))

    # final check
    assert faces.shape[0]+nstrips*3 == int(mesh[nverts+header].split()[2])
    
    return(points,faces)

def compute_face_normals(vertices, faces, return_area = False):
    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]
    
    
    if not return_area:
        return np.cross(B-A, C-A)/np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
    else:
        norms = np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
        return np.cross(B-A, C-A)/norms, norms[:,0]/2

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

def surf_to_stl(mesh, output_path, process = True, **kwargs):
    """
    Reads a (vertices, faces) tuple and converts it to an STL mesh file.
    """
    mesh = to_trimesh(mesh)
    
    for key, val in kwargs.items():
        assert len(mesh.vertices) == len(val), f"Vertex attributes don't match mesh size, check {key} attribute."
        mesh.vertex_attributes[key] = val
    
    if process:
        # Perform Cleaning Operations
        mesh.process(validate = True)
    
    if output_path[-4:]=='.stl':
        output_path = output_path[:-4]
    
    for key in kwargs.keys():
        np.save(f"{output_path}_{key}.npy", mesh.vertex_attributes[key])
    
    mesh.export(output_path+'.stl')

    return
    
def make_cereb_attributes(input_path):
    with open(input_path, 'r') as f:
        mesh = f.readlines()
    

    header = 5
    nverts = int(mesh[4].split()[1])
    nfaces = int(mesh[nverts+header].split()[1])
    
    output_dict = {}
    
    current_idx = header+nverts+1+nfaces+1
    while(current_idx<len(mesh)):
        key = mesh[current_idx].split()[1]
        dtype = mesh[current_idx].split()[2]
        if dtype == 'int':
            dtype = int
        if dtype == 'float':
            dtype = float
        val = mesh[current_idx+2:current_idx+2+nverts]
        val = np.array(list(map(lambda x: dtype(x.strip()), val)))
        output_dict[key] = val
        current_idx += 2+nverts
    
    return output_dict['thickness'], output_dict['GMparc']

def get_hippocampus_labels(label_path):
    with open(label_path, 'r') as f:
        hippocampus_labels = f.readlines()

    hippocampus_labels = list(map(lambda x: x.strip().split(','), hippocampus_labels[1:]))
    hippocampus_labels = list(map(lambda x: (int(x[0]), x[1].strip()), hippocampus_labels))
    hippocampus_labels = dict(hippocampus_labels)
    
    # add background
    hippocampus_labels[0] = 'Unknown'
    return hippocampus_labels

def make_dentate_attributes(label_path, volume_path, surface_path):
    dentate_label = dict(map(lambda x: (x[1], x[0]), get_hippocampus_labels(label_path).items()))['dentate_gyrus']
    dentate_img = nib.load(volume_path)
    voxel_volume = np.prod(dentate_img.header.get_zooms())
    dentate_img = dentate_img.get_fdata()
    dentate_volume = np.count_nonzero(dentate_img == dentate_label)*voxel_volume

    mesh = nib.load(surface_path).agg_data(('pointset', 'triangle'))
    mesh = trimesh.Trimesh(vertices = mesh[0], faces = mesh[1], process = False, validate = False)
    nverts = mesh.vertices.shape[0]
    dentate_area = mesh.area
    dentate_thickness = dentate_volume/dentate_area
    dentate_thickness = np.repeat(dentate_thickness, nverts)
    dentate_labels = np.repeat(dentate_label, nverts)
    return dentate_thickness, dentate_labels
    
def apply_trans(mesh, trans):
    # this utils takes a trimesh mesh or a (vertices, faces) tuple and applies an affine transformation
    
    if isinstance(mesh, trimesh.Trimesh):
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        vertices = (trans@(np.concatenate([vertices, np.ones(len(vertices))[:,np.newaxis]], axis = 1).T)).T[:,:3]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif isinstance(mesh, tuple):
        assert len(mesh)==2, 'mesh tuple format must have two entries (vertices, faces)'
        vertices, faces = mesh
        vertices = (trans@(np.concatenate([vertices, np.ones(len(vertices))[:,np.newaxis]], axis = 1).T)).T[:,:3]
        mesh = vertices, faces
    else:
        raise ValueError('Mesh must be either a trimesh mesh, or a tuple mesh')
    
    return mesh

def fix_FIRST_mesh(mesh):
    # this utils takes a FSL FIRST mesh and transforms it in nifti voxel space, reorients it, and brings it in world space
    # this is needed because FIRST oriented the meshes clockwise, that's probably intentional and due to the radiological convention that FSL follows (maaaaybe)
    
    ### OLD ###
    # this is equivalent to the trans variable approach below, just clearer
    # the code rescales the mesh if the MRI is not 1mm isotropic
    # and flips the structures left/right
    #
    # vertices = vertices/orig_T1.header['pixdim'][1:4]
    # vertices[:, 0] = orig_T1.shape[0]-vertices[:, 0]
    #############
    
    trans = np.eye(4)
    trans[np.arange(3), np.arange(3)] = 1/orig_T1.header['pixdim'][1:4]
    trans[0,0] *= -1
    trans[0, 3] = orig_T1.shape[0]
    
    return apply_trans(mesh, orig_T1.affine@trans)

def fix_freesurfer_mesh(mesh):
    # this utils takes a freesurfer mesh and transforms it in nifti world space
    return apply_trans(mesh, fs_T1_affine@np.linalg.inv(vox2ras_tkr))

def repair_mesh_topology(mesh):
    # repair topology# in mne docs they say that scalp surfaces can have topological issues
    # here we attempt repairing
    mesh = pymeshfix.clean_from_arrays(np.array(mesh[0]), np.array(mesh[1]))
    return mesh

def to_trimesh(mesh):
    # convert input to trimesh mesh object
    
    if isinstance(mesh, trimesh.Trimesh):
        pass
    elif isinstance(mesh, tuple):
        assert len(mesh)==2, 'mesh tuple format must have two entries (vertices, faces)'
        vertices, faces = mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process = False, validate = False)
    else:
        raise ValueError('Mesh must be either a trimesh mesh, or a tuple mesh')
    
    return mesh

def fix_intersection(fixed_mesh, moving_mesh, min_dist, step_size, dist_check = None):
    # inflates moving mesh to make it so that it's at at least min_dist distance OUTSIDE fixed_mesh
    # smoothing is applied at the end, so the min_dist requirement is not satisfied strictly
    
    if dist_check is None:
        dist_check = min_dist*0.8
    
    test_points = np.copy(moving_mesh.vertices)
    dists = trimesh.proximity.signed_distance(fixed_mesh, test_points)
    
    bad_points = dists>-min_dist
    
    if not np.any(bad_points):
        return moving_mesh
    
    while np.any(bad_points):
        outward_dir = compute_vertex_normals(fixed_mesh.vertices, fixed_mesh.faces)[bad_points]
        test_points[bad_points] += outward_dir*step_size
        
        dists = trimesh.proximity.signed_distance(fixed_mesh, test_points)
        bad_points = dists>-min_dist

    output_mesh = trimesh.Trimesh(vertices=test_points, faces=moving_mesh.faces, process = False, validate = False)
    output_mesh = trimesh.smoothing.filter_taubin(output_mesh, iterations=3)
    
    # check again for violations
    dists = trimesh.proximity.signed_distance(fixed_mesh, output_mesh.vertices)
    bad_points = dists>-dist_check  # we relax the constraint slightly
    
    if np.any(bad_points):
        print('Smoothing created new intersections, attempting repair')
        return fix_intersection(fixed_mesh, output_mesh, min_dist/0.8, step_size, dist_check=dist_check)
    
    return output_mesh

def add_subject_dir(*paths):
    if len(paths)==1:
        return os.path.join(subject_dir, paths[0])
    return tuple([add_subject_dir(x) for x in paths])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert relevant surfaces to nifti world space and save in .stl")
    parser.add_argument('--subject_dir', type=str, required = True, help='Path to the subject directory containing all reconstructions')
    args = parser.parse_args()

    # Get the base directory from the command line
    subject_dir = args.subject_dir

    vox2ras_tkr = nib.load(add_subject_dir("freesurfer/mri/T1.mgz")).header.get_vox2ras_tkr()
    fs_T1_affine = nib.load(add_subject_dir('freesurfer/mri/T1.mgz')).affine
    orig_T1 = nib.load(add_subject_dir('raw/T1.nii.gz'))


    # FSL first surfaces
    brstem = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-BrStem_first.vtk')))
    surf_to_stl(brstem, add_subject_dir('surfaces/first_BrStem.stl'))

    Laccu = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Accu_first.vtk')))
    surf_to_stl(Laccu, add_subject_dir('surfaces/first_L_Accu.stl'))

    Lamyg = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Amyg_first.vtk')))
    surf_to_stl(Lamyg, add_subject_dir('surfaces/first_L_Amyg.stl'))

    Lcaud = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Caud_first.vtk')))
    surf_to_stl(Lcaud, add_subject_dir('surfaces/first_L_Caud.stl'))

    Lhipp = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Hipp_first.vtk')))
    surf_to_stl(Lhipp, add_subject_dir('surfaces/first_L_Hipp.stl'))

    Lpall = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Pall_first.vtk')))
    surf_to_stl(Lpall, add_subject_dir('surfaces/first_L_Pall.stl'))

    Lputa = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Puta_first.vtk')))
    surf_to_stl(Lputa, add_subject_dir('surfaces/first_L_Puta.stl'))

    Lthal = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-L_Thal_first.vtk')))
    surf_to_stl(Lthal, add_subject_dir('surfaces/first_L_Thal.stl'))

    Raccu = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Accu_first.vtk')))
    surf_to_stl(Raccu, add_subject_dir('surfaces/first_R_Accu.stl'))

    Ramyg = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Amyg_first.vtk')))
    surf_to_stl(Ramyg, add_subject_dir('surfaces/first_R_Amyg.stl'))

    Rcaud = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Caud_first.vtk')))
    surf_to_stl(Rcaud, add_subject_dir('surfaces/first_R_Caud.stl'))

    Rhipp = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Hipp_first.vtk')))
    surf_to_stl(Rhipp, add_subject_dir('surfaces/first_R_Hipp.stl'))

    Rpall = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Pall_first.vtk')))
    surf_to_stl(Rpall, add_subject_dir('surfaces/first_R_Pall.stl'))

    Rputa = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Puta_first.vtk')))
    surf_to_stl(Rputa, add_subject_dir('surfaces/first_R_Puta.stl'))

    Rthal = fix_FIRST_mesh(read_vtk(add_subject_dir('fsl_first/FSL-R_Thal_first.vtk')))
    surf_to_stl(Rthal, add_subject_dir('surfaces/first_R_Thal.stl'))


    # freesurfer surfaces
    Lwhite = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/surf/lh.white")))
    surf_to_stl(Lwhite, add_subject_dir('surfaces/freesurfer_lh_white.stl'))

    Rwhite = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/surf/rh.white")))
    surf_to_stl(Rwhite, add_subject_dir('surfaces/freesurfer_rh_white.stl'))

    Lgray = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/surf/lh.pial")))
    surf_to_stl(Lgray, add_subject_dir('surfaces/freesurfer_lh_pial.stl'))

    Rgray = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/surf/rh.pial")))
    surf_to_stl(Rgray, add_subject_dir('surfaces/freesurfer_rh_pial.stl'))

    Lmiddle = ((Lwhite[0]+Lgray[0])/2, Lwhite[1])
    kwargs = {}
    for nparcels in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        labels, _, _ = nib.freesurfer.read_annot(add_subject_dir(f'freesurfer/label/lh.Schaefer2018_{nparcels}Parcels_17Networks_order.annot'))
        kwargs[f'original_labels_{nparcels}'] = labels
    surf_to_stl(Lmiddle, add_subject_dir('surfaces/freesurfer_lh_middle.stl'), volume = nib.freesurfer.io.read_morph_data(add_subject_dir('freesurfer/surf/lh.volume')).astype(float), thickness = nib.freesurfer.io.read_morph_data(add_subject_dir('freesurfer/surf/lh.thickness')).astype(float), **kwargs)

    Rmiddle = ((Rwhite[0]+Rgray[0])/2, Rwhite[1])
    kwargs = {}
    for nparcels in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        labels, _, _ = nib.freesurfer.read_annot(add_subject_dir(f'freesurfer/label/rh.Schaefer2018_{nparcels}Parcels_17Networks_order.annot'))
        kwargs[f'original_labels_{nparcels}'] = labels
    surf_to_stl(Rmiddle, add_subject_dir('surfaces/freesurfer_rh_middle.stl'), volume = nib.freesurfer.io.read_morph_data(add_subject_dir('freesurfer/surf/rh.volume')).astype(float), thickness = nib.freesurfer.io.read_morph_data(add_subject_dir('freesurfer/surf/rh.thickness')).astype(float), **kwargs)
    
    
    # BEM surfaces
    # watershed bem creates both brain and outer_skin from the MRI volume
    # and then inflates the brain to obtain inner_skull
    # and deflates the outer_skin to obtain outer_skull
    # It seems that inner_skull is generally decent, while outer_scalp (and, consequently, outer_skull) can be pretty bad
    # so we check for intersections and inflate outer_skull and outer_scalp having inner_skull as a reference
    
    # NOTE that this fix is not smart, it does not know where the anatomy is
    # it just inflates enough to get numerical stability in the BEM solver, the surfaces will likely still be crooked
    brain = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/brain.surf"))))
    surf_to_stl(brain, add_subject_dir('surfaces/freesurfer_BEM_brain.stl'))

    inner_skull = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/inner_skull.surf"))))
    surf_to_stl(inner_skull, add_subject_dir('surfaces/freesurfer_BEM_inner_skull.stl'))

    outer_skull = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/outer_skull.surf"))))
    outer_skull = fix_intersection(inner_skull, outer_skull, min_dist = 4, step_size = 0.1)
    surf_to_stl(outer_skull, add_subject_dir('surfaces/freesurfer_BEM_outer_skull.stl'))

    outer_skin = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/outer_skin.surf"))))
    outer_skin = fix_intersection(outer_skull, outer_skin, min_dist = 3, step_size = 0.1)
    surf_to_stl(outer_skin, add_subject_dir('surfaces/freesurfer_BEM_outer_skin.stl'))
    
    # MNE scalp
    scalp = (np.load(add_subject_dir("freesurfer/bem/vertices-scalp.npy")), np.load(add_subject_dir("freesurfer/bem/faces-scalp.npy")))
    scalp = fix_freesurfer_mesh(scalp)
    scalp = repair_mesh_topology(scalp)
    surf_to_stl(scalp, add_subject_dir('surfaces/MNE_scalp.stl'))


    # charm surfaces
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/white.stl"), add_subject_dir("surfaces/charm_white.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/gray.stl"), add_subject_dir("surfaces/charm_gray.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/scalp.stl"), add_subject_dir("surfaces/charm_scalp.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/eyes_balls.stl"), add_subject_dir("surfaces/charm_eyes_balls.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/eyes_muscles.stl"), add_subject_dir("surfaces/charm_eyes_muscles.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/CSF.stl"), add_subject_dir("surfaces/charm_CSF.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/bone_compact.stl"), add_subject_dir("surfaces/charm_bone_compact.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/bone_spongy.stl"), add_subject_dir("surfaces/charm_bone_spongy.stl"))
    shutil.copyfile(add_subject_dir("simnibs_charm/converted/blood.stl"), add_subject_dir("surfaces/charm_blood.stl"))


    # hippocampus surfaces
    surf_to_stl(nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-L_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_subject_dir('surfaces/hippunfold_L_hipp_middle.stl'), thickness = nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-L_space-T1w_den-0p5mm_label-hipp_thickness.shape.gii')).darrays[0].data.astype(float), original_labels = nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-L_space-T1w_den-0p5mm_label-hipp_atlas-multihist7_subfields.label.gii')).darrays[0].data)
    surf_to_stl(nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-R_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_subject_dir('surfaces/hippunfold_R_hipp_middle.stl'), thickness = nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-R_space-T1w_den-0p5mm_label-hipp_thickness.shape.gii')).darrays[0].data.astype(float), original_labels = nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-R_space-T1w_den-0p5mm_label-hipp_atlas-multihist7_subfields.label.gii')).darrays[0].data)
    
    # dentate gyrus surfaces
    thickness, labels = make_dentate_attributes(*add_subject_dir('hippunfold/LABELS.txt', 'hippunfold/anat/sub-subject_hemi-L_space-cropT1w_desc-subfields_atlas-multihist7_dseg.nii.gz', 'hippunfold/surf/sub-subject_hemi-L_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii'))
    surf_to_stl(nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-L_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_subject_dir('surfaces/hippunfold_L_dentate_middle.stl'), thickness = thickness, original_labels = labels)
    thickness, labels = make_dentate_attributes(*add_subject_dir('hippunfold/LABELS.txt', 'hippunfold/anat/sub-subject_hemi-R_space-cropT1w_desc-subfields_atlas-multihist7_dseg.nii.gz', 'hippunfold/surf/sub-subject_hemi-R_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii'))
    surf_to_stl(nib.load(add_subject_dir('hippunfold/surf/sub-subject_hemi-R_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_subject_dir('surfaces/hippunfold_R_dentate_middle.stl'), thickness = thickness, original_labels = labels)

    # cerebellum surfaces
    surf_to_stl(read_vtk(add_subject_dir('cerebellum/nonlinear_Cerebellum_Surf_GM_Labels.vtk')), add_subject_dir('surfaces/cereb_gray.stl'))
    surf_to_stl(read_vtk(add_subject_dir('cerebellum/nonlinear_Cerebellum_Surf_WM_Labels.vtk')), add_subject_dir('surfaces/cereb_white.stl'))
    surf_to_stl(read_vtk(add_subject_dir('cerebellum/nonlinear_Cerebellum_Surf_WM_Labels.vtk')), add_subject_dir('surfaces/cereb_inner_raw.stl'))
    thickness, labels = make_cereb_attributes(add_subject_dir('cerebellum/nonlinear_manifold_Cerebellum_Inner_Surf_With_Features.vtk'))
    surf_to_stl(read_vtk(add_subject_dir('cerebellum/nonlinear_manifold_Cerebellum_Inner_Surf_With_Features.vtk')), add_subject_dir('surfaces/cereb_inner.stl'), thickness = thickness, original_labels = labels, process = False)
