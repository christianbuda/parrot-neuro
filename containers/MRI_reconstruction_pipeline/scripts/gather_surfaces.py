import shutil
import trimesh
import nibabel as nib
import numpy as np
import argparse
import os

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

def surf_to_stl(mesh, output_path, process = True, **kwargs):
    """
    Reads a (vertices, faces) tuple and converts it to an STL mesh file.
    """
    mesh = trimesh.Trimesh(vertices = mesh[0], faces = mesh[1], process = False, validate = False)
    
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
    brain = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/brain.surf")))
    surf_to_stl(brain, add_subject_dir('surfaces/freesurfer_BEM_brain.stl'))

    inner_skull = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/inner_skull.surf")))
    surf_to_stl(inner_skull, add_subject_dir('surfaces/freesurfer_BEM_inner_skull.stl'))

    outer_skull = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/outer_skull.surf")))
    surf_to_stl(outer_skull, add_subject_dir('surfaces/freesurfer_BEM_outer_skull.stl'))

    outer_skin = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_subject_dir("freesurfer/bem/outer_skin.surf")))
    surf_to_stl(outer_skin, add_subject_dir('surfaces/freesurfer_BEM_outer_skin.stl'))


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
