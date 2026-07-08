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

def surf_to_ply(mesh, output_path, process = True, **kwargs):
    """
    Reads a (vertices, faces) tuple, adds attributes and dumps it to disk
    """
    mesh = to_trimesh(mesh)
    
    for key, val in kwargs.items():
        assert len(mesh.vertices) == len(val), f"Vertex attributes don't match mesh size, check {key} attribute."
        mesh.vertex_attributes[key] = val
    
    if process:
        # Perform Cleaning Operations
        mesh.process(validate = True)
    
    if output_path[-4:]=='.ply':
        output_path = output_path[:-4]
    
    for key in kwargs.keys():
        np.save(f"{output_path}_{key}.npy", mesh.vertex_attributes[key])
    
    # clear mesh attributes before exporting
    mesh.vertex_attributes.clear()
    
    mesh.export(output_path+'.ply')
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

def fix_intersection(fixed_mesh, moving_mesh, min_dist, step_size,
                     max_iter=300, smooth_rounds=3, relax=0.8, step_gain=1.0):
    # Inflate moving_mesh so it sits at least `min_dist` mm OUTSIDE fixed_mesh,
    # then smooth away the inflation bumps. The outward direction for each
    # violating vertex comes from fixed_mesh's closest triangle (the signed-
    # distance gradient), so the repair is robust however badly moving_mesh is
    # deformed -- no vertex correspondence is assumed.
    #
    # WHY THIS IS WRITTEN THE WAY IT IS: the previous version smoothed once and,
    # if smoothing pulled any vertex back within the band, RECURSED with an
    # ever-larger min_dist (min_dist/0.8). But phase-1 inflation only moves the
    # violating vertices, which creates sharp isolated spikes, and Taubin
    # smoothing preferentially attenuates exactly those spikes -- so inflation
    # and smoothing are antagonistic. Growing min_dist built taller spikes that
    # smoothing pulled back harder; with no depth cap the recursion could
    # diverge on subjects where the required clearance is geometrically capped
    # (a corresponding outer vertex in a concavity simply cannot get min_dist
    # away from every part of fixed_mesh). That was the "stuck forever" bug.
    #
    # Fix: CONSTRAIN smoothing so it can never pull a vertex back inside the
    # (relaxed) clearance band -- this removes the antagonism by construction --
    # and hard-bound both loops as a backstop so a pathological head degrades to
    # a best-effort surface + WARNING instead of hanging. These surfaces only
    # need to be non-intersecting for BEM solver stability, so best-effort is an
    # acceptable failure mode.

    def signed_clearance(points):
        # AUTHORITATIVE signed clearance: -signed_distance, +outside / -inside.
        # trimesh's signed_distance uses a pseudonormal sign test that stays correct
        # near edges/folds. A naive dot(point - closest, face_normal) does NOT -- on
        # a deformed shell it reads a deeply-inside vertex that happens to be closest
        # to a nearby fold as "outside", so the scan never flags it and Phase 1
        # never repairs it, silently leaving an intersection. So the SIGN must come
        # from signed_distance; closest_point is used only for the push direction.
        return -trimesh.proximity.signed_distance(fixed_mesh, points)

    def outward_dir(points):
        # Outward push direction: normal of the closest fixed_mesh triangle (the
        # signed-distance gradient), from one cheap closest_point query. Stepping
        # along this -- not a correspondent vertex normal -- is what makes Phase 1
        # converge in a few iterations even in a deep dent.
        _, _, tri = trimesh.proximity.closest_point(fixed_mesh, points)
        return fixed_mesh.face_normals[tri]

    test_points = np.copy(moving_mesh.vertices)
    faces = np.array(moving_mesh.faces)

    # One full authoritative scan to find EVERY violating vertex (including the
    # deeply-inside ones on a deformed shell). Violations are localized, so Phase 1
    # then works only on the shrinking bad subset.
    bad_idx = np.flatnonzero(signed_clearance(test_points) < min_dist)
    if bad_idx.size == 0:
        return moving_mesh  # already clear -- leave the surface untouched

    # Phase 1: ADAPTIVE outward stepping. The signed clearance says exactly how far
    # short each vertex is, so we push it out by ~that deficit along the closest-
    # triangle normal in one shot instead of creeping a fixed 0.1 mm -- a ~14 mm,
    # 2500-vertex collapse clears in a handful of iterations (minutes -> seconds).
    # `step_size` is a small progress floor; `max_iter` is a hard backstop.
    it = 0
    while bad_idx.size and it < max_iter:
        deficit = min_dist - signed_clearance(test_points[bad_idx])
        still = deficit > 0
        bad_idx = bad_idx[still]                     # cleared vertices drop out
        if bad_idx.size == 0:
            break
        step_len = np.maximum(deficit[still] * step_gain, step_size)
        test_points[bad_idx] += outward_dir(test_points[bad_idx]) * step_len[:, None]
        it += 1
    if bad_idx.size:
        print(f"[fix_intersection] WARNING: {bad_idx.size} vertices could not clear "
              f"{min_dist} mm after {max_iter} adaptive steps; keeping best effort.")

    # Snapshot the post-Phase-1 positions. Every vertex here is authoritatively
    # >= min_dist outside (except best-effort ones, which are still the best we got),
    # so this is a known-safe fallback the smoothing below can revert to.
    cleared = np.copy(test_points)

    # Phase 2: smooth to relax the inflation spikes, but revert any smoothed vertex
    # that falls back inside the relaxed band (relax*min_dist) to its cleared
    # position, so smoothing can never re-create an intersection -- no recursion,
    # terminates in `smooth_rounds`. The revert-check is a FULL-mesh authoritative
    # scan: a vertex that Phase 1 left untouched can still be dragged inward by
    # smoothing when it neighbours a large inflated bulge, so checking only the
    # moved subset silently leaks intersections.
    for _ in range(smooth_rounds):
        mesh = trimesh.Trimesh(vertices=test_points, faces=faces, process=False, validate=False)
        smoothed = np.array(trimesh.smoothing.filter_taubin(mesh, iterations=3).vertices)
        reverted = signed_clearance(smoothed) < relax * min_dist
        smoothed[reverted] = cleared[reverted]
        test_points = smoothed

    # Final guarantee: any vertex STILL inside the relaxed band (e.g. one a fold
    # trap keeps the smooth/revert cycle from settling) is snapped back to its
    # known-safe Phase-1 position and left there. So the returned surface is
    # non-intersecting wherever Phase 1 succeeded -- by construction, not just in
    # expectation. Anything that remains inside was never clearable (best-effort,
    # already flagged) and is handled by the downstream repair + non-fatal solver.
    resid = signed_clearance(test_points) < relax * min_dist
    if np.any(resid):
        test_points[resid] = cleared[resid]

    return trimesh.Trimesh(vertices=test_points, faces=faces, process=False, validate=False)

def add_output_dir(*paths):
    if len(paths)==1:
        return os.path.join(output_dir, paths[0])
    return tuple([add_output_dir(x) for x in paths])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert relevant surfaces to nifti world space and save in .ply")
    
    parser.add_argument(
        '--subject', 
        type=str,
        required=True,
        help='Subject ID (e.g. "01")'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output folder (e.g., /derivatives/)'
    )

    # Recon surfaces dir: fastsurfer (full mode) or freesurfer (freesurfer/HCP mode).
    parser.add_argument(
        '--surf_dir',
        type=str,
        default='fastsurfer',
        help='Derivatives subdir holding the recon surfaces (fastsurfer or freesurfer)'
    )

    args = parser.parse_args()

    # Get the base directory from the command line
    subject = args.subject
    output_dir = args.output_dir
    surf_dir = args.surf_dir

    vox2ras_tkr = nib.load(add_output_dir(f"{surf_dir}/sub-{subject}/mri/T1.mgz")).header.get_vox2ras_tkr()
    fs_T1_affine = nib.load(add_output_dir(f'{surf_dir}/sub-{subject}/mri/T1.mgz')).affine
    orig_T1 = nib.load(add_output_dir(f'raw/sub-{subject}/T1.nii.gz'))


    # FSL first surfaces
    brstem = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-BrStem_first.vtk')))
    surf_to_ply(brstem, add_output_dir(f'surfaces/sub-{subject}/first_BrStem.ply'))

    Laccu = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Accu_first.vtk')))
    surf_to_ply(Laccu, add_output_dir(f'surfaces/sub-{subject}/first_L_Accu.ply'))

    Lamyg = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Amyg_first.vtk')))
    surf_to_ply(Lamyg, add_output_dir(f'surfaces/sub-{subject}/first_L_Amyg.ply'))

    Lcaud = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Caud_first.vtk')))
    surf_to_ply(Lcaud, add_output_dir(f'surfaces/sub-{subject}/first_L_Caud.ply'))

    Lhipp = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Hipp_first.vtk')))
    surf_to_ply(Lhipp, add_output_dir(f'surfaces/sub-{subject}/first_L_Hipp.ply'))

    Lpall = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Pall_first.vtk')))
    surf_to_ply(Lpall, add_output_dir(f'surfaces/sub-{subject}/first_L_Pall.ply'))

    Lputa = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Puta_first.vtk')))
    surf_to_ply(Lputa, add_output_dir(f'surfaces/sub-{subject}/first_L_Puta.ply'))

    Lthal = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-L_Thal_first.vtk')))
    surf_to_ply(Lthal, add_output_dir(f'surfaces/sub-{subject}/first_L_Thal.ply'))

    Raccu = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Accu_first.vtk')))
    surf_to_ply(Raccu, add_output_dir(f'surfaces/sub-{subject}/first_R_Accu.ply'))

    Ramyg = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Amyg_first.vtk')))
    surf_to_ply(Ramyg, add_output_dir(f'surfaces/sub-{subject}/first_R_Amyg.ply'))

    Rcaud = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Caud_first.vtk')))
    surf_to_ply(Rcaud, add_output_dir(f'surfaces/sub-{subject}/first_R_Caud.ply'))

    Rhipp = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Hipp_first.vtk')))
    surf_to_ply(Rhipp, add_output_dir(f'surfaces/sub-{subject}/first_R_Hipp.ply'))

    Rpall = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Pall_first.vtk')))
    surf_to_ply(Rpall, add_output_dir(f'surfaces/sub-{subject}/first_R_Pall.ply'))

    Rputa = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Puta_first.vtk')))
    surf_to_ply(Rputa, add_output_dir(f'surfaces/sub-{subject}/first_R_Puta.ply'))

    Rthal = fix_FIRST_mesh(read_vtk(add_output_dir(f'fslfirst/sub-{subject}/FSL-R_Thal_first.vtk')))
    surf_to_ply(Rthal, add_output_dir(f'surfaces/sub-{subject}/first_R_Thal.ply'))


    # freesurfer surfaces
    Lwhite = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/surf/lh.white")))
    surf_to_ply(Lwhite, add_output_dir(f'surfaces/sub-{subject}/freesurfer_lh_white.ply'))

    Rwhite = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/surf/rh.white")))
    surf_to_ply(Rwhite, add_output_dir(f'surfaces/sub-{subject}/freesurfer_rh_white.ply'))

    Lgray = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/surf/lh.pial")))
    surf_to_ply(Lgray, add_output_dir(f'surfaces/sub-{subject}/freesurfer_lh_pial.ply'))

    Rgray = fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/surf/rh.pial")))
    surf_to_ply(Rgray, add_output_dir(f'surfaces/sub-{subject}/freesurfer_rh_pial.ply'))

    Lmiddle = ((Lwhite[0]+Lgray[0])/2, Lwhite[1])
    kwargs = {}
    for nparcels in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        labels, _, _ = nib.freesurfer.read_annot(add_output_dir(f'{surf_dir}/sub-{subject}/label/lh.Schaefer2018_{nparcels}Parcels_17Networks_order.annot'))
        kwargs[f'original_labels_{nparcels}'] = labels
    surf_to_ply(Lmiddle, add_output_dir(f'surfaces/sub-{subject}/freesurfer_lh_middle.ply'), volume = nib.freesurfer.io.read_morph_data(add_output_dir(f'{surf_dir}/sub-{subject}/surf/lh.volume')).astype(float), thickness = nib.freesurfer.io.read_morph_data(add_output_dir(f'{surf_dir}/sub-{subject}/surf/lh.thickness')).astype(float), **kwargs)

    Rmiddle = ((Rwhite[0]+Rgray[0])/2, Rwhite[1])
    kwargs = {}
    for nparcels in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        labels, _, _ = nib.freesurfer.read_annot(add_output_dir(f'{surf_dir}/sub-{subject}/label/rh.Schaefer2018_{nparcels}Parcels_17Networks_order.annot'))
        kwargs[f'original_labels_{nparcels}'] = labels
    surf_to_ply(Rmiddle, add_output_dir(f'surfaces/sub-{subject}/freesurfer_rh_middle.ply'), volume = nib.freesurfer.io.read_morph_data(add_output_dir(f'{surf_dir}/sub-{subject}/surf/rh.volume')).astype(float), thickness = nib.freesurfer.io.read_morph_data(add_output_dir(f'{surf_dir}/sub-{subject}/surf/rh.thickness')).astype(float), **kwargs)
    
    
    # BEM surfaces
    # watershed bem creates both brain and outer_skin from the MRI volume
    # and then inflates the brain to obtain inner_skull
    # and deflates the outer_skin to obtain outer_skull
    # It seems that inner_skull is generally decent, while outer_scalp (and, consequently, outer_skull) can be pretty bad
    # so we check for intersections and inflate outer_skull and outer_scalp having inner_skull as a reference
    
    # NOTE that this fix is not smart, it does not know where the anatomy is
    # it just inflates enough to get numerical stability in the BEM solver, the surfaces will likely still be crooked
    brain = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/bem/brain.surf"))))
    surf_to_ply(brain, add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_brain.ply'))

    inner_skull = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/bem/inner_skull.surf"))))
    surf_to_ply(inner_skull, add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_inner_skull.ply'))

    outer_skull = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/bem/outer_skull.surf"))))
    outer_skull = fix_intersection(inner_skull, outer_skull, min_dist = 4, step_size = 0.1)
    surf_to_ply(outer_skull, add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_outer_skull.ply'))

    outer_skin = to_trimesh(fix_freesurfer_mesh(nib.freesurfer.read_geometry(add_output_dir(f"{surf_dir}/sub-{subject}/bem/outer_skin.surf"))))
    outer_skin = fix_intersection(outer_skull, outer_skin, min_dist = 3, step_size = 0.1)
    surf_to_ply(outer_skin, add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_outer_skin.ply'))
    
    # MNE scalp
    scalp = (np.load(add_output_dir(f"{surf_dir}/sub-{subject}/bem/vertices-scalp.npy")), np.load(add_output_dir(f"{surf_dir}/sub-{subject}/bem/faces-scalp.npy")))
    scalp = fix_freesurfer_mesh(scalp)
    scalp = repair_mesh_topology(scalp)
    surf_to_ply(scalp, add_output_dir(f'surfaces/sub-{subject}/MNE_scalp.ply'))


    # charm surfaces
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/white.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_white.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/gray.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_gray.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/scalp.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_scalp.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/eyes_balls.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_eyes_balls.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/eyes_muscles.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_eyes_muscles.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/CSF.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_CSF.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/bone_compact.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_bone_compact.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/bone_spongy.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_bone_spongy.ply"))
    surf_to_ply(trimesh.load_mesh(add_output_dir(f"simnibscharm/sub-{subject}/converted/blood.stl")), add_output_dir(f"surfaces/sub-{subject}/charm_blood.ply"))


    # hippocampus surfaces
    surf_to_ply(nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_output_dir(f'surfaces/sub-{subject}/hippunfold_L_hipp_middle.ply'), thickness = nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-0p5mm_label-hipp_thickness.shape.gii')).darrays[0].data.astype(float), original_labels = nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-0p5mm_label-hipp_atlas-multihist7_subfields.label.gii')).darrays[0].data)
    surf_to_ply(nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-0p5mm_label-hipp_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_output_dir(f'surfaces/sub-{subject}/hippunfold_R_hipp_middle.ply'), thickness = nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-0p5mm_label-hipp_thickness.shape.gii')).darrays[0].data.astype(float), original_labels = nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-0p5mm_label-hipp_atlas-multihist7_subfields.label.gii')).darrays[0].data)
    
    # dentate gyrus surfaces
    thickness, labels = make_dentate_attributes(*add_output_dir(f'hippunfold/sub-{subject}/LABELS.txt', f'hippunfold/sub-{subject}/anat/sub-{subject}_hemi-L_space-cropT1w_desc-subfields_atlas-multihist7_dseg.nii.gz', f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii'))
    surf_to_ply(nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_output_dir(f'surfaces/sub-{subject}/hippunfold_L_dentate_middle.ply'), thickness = thickness, original_labels = labels)
    thickness, labels = make_dentate_attributes(*add_output_dir(f'hippunfold/sub-{subject}/LABELS.txt', f'hippunfold/sub-{subject}/anat/sub-{subject}_hemi-R_space-cropT1w_desc-subfields_atlas-multihist7_dseg.nii.gz', f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii'))
    surf_to_ply(nib.load(add_output_dir(f'hippunfold/sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-0p5mm_label-dentate_midthickness.surf.gii')).agg_data(('pointset', 'triangle')), add_output_dir(f'surfaces/sub-{subject}/hippunfold_R_dentate_middle.ply'), thickness = thickness, original_labels = labels)

    # cerebellum surfaces
    surf_to_ply(read_vtk(add_output_dir(f'cerebellum/sub-{subject}/nonlinear_Cerebellum_Surf_GM_Labels.vtk')), add_output_dir(f'surfaces/sub-{subject}/cereb_gray.ply'))
    surf_to_ply(read_vtk(add_output_dir(f'cerebellum/sub-{subject}/nonlinear_Cerebellum_Surf_WM_Labels.vtk')), add_output_dir(f'surfaces/sub-{subject}/cereb_white.ply'))
    surf_to_ply(read_vtk(add_output_dir(f'cerebellum/sub-{subject}/nonlinear_Cerebellum_Surf_WM_Labels.vtk')), add_output_dir(f'surfaces/sub-{subject}/cereb_inner.ply'))
    thickness, labels = make_cereb_attributes(add_output_dir(f'cerebellum/sub-{subject}/nonlinear_manifold_Cerebellum_Inner_Surf_With_Features.vtk'))
    surf_to_ply(read_vtk(add_output_dir(f'cerebellum/sub-{subject}/nonlinear_manifold_Cerebellum_Inner_Surf_With_Features.vtk')), add_output_dir(f'surfaces/sub-{subject}/cereb_inner_processed.ply'), thickness = thickness, original_labels = labels, process = False)
