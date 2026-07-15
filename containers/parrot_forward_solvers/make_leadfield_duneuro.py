import duneuropy as dp
import numpy as np
import argparse
import os
from scipy.spatial import cKDTree
from mpi4py import MPI
import sys
import json

# Mesh/label-table readers are shared with the anisotropy front-end
# (dti_to_conductivity_tensors.py) so both read the mesh identically.
from mesh_io import read_mesh, read_conductivities, read_tissues

# Force line buffering for standard output
sys.stdout.reconfigure(line_buffering=True)

def add_output_dir(*paths):
    if len(paths)==1:
        return os.path.join(output_dir, paths[0])
    return tuple([add_output_dir(x) for x in paths])

def get_dipole_tissues(dipoles, nodes, elements, labels):
    """
    Finds which tetrahedron each dipole falls into and returns its tissue label.
    """
    # 1. Get the actual (x,y,z) coordinates for all 4 nodes of every tetrahedron
    nodes = nodes[elements]  # Shape: (N_elements, 4, 3)

    # 2. Calculate the center (centroid) of every tetrahedron
    centroids = np.mean(nodes, axis=1)

    # 3. Build a fast KD-Tree using the centroids
    tree = cKDTree(centroids)

    n_elements = len(centroids)
    dipole_element_indices = -np.ones(len(dipoles), dtype = int)

    iter = 0
    max_iter = 50
    ntop = 20
    # A dipole inside a tetrahedron sits among that tet's handful of nearest centroids, so
    # cap the candidate count: querying k > n_elements makes cKDTree return an out-of-bounds
    # sentinel index (-> IndexError), and huge-k queries are pathologically slow. Anything
    # still unmatched at the cap is (marginally) outside every tet -> snapped after the loop.
    ntop_max = min(n_elements, 512)
    tol = -1e-6
    remaining_dipoles = np.arange(len(dipoles), dtype = int) # indices of the dipoles to check
    while len(remaining_dipoles)>0:
        if iter > max_iter:
            break

        # Find the indices of the ntop closest tetrahedra to each dipole (never > n_elements)
        k = min(ntop, ntop_max)
        _, candidate_idx = tree.query(dipoles[remaining_dipoles], k=k)

        ### Test the candidates using Barycentric Coordinates:
        
        # these are the vertices of each candidate tetrahedron for each dipole (len(remaining_dipoles), ntop, 4, 3)
        r_mat = nodes[candidate_idx]
        
        # we take the last vertex of each tetrahedron and we use it to move the frame of reference by setting it at 0
        r4 = r_mat[:,:,-1]
        r_mat = (r_mat-r_mat[:,:,-1:])[:,:,:-1]
        
        # now we build the transform that brings from this new frame of reference to the barycentric frame
        # this is a 3x3 matrix for each dipole and for each candidate tetrahedron
        T_inv = np.linalg.inv(r_mat.transpose((0,1,3,2)))
        
        # these are the barycentric coordinates (a triplet for each dipole and each candidate tetrahedron)
        # obtained by applying the matrices above to the dipole positions (offsetted by the last vertex)
        # shape is (len(remaining_dipoles), ntop, 3)
        beta = np.einsum('ijkl,ijl->ijk', T_inv, (dipoles[remaining_dipoles][:,np.newaxis]-r4))
        
        # A point is strictly inside the tetrahedron if all betas are >= 0 and their sum is <= 1
        # we use a tolerance in case dipoles are very close to the edge of a tetrahedron
        which_inside = np.all(beta>=tol, axis = -1) & (np.sum(beta, axis = -1) <= 1-tol)
        found_idx = np.sum(which_inside, axis = -1)

        # this logic is necessary to remove the dipoles for which we found the tetrahedron
        # and to proceed seemlessly to the next iteration if needed
        current_idx = remaining_dipoles[found_idx==1]
        dipole_element_indices[current_idx] = candidate_idx[found_idx==1][which_inside[found_idx==1]]

        # get the indices of the dipoles that need to be checked better
        n_before = len(remaining_dipoles)
        remaining_dipoles = remaining_dipoles[found_idx!=1]

        # adjust tolerances and neighbours as needed
        if np.any(found_idx>1):
            tol/=3
        if np.any(found_idx==0):
            ntop = int(ntop*1.5)
        # at the candidate cap with no progress -> the rest lie outside the mesh (snapped below)
        if k>=ntop_max and len(remaining_dipoles)==n_before:
            break
        iter += 1

    # Boundary dipoles are sampled on a surface that can sit a fraction of a mm outside the
    # tetrahedral volume, so they fall inside no tet. Snap them to the nearest tetrahedron
    # (as adjust_dipoles does for wrong-tissue dipoles) instead of failing the whole leadfield
    # -- but a dipole grossly outside signals a real placement/mesh problem, so guard by distance.
    if len(remaining_dipoles)>0:
        dist, nearest = tree.query(dipoles[remaining_dipoles], k=1)
        max_off = 0.01  # 10 mm (metres): far beyond any surface/volume boundary mismatch
        if np.any(dist>max_off):
            n_bad = int((dist>max_off).sum())
            raise ValueError(f'{n_bad} dipole(s) lie more than {max_off*1000:.0f} mm outside the '
                             f'mesh (max {dist.max()*1000:.2f} mm): likely a dipole-placement or '
                             f'mesh problem, check please!')
        dipole_element_indices[remaining_dipoles] = nearest
        print(f'Snapped {len(remaining_dipoles)} boundary dipole(s) to the nearest tetrahedron '
              f'(max offset {dist.max()*1000:.3f} mm).', flush=True)

    assert np.all(dipole_element_indices>=0), 'Internal error: dipole left unresolved after snap.'

    dipole_labels = labels[dipole_element_indices]

    return np.array(dipole_element_indices), np.array(dipole_labels)

def adjust_dipoles(dipoles, dipole_labels, valid_tissues, nodes, elements, labels):
    elements = elements[np.isin(labels, valid_tissues)]
    
    # 1. Get the actual (x,y,z) coordinates for all 4 nodes of every valid tetrahedron
    nodes = nodes[elements]

    # 2. Calculate the center (centroid) of every valid tetrahedron
    centroids = np.mean(nodes, axis=1)

    # 3. Build a fast KD-Tree using the centroids
    tree = cKDTree(centroids)
    
    # find the dipoles that need to be moved
    dipoles_to_adjust = np.logical_not(np.isin(dipole_labels, valid_tissues))
    
    # Find the indices of the closest tetrahedra to each dipole
    distances, idx = tree.query(dipoles[dipoles_to_adjust], k=1)
    
    dipoles[dipoles_to_adjust] = centroids[idx]
    dipole_labels[dipoles_to_adjust] = labels[np.isin(labels, valid_tissues)][idx]
    
    print(f'Adjusted positions of {np.count_nonzero(dipoles_to_adjust)} dipoles, maximum distance moved was {max(distances)}.')
    print(f'Dipoles were redistributed as following:')
    for i in valid_tissues:
        print(f'\t{np.count_nonzero(dipole_labels[dipoles_to_adjust]==i)} dipoles in tissue {i}')

    return distances, np.where(dipoles_to_adjust)[0]

def read_electrodes(filename):
    with open(filename, 'r') as f:
        el = f.readlines()
    el = np.array(list(map(lambda x: x.split(',')[1:], el))).astype(float)
    
    # convert units to meters
    el/=1000
    
    el = list(map(lambda x: dp.FieldVector3D(x), el.tolist()))
    return(el)

def convert_dipoles(dipoles):
    # converts dipole array for leadfield computation
    normals = np.concatenate([np.eye(3) for _ in range(len(dipoles))], axis = 0)
    
    dipoles = np.repeat(dipoles, 3, axis = 0)
    
    dipoles = np.concatenate([dipoles, normals], axis = 1)
    
    dipoles = list(map(lambda x: dp.Dipole3d(x[:3], x[3:]), dipoles.tolist()))
    return dipoles

def get_dipoles(dipoles_path, nodes, tetrahedra, tissue_label, tissue_names, valid_tissues_names):
    # convert names to labels
    valid_tissues = []
    for name in [n.lower() for n in valid_tissues_names]:
        try:
            valid_tissues.append(tissue_names.index(name))
        except ValueError:
            raise ValueError(f'List of valid tissues provided {valid_tissues_names} is not compatible with tissue names provided:\n{tissue_names}.\nIn particular, could not find {name} inside names list.')
    
    # load and convert to meters
    dipoles = np.load(dipoles_path)/1000

    # find out the label assigned to each dipole
    _, dipole_labels = get_dipole_tissues(dipoles, nodes, tetrahedra, tissue_label)

    # check whether all dipoles are inside the correct tissue
    # if not move them
    if np.any(np.logical_not(np.isin(dipole_labels, valid_tissues))):
        print(f'Found dipoles outside valid tissues {valid_tissues}:')
        unique, counts = np.unique(dipole_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Tissue Label {u}: contains {c} dipoles")
        print(f'The dipoles outside tissues {valid_tissues} will be moved.')
        # adjust works inplace on dipoles and dipole_labels
        adjust_dipoles(dipoles, dipole_labels, valid_tissues, nodes, tetrahedra, tissue_label)
        
    # Print the diagnostics
    print("\n--- Dipole Location Diagnostics ---")
    unique, counts = np.unique(dipole_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Tissue Label {u}: contains {c} dipoles")

    dipoles = convert_dipoles(dipoles)
    
    return dipoles

def avg_ref(mat):
    # manually average reference forward solution
    nEl = mat.shape[0]
    avg_ref_op = -np.ones((nEl,nEl))/nEl
    avg_ref_op[np.diag_indices_from(avg_ref_op)] = 1-1/nEl

    return np.dot(avg_ref_op, mat)

# Fraction of all-zero source columns above which the processed leadfield is
# rejected as corrupt (see the backstop check at the end of process_leadfield).
DEAD_SOURCE_FAIL_FRAC = 0.05


def process_leadfield(leadfield, adjust_volume = True, adjust_density = True, neuronal_strength_dict = None, rereference = True):
    if adjust_volume:
        volume = np.load(add_output_dir(f'dipoles/sub-{subject}/spacing{dipole_spacing}mm/dipole_volume.npy'))/1e9 # convert from mm3 to m3, not needed though
        volume = np.repeat(volume, 3)
        leadfield = leadfield*volume
        
    if adjust_density:
        neuron_density = np.load(add_output_dir(f'dipoles/sub-{subject}/spacing{dipole_spacing}mm/dipole_neural_density.npy'))
        neuron_density = np.repeat(neuron_density, 3)
        leadfield = leadfield*neuron_density
        
    if neuronal_strength_dict is not None:
        orient_type = np.load(add_output_dir(f'dipoles/sub-{subject}/spacing{dipole_spacing}mm/orient_type.npy'))
        orient_type = np.repeat(orient_type, 3)
        assert not np.any(np.isin('U', orient_type)), 'ERROR: some orientation type are Unassigned, something went wrong during dipole generation.'

        labels_to_strength = np.vectorize(neuronal_strength_dict.get)
        dipole_strength = labels_to_strength(orient_type).astype(float)

        leadfield = leadfield*dipole_strength
    
    if rereference:
        leadfield = avg_ref(leadfield)

    # Backstop: a healthy raw leadfield has no all-zero source columns, but the
    # per-dipole weighting above (notably the BigBrain-derived neural density) can
    # zero a column if a factor is zero -- which is how a silently mis-registered
    # BigBrain warp deletes whole cortical regions. A few boundary sources at ~0 is
    # normal (<0.25% across the cohort); a percent-scale block is a data fault, so we
    # fail rather than emit a quietly corrupt leadfield. The bigbrain stage's coverage
    # gate should catch this upstream; this is defense-in-depth for any other cause.
    n_dip = leadfield.shape[1] // 3
    per_dip = np.sqrt((leadfield.reshape(leadfield.shape[0], n_dip, 3) ** 2).sum(axis=(0, 2)))
    frac_dead = (per_dip == 0).sum() / n_dip
    if frac_dead > DEAD_SOURCE_FAIL_FRAC:
        raise ValueError(
            f'{(per_dip == 0).sum()}/{n_dip} ({frac_dead*100:.1f}%) leadfield source '
            f'columns are all-zero (> {DEAD_SOURCE_FAIL_FRAC*100:.0f}% threshold). This '
            'signals corrupt per-dipole weighting -- most often a failed BigBrain warp '
            'zeroing the neural density (check the bigbrain stage / QC coverage).')

    return leadfield

def build_conductivity_config(tissue_label, conductivities, dti_tensors_path=None):
    """Build the duneuro `volume_conductor['tensors']` sub-dict.

    Returns the *conductor* labeling (which must NOT be used for dipole
    placement -- that uses the original per-element tissue_label).

    Without anisotropy: the original per-label scalar path -- duneuro builds
    value*I internally. Byte-identical to the historical behaviour.

    With anisotropy (--dti_tensors_path): a hybrid per-element tensor path.
    Non-WM tissues keep their shared tissue label (one isotropic sigma*I each);
    every WM tetrahedron gets a *unique* label pointing at its own 3x3
    conductivity tensor (from dti_to_conductivity_tensors.py). So the tensor
    list has length n_tissues + n_WM_tets rather than one entry per element --
    far smaller than a full per-element list. duneuro indexes tensors by label
    (`tensors[labels[element]]`), reading each as a full 3x3 (no Voigt packing).
    """
    if dti_tensors_path is None:
        # scalar per-label; duneuro forms sigma*I internally
        return {'labels': tissue_label, 'conductivities': conductivities}

    # WM-only conductivity tensors + their element indices (written side by side)
    wm_tensors = np.load(dti_tensors_path)                                  # (M, 3, 3)
    wm_indices = np.load(os.path.join(os.path.dirname(dti_tensors_path),
                                      'wm_element_indices.npy'))            # (M,)
    assert len(wm_tensors) == len(wm_indices), 'tensor/index length mismatch'

    n_tissues = len(conductivities)
    # one isotropic 3x3 per tissue label, then one anisotropic 3x3 per WM tet
    tensor_list = [float(c) * np.eye(3) for c in conductivities]
    tensor_list.extend(np.asarray(wm_tensors, dtype=np.float64))

    # conductor labels: copy tissue labels, then give WM tets unique labels
    # that index into the appended tensors (n_tissues + 0 .. n_tissues + M-1).
    conductor_labels = tissue_label.copy()
    conductor_labels[wm_indices] = n_tissues + np.arange(len(wm_indices))

    print(f"Anisotropy ON: {len(wm_indices)} WM tetrahedra get per-element tensors "
          f"({len(tensor_list)} total tensors = {n_tissues} tissues + {len(wm_indices)} WM).")
    return {'labels': conductor_labels, 'tensors': tensor_list}

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Makes leadfield using OpenMEEG",
        formatter_class=argparse.RawTextHelpFormatter
    )

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
        help='Path to the derivatives folder (e.g., /derivatives/)'
    )

    parser.add_argument(
        '--dipole_spacing',
        type=str,
        required=True,
        help='Spacing between dipoles, in mm (typical values range from 1 to 10). Dipoles are not sampled here, this script expects that results are available in subject folder for specified spacing.'
    )
    
    parser.add_argument(
        '--mesh_path',
        type=str,
        required=True,
        help='Path to tetrahedral mesh relative to subject directory (points should be expressed in mm).'
    )
    
    parser.add_argument(
        '--tissue_names',
        type=str,
        required=True,
        help='Path to names file for the tissues, relative to subject directory.'
    )
    
    parser.add_argument(
        '--conductivities_path',
        type=str,
        required=True,
        help='Path to conductivities file relative to subject directory (values are expected in S/m).'
    )
    
    parser.add_argument(
        '--label',
        type=str,
        required=False,
        default='',
        help='Label to give to output (i.e. the output will be saved as duneuro<{outlabel>-<dipole_spacing>mm-leadfield.npy, default is empty).'
    )

    parser.add_argument(
        '--valid_tissues',
        nargs='+',
        type=str,
        required=True,
        help='Name of the tissues that can contain electrical generators (typically just the gray matter). This must be a list of integer separated by a space (e.g. "--valid_tissues Gray_matter White_matter").'
    )
    
    parser.add_argument(
        '--neuronal_strength_dict',
        type=str,
        required=False,
        default='./neuronal_strength_dict.json',
        help="Dictionary that maps dipole orientation type ('N', 'G', 'P', 'R') to base strength. Default values are an heuristics from Murakami and Okada (2006)."
    )

    parser.add_argument(
        '--dti_tensors_path',
        type=str,
        required=False,
        default=None,
        help='Optional path to per-WM-tet conductivity tensors (anisotropy/.../conductivity_tensors.npy from dti_to_conductivity_tensors.py; wm_element_indices.npy must sit beside it). When given, white matter uses anisotropic per-element tensors; when omitted, behaviour is identical to the isotropic solver.'
    )

    parser.add_argument(
        '--threads',
        type=int,
        required=False,
        default=0,
        help='Threads for the (TBB-parallel) transfer-matrix solve. Each thread holds its own solver/AMG hierarchy, so on many-core machines the default (0 = all cores) can exhaust memory on large meshes. Set a modest number (e.g. 32). 0 keeps DUNEuro\'s automatic (all-cores) behaviour.'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    subject = args.subject
    output_dir = args.output_dir
    dipole_spacing = args.dipole_spacing
    mesh_path = args.mesh_path
    tissue_names = args.tissue_names
    cond_path = args.conductivities_path
    outlabel = args.label
    valid_tissues = args.valid_tissues
    neuronal_strength_dict = args.neuronal_strength_dict
    dti_tensors_path = args.dti_tensors_path

    nodes, tetrahedra, tissue_label = read_mesh(mesh_path)
    tissue_names = read_tissues(tissue_names)
    conductivities = read_conductivities(cond_path)
    electrodes = read_electrodes(add_output_dir(f'electrodes/sub-{subject}/landmarks_10-5-full.csv'))
    # NOTE: dipole placement classifies dipoles by ORIGINAL tissue identity, so it
    # must receive the unmodified per-element tissue_label -- never the per-element
    # conductor labeling built below. Keeping White-Matter out of valid_tissues is
    # what guarantees no dipole lands in an anisotropic WM tet (Venant-safe).
    dipoles = get_dipoles(add_output_dir(f'dipoles/sub-{subject}/spacing{dipole_spacing}mm/dipole_positions.npy'), nodes, tetrahedra, tissue_label, tissue_names, valid_tissues)

    config = {
        'type' : 'fitted',
        'solver_type' : 'cg',
        'element_type' : 'tetrahedron',
        'volume_conductor' : {
            'grid' : {'nodes' : nodes,
                      'elements' : tetrahedra},
            'tensors' : build_conductivity_config(tissue_label, conductivities, dti_tensors_path)
        },
        # 'solver' : {'verbose' : 1}
    }

    electrode_config = {
        'type': 'closest_subentity_center',
        'codims': [3]
    }

    # numberOfThreads is read at the TOP level of tm_config (only 'reduction' lives
    # under 'solver'); omitting it makes DUNEuro use all cores, which can blow up
    # memory (one solver/AMG hierarchy per thread) on large meshes + many cores.
    tm_config = {'solver' : {'reduction' : 1e-10}}
    if args.threads and args.threads > 0:
        tm_config['numberOfThreads'] = args.threads

    source_model_config = {
        # Average-referencing the EEG is standard practice
        'subtract_mean': True, 
        'post_process': False, # Only needed for the Subtraction approach
        
        'source_model': {
            'type': 'venant',
            'restrict': True,             # Keeps monopoles strictly inside their own compartment
            'numberOfMoments': 3,         # High accuracy
            'weightingExponent': 1,       # Standard regularization
            'relaxationFactor': 1e-6,     # Standard stability factor
            'initialization': 'closest_vertex', 
            'referenceLength': 0.02,      # 20 mm converted to METERS!
            'mixedMoments': True          # Standard for 2nd order moments
        }
    }

    print('Initializing DUNEuro driver...')
    driver = dp.MEEGDriver3d(config)
    
    print('Setting electrodes...')
    driver.setElectrodes(electrodes, electrode_config)
    
    print('Computing transfer matrix...')
    transfer_matrix, transfer_info = driver.computeEEGTransferMatrix(tm_config)

    print('Computing leadfield...')
    leadfield, info = driver.applyEEGTransfer(np.array(transfer_matrix), dipoles, source_model_config)
    
    # transpose to make it (number_electrodes, number_dipoles)
    leadfield = np.array(leadfield).T
    
    np.save(add_output_dir(f'forwardsolvers/sub-{subject}/raw_duneuro{outlabel}-{dipole_spacing}mm-leadfield.npy'), leadfield)
    
    print('Processing leadfield...')
    with open(neuronal_strength_dict,'r') as f:
        neuronal_strength_dict = json.load(f)
    
    leadfield = process_leadfield(leadfield, adjust_volume = True, adjust_density = True, neuronal_strength_dict = neuronal_strength_dict, rereference = True)
    np.save(add_output_dir(f'leadfields/sub-{subject}/processed_duneuro{outlabel}-{dipole_spacing}mm-leadfield.npy'), leadfield)