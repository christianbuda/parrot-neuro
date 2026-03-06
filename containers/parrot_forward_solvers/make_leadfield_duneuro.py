import duneuropy as dp
import numpy as np
import meshio
import argparse
import os
from scipy.spatial import cKDTree
from mpi4py import MPI
import sys
import json

# Force line buffering for standard output
sys.stdout.reconfigure(line_buffering=True)

def add_subject_dir(*paths):
    if len(paths)==1:
        return os.path.join(subject_dir, paths[0])
    return tuple([add_subject_dir(x) for x in paths])

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

    dipole_element_indices = -np.ones(len(dipoles), dtype = int)

    iter = 0
    max_iter = 50
    ntop = 20
    tol = -1e-6
    remaining_dipoles = np.arange(len(dipoles), dtype = int) # indices of the dipoles to check
    while len(remaining_dipoles)>0:
        if iter > max_iter:
            break

        # Find the indices of the ntop closest tetrahedra to each dipole
        _, candidate_idx = tree.query(dipoles[remaining_dipoles], k=ntop)

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
        remaining_dipoles = remaining_dipoles[found_idx!=1]
        
        # adjust tolerances and neighbours as needed
        if np.any(found_idx>1):
            tol/=3
        if np.any(found_idx==0):
            ntop = int(ntop*1.5)
        iter += 1

    assert np.all(dipole_element_indices>=0), 'Could not find a tetrahedron for each dipole, it is possible that a dipole may lie outside the mesh, check please!'

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

def read_mesh(filename):
    mesh = meshio.read(filename)
    
    # convert units to meters
    points = np.ascontiguousarray(mesh.points.astype(np.dtype('float64'))/1000)
    tetrahedra = np.ascontiguousarray(mesh.cells_dict['tetra'].astype(np.dtype('int64')))
    if filename[-4:] == '.msh':
        labels = np.ascontiguousarray(mesh.cell_data['gmsh:physical'][1].astype(np.dtype('int64')))
    elif filename[-5:] == '.mesh':
        labels = np.ascontiguousarray(mesh.cell_data['medit:ref'][1].astype(np.dtype('int64')))
    else:
        raise ValueError('Wrong mesh input type (this is not a general reader!).')
    
    assert len(tetrahedra) == len(labels), 'Labels don\'t match tetrahedra, check reader!'
    return points, tetrahedra, labels

def read_conductivities(filename):
    with open(filename, 'r') as f:
        cond = f.readlines()
    
    cond = np.array(list(map(lambda x: float(x.split(',')[-1]), cond)))
    
    # Replace any absolute 0.0 conductivity with a tiny, safe number
    cond[np.isclose(cond, 0)] = 1e-6
    return(cond)

def read_tissues(filename):
    with open(filename, 'r') as f:
        names = f.readlines()
    names = list(map(lambda x: x.split(',')[-1].strip().lower(), names))
    return(names)

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

def process_leadfield(leadfield, adjust_volume = True, adjust_density = True, neuronal_strength_dict = None, rereference = True):
    if adjust_volume:
        volume = np.load(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/dipole_volume.npy'))/1e9 # convert from mm3 to m3, not needed though
        volume = np.repeat(volume, 3)
        leadfield = leadfield*volume
        
    if adjust_density:
        neuron_density = np.load(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/dipole_neural_density.npy'))
        neuron_density = np.repeat(neuron_density, 3)
        leadfield = leadfield*neuron_density
        
    if neuronal_strength_dict is not None:
        orient_type = np.load(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/orient_type.npy'))
        orient_type = np.repeat(orient_type, 3)
        assert not np.any(np.isin('U', orient_type)), 'ERROR: some orientation type are Unassigned, something went wrong during dipole generation.'

        labels_to_strength = np.vectorize(neuronal_strength_dict.get)
        dipole_strength = labels_to_strength(orient_type).astype(float)

        leadfield = leadfield*dipole_strength
    
    if rereference:
        leadfield = avg_ref(leadfield)
    
    return leadfield

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Makes leadfield using OpenMEEG",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Define the Subject Folder Argument
    parser.add_argument(
        '--subject_dir',
        type=str,
        required=False,
        default='/subject/', # to be used inside container
        help='Path to the subject folder (e.g., /SUBJECTS/<subjectname>/)'
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
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the base directory from the command line
    subject_dir = args.subject_dir
    dipole_spacing = args.dipole_spacing
    mesh_path = args.mesh_path
    tissue_names = args.tissue_names
    cond_path = args.conductivities_path
    outlabel = args.label
    valid_tissues = args.valid_tissues
    neuronal_strength_dict = args.neuronal_strength_dict
    
    nodes, tetrahedra, tissue_label = read_mesh(add_subject_dir(mesh_path))
    tissue_names = read_tissues(add_subject_dir(tissue_names))
    conductivities = read_conductivities(add_subject_dir(cond_path))
    electrodes = read_electrodes(add_subject_dir('electrodes/landmarks_10-5-full.csv'))
    dipoles = get_dipoles(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/dipole_positions.npy'), nodes, tetrahedra, tissue_label, tissue_names, valid_tissues)

    config = {
        'type' : 'fitted',
        'solver_type' : 'cg',
        'element_type' : 'tetrahedron',
        'volume_conductor' : {
            'grid' : {'nodes' : nodes,
                      'elements' : tetrahedra},
            'tensors' : {'labels' : tissue_label,
                        'conductivities' : conductivities}
        },
        # 'solver' : {'verbose' : 1}
    }

    electrode_config = {
        'type': 'closest_subentity_center',
        'codims': [3]
    }

    tm_config = {'solver' : {'reduction' : 1e-10}}

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
    
    np.save(add_subject_dir(f'forward_solvers/raw_duneuro{outlabel}-{dipole_spacing}mm-leadfield.npy'), leadfield)
    
    print('Processing leadfield...')
    with open(neuronal_strength_dict,'r') as f:
        neuronal_strength_dict = json.load(f)
    
    leadfield = process_leadfield(leadfield, adjust_volume = True, adjust_density = True, neuronal_strength_dict = neuronal_strength_dict, rereference = True)
    np.save(add_subject_dir(f'leadfields/processed_duneuro{outlabel}-{dipole_spacing}mm-leadfield.npy'), leadfield)