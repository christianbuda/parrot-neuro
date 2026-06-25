import trimesh
import os
import numpy as np
import argparse
import csv
import h5py
import subprocess
import pymeshlab
import sys
import json

# Force line buffering for standard output
sys.stdout.reconfigure(line_buffering=True)

def add_output_dir(*paths):
    if len(paths)==1:
        return os.path.join(output_dir, paths[0])
    return tuple([add_output_dir(x) for x in paths])

def convert_gain(input, is_inside = None):
    with h5py.File(input, 'r') as f:
        # Load openmeeg leadfield (assuming the key is 'linop')
        # Note: MATLAB saves matrices transposed compared to Python/C order,
        # so we usually need to transpose it back (.T)
        leadfield = np.array(f['linop']).T
    
    # here we insert zero in places where dipoles were not inside the brain domain
    if is_inside is not None:
        complete_leadfield = np.zeros_like(leadfield)
        complete_leadfield[:,is_inside] = leadfield
        leadfield = complete_leadfield
    
    return leadfield

def write_electrodes(filename, electrodes, names):
    with open(filename, 'w') as f:
        for idx in range(len(electrodes)):
            f.write(f"{names[idx]} {electrodes[idx, 0]:.6f} {electrodes[idx, 1]:.6f} {electrodes[idx, 2]:.6f}\n")
    return

def write_dipoles(filename, dipoles, normals):
    with open(filename, 'w') as f:
        for idx in range(len(dipoles)):
            f.write(f"{dipoles[idx, 0]:.6f} {dipoles[idx, 1]:.6f} {dipoles[idx, 2]:.6f} {normals[idx, 0]:.6f} {normals[idx, 1]:.6f} {normals[idx, 2]:.6f}\n")
    return

def write_brainvisa_tri(filename, mesh):
    """
    Write a surface mesh to BrainVISA-compatible .tri format.

    Parameters:
    - filename: str, output path for .tri file
    - mesh: trimesh mesh
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    faces = mesh.faces
    
    with open(filename, 'w') as f:
        f.write(f"- {len(vertices)}\n")
        for idx, v in enumerate(vertices):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {normals[idx][0]:.6f} {normals[idx][1]:.6f} {normals[idx][2]:.6f}\n")
        f.write(f"- {len(faces)} {len(faces)} {len(faces)}\n")
        for t in faces:
            f.write(f"{t[0]} {t[1]} {t[2]}\n")
    return

def decimate_mesh(mesh, target_faces):
    # --- 1. Pass Trimesh data to PyMeshLab ---
    # We extract the numpy arrays from Trimesh and build a PyMeshLab Mesh
    m = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
    
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)
    
    # --- 2. Run the Decimation ---
    # Using the safe parameters from the PyMeshLab docs you provided
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preservetopology=True,  # Crucial for BEM: keeps the mesh watertight
        preservenormal=True,    # Crucial for BEM: stops normals from flipping inward
        autoclean=True          # Cleans up unreferenced vertices/bad faces after
    )
    
    # --- 3. Extract back to Trimesh ---
    out_mesh = ms.current_mesh()
    
    decimated_mesh = trimesh.Trimesh(
        vertices=out_mesh.vertex_matrix(), 
        faces=out_mesh.face_matrix()
    )
    
    return decimated_mesh

def run_openmeeg_pipeline():
    # Define the commands as lists of strings (best practice for subprocess)
    commands = [
        ["om_assemble", "-HM", "geometry.geom", "conductivities.cond", "head.hm"],
        ["om_assemble", "-DSM", "geometry.geom", "conductivities.cond", "dipoles.txt", "head.dsm", "BRAIN"],
        ["om_assemble", "-h2em", "geometry.geom", "conductivities.cond", "electrodes.txt", "head.h2em"],
        ["om_minverser", "head.hm", "head.hm_inv"],
        ["om_gain", "-EEG", "head.hm_inv", "head.dsm", "head.h2em", "head.gain"]
    ]

    for cmd in commands:
        # Reconstruct the command string just for the print output
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}")
        
        try:
            # check=True will automatically raise an exception if the command fails
            subprocess.run(cmd, check=True)
            print("Status: Success\n")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed at command: {cmd_str}")
            print(f"Error details: {e}", file=sys.stderr)
            # Exit the script if a step fails so subsequent commands don't run on bad data
            sys.exit(1)
            
    print("OpenMEEG pipeline completed successfully!")

def avg_ref(mat):
    # manually average reference forward solution
    nEl = mat.shape[0]
    avg_ref_op = -np.ones((nEl,nEl))/nEl
    avg_ref_op[np.diag_indices_from(avg_ref_op)] = 1-1/nEl

    return np.dot(avg_ref_op, mat)

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
    
    return leadfield

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
        '--neuronal_strength_dict',
        type=str,
        required=False,
        default='./neuronal_strength_dict.json',
        help="Dictionary that maps dipole orientation type ('N', 'G', 'P', 'R') to base strength. Default values are an heuristics from Murakami and Okada (2006)."
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    subject = args.subject
    output_dir = args.output_dir
    dipole_spacing = args.dipole_spacing
    neuronal_strength_dict = args.neuronal_strength_dict
    
    
    # load BEM meshes
    brain = trimesh.load(add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_brain.ply'))
    inner_skull = trimesh.load(add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_inner_skull.ply'))
    outer_skull = trimesh.load(add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_outer_skull.ply'))
    outer_skin = trimesh.load(add_output_dir(f'surfaces/sub-{subject}/freesurfer_BEM_outer_skin.ply'))
    
    # convert meshes to meters
    brain.vertices = brain.vertices/1000
    inner_skull.vertices = inner_skull.vertices/1000
    outer_skull.vertices = outer_skull.vertices/1000
    outer_skin.vertices = outer_skin.vertices/1000

    # decimate the meshes to avoid computational issues with openmeeg
    brain = decimate_mesh(brain, target_faces=6000)
    inner_skull = decimate_mesh(inner_skull, target_faces=6000)
    outer_skull = decimate_mesh(outer_skull, target_faces=6000)
    outer_skin = decimate_mesh(outer_skin, target_faces=6000)

    # dump BEM surfaces
    write_brainvisa_tri('brain.tri', brain)
    write_brainvisa_tri('inner_skull.tri', inner_skull)
    write_brainvisa_tri('outer_skull.tri', outer_skull)
    write_brainvisa_tri('scalp.tri', outer_skin)

    # dump BEM dipoles
    dipoles = np.load(add_output_dir(f'dipoles/sub-{subject}/spacing{dipole_spacing}mm/dipole_positions.npy'))
    # convert to meters
    dipoles /= 1000
    
    # create normals as orthogonal triplet
    normals = np.concatenate([np.eye(3) for _ in range(len(dipoles))], axis = 0)
    dipoles = np.repeat(dipoles, 3, axis = 0)
    
    # returns a binary mask of all the dipoles inside the brain domain
    is_inside = brain.contains(dipoles)
    if not np.all(is_inside):
        print('WARNING: found dipoles outside brain domain, removing them temporarily from the leadfield...')
    
    write_dipoles('dipoles.txt', dipoles[is_inside], normals)

    # load electrodes positions
    with open(add_output_dir(f'electrodes/sub-{subject}/landmarks_10-5-full.csv'), 'r') as f:
        reader = csv.reader(f)
        electrodes = np.array(list(reader))
        names = electrodes[:,0]
        
        # convert to meters
        electrodes = electrodes[:,1:].astype(float)/1000

    # project electrodes on scalp
    projected_electrodes, _, _ = trimesh.proximity.closest_point(outer_skin, electrodes)

    # write electrodes
    write_electrodes('electrodes.txt', projected_electrodes, names)

    # run openmeeg
    # this script uses a four layer model with conductivities taken from brainstorm.
    # BRAIN   0.33
    # CSF     1.79
    # SKULL   0.0041
    # SCALP   0.33
    # AIR     0.0
    run_openmeeg_pipeline()

    leadfield = convert_gain('head.gain', is_inside=is_inside)
    np.save(add_output_dir(f'forwardsolvers/sub-{subject}/raw_openmeeg-{dipole_spacing}mm-leadfield.npy'), leadfield)
    
    print('Processing leadfield...')
    with open(neuronal_strength_dict,'r') as f:
        neuronal_strength_dict = json.load(f)
    
    leadfield = process_leadfield(leadfield, adjust_volume = True, adjust_density = True, neuronal_strength_dict = neuronal_strength_dict, rereference = True)
    np.save(add_output_dir(f'leadfields/sub-{subject}/processed_openmeeg-{dipole_spacing}mm-leadfield.npy'), leadfield)