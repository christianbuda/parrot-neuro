import trimesh
import os
import numpy as np
import argparse
import csv
import h5py
import subprocess
import pymeshlab
import sys

# Force line buffering for standard output
sys.stdout.reconfigure(line_buffering=True)

def add_subject_dir(*paths):
    if len(paths)==1:
        return os.path.join(subject_dir, paths[0])
    return tuple([add_subject_dir(x) for x in paths])

def dump_gain(input, output, is_inside = None):
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
    
    np.save(output, leadfield)
    return   

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
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the base directory from the command line
    subject_dir = args.subject_dir
    dipole_spacing = args.dipole_spacing
    
    
    # load BEM meshes
    brain = trimesh.load(add_subject_dir('surfaces/freesurfer_BEM_brain.stl'))
    inner_skull = trimesh.load(add_subject_dir('surfaces/freesurfer_BEM_inner_skull.stl'))
    outer_skull = trimesh.load(add_subject_dir('surfaces/freesurfer_BEM_outer_skull.stl'))
    outer_skin = trimesh.load(add_subject_dir('surfaces/freesurfer_BEM_outer_skin.stl'))
    
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
    dipoles = np.load(add_subject_dir(f'dipoles/spacing{dipole_spacing}mm/dipole_positions.npy'))
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
    with open(add_subject_dir('electrodes/landmarks_10-5-full.csv'), 'r') as f:
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

    dump_gain('head.gain', add_subject_dir(f'forward_solvers/openmeeg-{dipole_spacing}mm-leadfield.npy'), is_inside=is_inside)