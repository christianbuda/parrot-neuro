from electrodes_positions.utils.point_picking import project_fid_on_mesh, select_feasible_positions
from electrodes_positions.montages import create_standard_montage
import json
import trimesh
import os
import numpy as np
import argparse

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Places electrodes on subject's head.",
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

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the base directory and dipole spacing from the command line
    subject_dir = args.subject_dir
    
    # make output directory if needed
    os.makedirs(os.path.join(subject_dir, 'electrodes/'), exist_ok=True)
    
    mesh = trimesh.load(os.path.join(subject_dir, 'surfaces/charm_scalp.stl'))
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    with open(os.path.join(subject_dir, 'scalp_landmarks/fiducials.json'), 'r') as f:
        fiducials = json.load(f)

    points = [fiducials['RPA'], fiducials['LPA'], fiducials['NAS'], fiducials['IN']]

    # project the fiducials on the mesh vertices to get fid indices
    (RPA, LPA, NAS, IN), (RPA_idx, LPA_idx, NAS_idx, IN_idx) = project_fid_on_mesh(points, vertices, return_positions = True, return_indices=True)

    # place electrodes
    newverts, newfac, all_landmarks = create_standard_montage(vertices, faces, fiducials = (RPA_idx, LPA_idx, NAS_idx, IN_idx), system = '10-5-full', return_indices = True)

    if os.path.isfile(os.path.join(subject_dir, 'scalp_landmarks/outlines.npy')):
        outlines = np.load(os.path.join(subject_dir, 'scalp_landmarks/outlines.npy'))
        selected_landmarks = select_feasible_positions(newverts, newfac, outlines = outlines, landmarks = all_landmarks, positions = None, project_outlines = True)
    else:
        selected_landmarks = all_landmarks


    with open(os.path.join(subject_dir, 'electrodes/landmarks_10-5-full.csv'), 'w') as f:
        for key, val in all_landmarks.items():
            f.write(f'{key}, {newverts[val][0]}, {newverts[val][1]}, {newverts[val][2]}\n')

    with open(os.path.join(subject_dir, 'electrodes/selected_landmarks_10-5-full.json'), 'w') as f:
        json.dump([key for key in all_landmarks.keys() if key in selected_landmarks.keys()], f)
