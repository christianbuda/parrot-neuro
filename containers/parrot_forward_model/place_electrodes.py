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

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the base directory and dipole spacing from the command line
    subject = args.subject
    output_dir = args.output_dir
    
    # make output directory if needed
    os.makedirs(os.path.join(output_dir, f'electrodes/sub-{subject}/'), exist_ok=True)
    
    mesh = trimesh.load(os.path.join(output_dir, f'surfaces/sub-{subject}/charm_scalp.ply'))
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    if os.path.isfile(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/fiducials.json')):
        with open(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/fiducials.json'), 'r') as f:
            fiducials = json.load(f)
    else:
        # take simnibs fiducials and dump them in electrodes folder
        fiducials = np.loadtxt(os.path.join(output_dir, f'simnibscharm/sub-{subject}/eeg_positions/Fiducials.csv'), delimiter=',', dtype=str)
        names = fiducials[:,-1].tolist()
        points = fiducials[:,1:-1].astype(float).tolist()
        
        # project the fiducials on the mesh vertices to get fid indices
        points, _ = project_fid_on_mesh(points, vertices, return_positions = True, return_indices=True)
    
        fiducials = dict(zip(names, points))
        if 'Nz' in fiducials.keys():
            fiducials['NAS'] = fiducials.pop('Nz')
        if 'Iz' in fiducials.keys():
            fiducials['IN'] = fiducials.pop('Iz')
        
        os.makedirs(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/'), exist_ok=True)
        with open(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/fiducials.json'), 'w') as f:
            # project_fid_on_mesh returns numpy arrays; cast to lists so json can serialise them
            json.dump({k: np.asarray(v).tolist() for k, v in fiducials.items()}, f)

    points = [fiducials['RPA'], fiducials['LPA'], fiducials['NAS'], fiducials['IN']]

    # project the fiducials on the mesh vertices to get fid indices
    (RPA, LPA, NAS, IN), (RPA_idx, LPA_idx, NAS_idx, IN_idx) = project_fid_on_mesh(points, vertices, return_positions = True, return_indices=True)

    # place electrodes
    newverts, newfac, all_landmarks = create_standard_montage(vertices, faces, fiducials = (RPA_idx, LPA_idx, NAS_idx, IN_idx), system = '10-5-full', return_indices = True)

    if os.path.isfile(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/outlines.npy')):
        outlines = np.load(os.path.join(output_dir, f'scalplandmarks/sub-{subject}/outlines.npy'))
        selected_landmarks = select_feasible_positions(newverts, newfac, outlines = outlines, landmarks = all_landmarks, positions = None, project_outlines = True)
    else:
        selected_landmarks = all_landmarks


    with open(os.path.join(output_dir, f'electrodes/sub-{subject}/landmarks_10-5-full.csv'), 'w') as f:
        for key, val in all_landmarks.items():
            f.write(f'{key}, {newverts[val][0]}, {newverts[val][1]}, {newverts[val][2]}\n')

    with open(os.path.join(output_dir, f'electrodes/sub-{subject}/selected_landmarks_10-5-full.json'), 'w') as f:
        json.dump([key for key in all_landmarks.keys() if key in selected_landmarks.keys()], f)
