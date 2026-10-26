from electrodes_positions.utils.point_picking import project_fid_on_mesh, select_feasible_positions
from electrodes_positions.montages import create_standard_montage, get_upper_path
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
    
    # Fiducials come from the `fiducials` stage (make_fiducials.py), which warps corrected
    # MNI coordinates into the subject. Deriving them here from the SimNIBS CSV is gone: those
    # template points are ~7.6 mm posterior of the preauricular point, which lands them on the
    # pinna and tilts the montage.
    fid_path = os.path.join(output_dir, f'scalplandmarks/sub-{subject}/fiducials.json')
    if not os.path.isfile(fid_path):
        raise SystemExit(f'No fiducials at {fid_path}. Run the `fiducials` stage '
                         '(make_fiducials.py, parrot_mri_reconstruction) first, or place them by hand.')
    with open(fid_path, 'r') as f:
        fiducials = json.load(f)

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

    # Placement diagnostic. Written last and never fatal: the montage outputs above must not
    # depend on it. Cz is the arc-length midpoint of the coronal cut, so anything that inflates
    # the arc on one side (the cut wrapping the pinna) biases Cz and tilts the whole montage.
    # Tortuosity (arc/chord) over the last 30 mm before each ear endpoint measures that directly
    # -- it is ~1.00 on smooth scalp for every head, so unlike left/right asymmetry it has no
    # natural-variation background to hide in.
    try:
        Cz_pos = newverts[all_landmarks['Cz']]
        cut, _ = get_upper_path(newverts, newfac,
                                np.cross(newverts[RPA_idx] - Cz_pos, newverts[LPA_idx] - Cz_pos),
                                start_point=RPA_idx, end_point=LPA_idx)
        tort = {}
        for end, name in ((cut, 'RPA'), (cut[::-1], 'LPA')):
            step = np.linalg.norm(np.diff(end, axis=0), axis=-1)
            cum = np.concatenate([[0.0], np.cumsum(step)])
            k = int(np.searchsorted(cum, 30.0))
            chord = np.linalg.norm(end[k] - end[0])
            tort[name] = float(cum[k] / chord) if chord > 1e-9 else float('nan')
        with open(os.path.join(output_dir, f'electrodes/sub-{subject}/placement_qc.json'), 'w') as f:
            json.dump({'ear_approach_tortuosity': tort,
                       'max_tortuosity': float(max(tort.values())),
                       'window_mm': 30.0}, f)
    except BaseException as e:
        # get_upper_path raises BaseException, so Exception would not catch it.
        print(f'WARNING: placement diagnostic failed ({type(e).__name__}: {e}); '
              'electrode positions are unaffected')
