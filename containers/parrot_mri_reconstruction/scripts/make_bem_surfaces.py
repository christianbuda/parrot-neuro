import numpy as np
import argparse
import os
import mne
from mne import bem

def read_fif(fname):
    # Load the FIF once using MNE
    surfaces = mne.read_bem_surfaces(fname, on_defects= 'ignore')
    vertices = surfaces[0]['rr'] * 1000.0 # Convert to mm!
    faces = surfaces[0]['tris']
    return vertices, faces
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make BEM surfaces using the watershed algorithm and MNE.")
    parser.add_argument('--subject', type=str, required = True, help='Subject name for the reconstruction.')
    parser.add_argument('--subjects_dir', type=str, required = True, help='Path to freesurfer subjects directory.')
    
    args = parser.parse_args()
    subject = args.subject
    subjects_dir = args.subjects_dir
    

    bem.make_watershed_bem(subject = subject, subjects_dir = subjects_dir, overwrite=True)
    bem.make_scalp_surfaces(subject = subject, subjects_dir = subjects_dir, force=True, overwrite=True)

    # save scalp mesh arrays
    vertices, faces = read_fif(os.path.join(subjects_dir, subject, f'bem/{subject}-head-dense.fif'))
    np.save(os.path.join(subjects_dir, subject, f'bem/vertices-scalp.npy'), vertices)
    np.save(os.path.join(subjects_dir, subject, f'bem/faces-scalp.npy'), faces)
