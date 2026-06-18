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
    parser.add_argument('--volume', type=str, default='T1',
                        help="mri/<volume>.mgz that watershed segments the skull from. "
                             "Default T1; pass INV2 for MP2RAGE, whose UNI/MPRAGEised T1 "
                             "lacks the extracranial contrast mri_watershed needs.")

    args = parser.parse_args()
    subject = args.subject
    subjects_dir = args.subjects_dir

    # Skull surfaces (inner/outer skull, outer skin) via watershed on the chosen volume.
    bem.make_watershed_bem(subject = subject, subjects_dir = subjects_dir, volume = args.volume, overwrite=True)
    # Dense scalp stays on T1.mgz (mkheadsurf): the head/air boundary is clean there.
    bem.make_scalp_surfaces(subject = subject, subjects_dir = subjects_dir, force=True, overwrite=True)

    # save scalp mesh arrays
    vertices, faces = read_fif(os.path.join(subjects_dir, subject, f'bem/{subject}-head-dense.fif'))
    np.save(os.path.join(subjects_dir, subject, f'bem/vertices-scalp.npy'), vertices)
    np.save(os.path.join(subjects_dir, subject, f'bem/faces-scalp.npy'), faces)
