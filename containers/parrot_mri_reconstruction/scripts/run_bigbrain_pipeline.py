import os
import argparse
import numpy as np
import pandas as pd


def run_multistage_registration(fixed, moving, outprefix):
    """
    Runs Translation + Rigid + Similarity + Affine + SyN registration with Masks using fixed parameters
    """
    
    # 1. Get memory pointers for images
    f_ptr = ants.internal.get_pointer_string(fixed)
    m_ptr = ants.internal.get_pointer_string(moving)

    # 3. Build the argument list
    args = [
        "--dimensionality", str(fixed.dimension),
        "--verbose", "1",
        "--float", "1",
        "--output", outprefix,
    ]

    # --- STAGE 1: TRANSLATION ---
    args.extend([
        "--initial-moving-transform", f"[{f_ptr},{m_ptr},0]",
        "--transform", "Translation[1]",
        "--metric", f"mattes[{f_ptr},{m_ptr},1,32,None]",
        "--convergence", "[10000x10000x0x0,1.e-8,10]",
        "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "4x2x1x0",
        "--use-histogram-matching", "1"
    ])

    # --- STAGE 2: RIGID ---
    args.extend([
        "--transform", "Rigid[1]",
        "--metric", f"mattes[{f_ptr},{m_ptr},1,32,None]",
        "--convergence", "[10000x10000x0x0,1.e-8,10]",
        "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "4x2x1x0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1"
    ])

    # --- STAGE 3: SIMILARITY ---
    args.extend([
        "--transform", "Similarity[1]",
        "--metric", f"mattes[{f_ptr},{m_ptr},1,32,None]",
        "--convergence", "[10000x10000x1500x20,1.e-8,10]",
        "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "4x2x1x0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1"
    ])

    # --- STAGE 4: AFFINE ---
    args.extend([
        "--transform", "Affine[1]",
        "--metric", f"mattes[{f_ptr},{m_ptr},1,32,None]",
        "--convergence", "[10000x10000x1500x20,1.e-8,20]",
        "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "4x2x1x0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1"
    ])

    # --- STAGE 5: DEFORMABLE (SyN) ---
    args.extend([
        "--transform", "SyN[0.2,3,0]",
        "--metric", f"CC[{f_ptr},{m_ptr},1,4]",
        "--convergence", "[200x200x200x200,1e-8,8]",
        "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "3x2x1x0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1"
    ])

    # 4. Execute
    print(f"Running antsRegistration... Output: {outprefix}")
    print(f"\n\nEquivalent command in the CLI is:\nantsRegistration {' '.join(args)}\n\n")
    exit_code = ants.registration(fixed=args, moving=None)
    
    if exit_code == 0:
        print("Success.")
    else:
        raise RuntimeError(f"Registration failed with error code: {exit_code}")
    

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Register subject data on the bigbrain template brain.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 1. Define the Subject Folder Argument
    parser.add_argument(
        '--subject_dir', 
        type=str,
        required=True,
        help='Path to the subject folder (e.g., /SUBJECTS/<subjectname>/)'
    )

    # 2. Define the Template Folder Argument
    parser.add_argument(
        '--template_dir', 
        type=str,
        required=True,
        help='Path to the bigbrain scans folder (e.g., /home/bigbrain_scans/)'
    )

    # 3. Get number of threads
    parser.add_argument(
        '--threads',
        type=str,
        required=True,
        help='Number of threads to use during ants registration'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # set ants number of threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = args.threads
    subject_folder = args.subject_dir
    bigbrain_scans_folder = args.template_dir

    ################## LOAD FILES ########################
    import ants
    
    # load brain stripped T1
    subject_brain = ants.image_read(os.path.join(subject_folder,'synthstrip/T1_stripped.nii.gz'))

    # apply bias field correction and reapply brain mask
    subject_brain = ants.n4_bias_field_correction(subject_brain)
    subject_brain = subject_brain * ants.image_read(os.path.join(subject_folder,'synthstrip/T1_stripped_mask.nii.gz'))

    # save bias field corrected brain
    ants.image_write(subject_brain, os.path.join(subject_folder,'bigbrain/T1_stripped_N4corrected.nii.gz'))

    # load template files
    template_brain = ants.image_read(os.path.join(bigbrain_scans_folder, 'reference_brain.nii.gz'))

    ############## RUN REGISTRATIONS #################
    # run affine+nonlinear transform
    os.mkdir(os.path.join(subject_folder, 'bigbrain/transform_files'))

    print('Running nonlinear registration...')
    run_multistage_registration(fixed=template_brain, moving=subject_brain, outprefix=os.path.join(subject_folder,'bigbrain/transform_files/'))

    ################## APPLY TRANSFORM ###################
    bigbrain_100um_staining = ants.image_read(os.path.join(bigbrain_scans_folder, 'full16_100um_2009b_sym.nii.gz'))

    warped_bigbrain_100um_staining = ants.apply_transforms(
        fixed=subject_brain,
        moving=bigbrain_100um_staining,
        transformlist=[os.path.join(subject_folder,'bigbrain/transform_files/0GenericAffine.mat'), os.path.join(subject_folder,'bigbrain/transform_files/1InverseWarp.nii.gz')],
        whichtoinvert=[True, False],
	    interpolator='linear'
    )

    ants.image_write(warped_bigbrain_100um_staining, os.path.join(subject_folder,"bigbrain/subject_full16_100um_2009b_sym.nii.gz"))
