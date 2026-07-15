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
        # Initialize by centre of MASS (feature 1), not geometric centre (0). Both
        # images are skull-stripped brains (subject: T1_stripped*mask; template:
        # reference_brain), so the intensity centroid is an unbiased, far more robust
        # starting estimate: geometric-centre init let the hard cross-contrast affine
        # lock into a bad basin (near-identity linear + ~10 cm z-shift), leaving the
        # warp covering only the upper third of the brain (silently corrupting the
        # neural density downstream). Guarded by the coverage gate below.
        "--initial-moving-transform", f"[{f_ptr},{m_ptr},1]",
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

# 1. Define the Subject Folders Arguments
    parser.add_argument(
        '--subject', 
        type=str,
        required=True,
        help='Identifier of the subject (e.g., "01")'
    )
        
    parser.add_argument(
        '--output_dir', 
        type=str,
        required=True,
        help='Path to the output folder (e.g., /derivatives/)'
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
    subject = args.subject
    output_dir = args.output_dir
    bigbrain_scans_folder = args.template_dir

    ################## LOAD FILES ########################
    import ants
    
    # load brain stripped T1
    subject_brain = ants.image_read(os.path.join(output_dir,f'synthstrip/sub-{subject}/T1_stripped.nii.gz'))

    # apply bias field correction and reapply brain mask
    subject_brain = ants.n4_bias_field_correction(subject_brain)
    subject_brain = subject_brain * ants.image_read(os.path.join(output_dir,f'synthstrip/sub-{subject}/T1_stripped_mask.nii.gz'))

    # save bias field corrected brain
    ants.image_write(subject_brain, os.path.join(output_dir,f'bigbrain/sub-{subject}/T1_stripped_N4corrected.nii.gz'))

    # load template files
    template_brain = ants.image_read(os.path.join(bigbrain_scans_folder, 'reference_brain.nii.gz'))

    ############## RUN REGISTRATIONS #################
    # run affine+nonlinear transform
    os.mkdir(os.path.join(output_dir, f'bigbrain/sub-{subject}/transform_files'))

    print('Running nonlinear registration...')
    run_multistage_registration(fixed=template_brain, moving=subject_brain, outprefix=os.path.join(output_dir,f'bigbrain/sub-{subject}/transform_files/'))

    ################## APPLY TRANSFORM ###################
    bigbrain_100um_staining = ants.image_read(os.path.join(bigbrain_scans_folder, 'full16_100um_2009b_sym.nii.gz'))

    warped_bigbrain_100um_staining = ants.apply_transforms(
        fixed=subject_brain,
        moving=bigbrain_100um_staining,
        transformlist=[os.path.join(output_dir,f'bigbrain/sub-{subject}/transform_files/0GenericAffine.mat'), os.path.join(output_dir,f'bigbrain/sub-{subject}/transform_files/1InverseWarp.nii.gz')],
        whichtoinvert=[True, False],
	    interpolator='linear'
    )

    ants.image_write(warped_bigbrain_100um_staining, os.path.join(output_dir,f"bigbrain/sub-{subject}/subject_full16_100um_2009b_sym.nii.gz"))

    ################## VALIDATE COVERAGE #################
    # Fail fast on a mis-registration. A healthy warp fills essentially the whole
    # cortex; a failed subject<->BigBrain affine leaves the warped staining covering
    # only a fraction of the brain, silently corrupting the per-parcel neural density
    # (uncovered voxels default to a flat weight, saturated ones to zero). That is a
    # SILENT data-quality failure -- ANTs still reports success -- so we abort here,
    # before the expensive dipole/leadfield stages consume the bad density, rather
    # than degrade. Cohort stats (227 LEMON subjects): healthy warps cover 99.7-99.8%;
    # the five failures covered 3-30%. 90% sits far below every good subject and far
    # above every failure. (The parrot_qc bigbrain stage reports the same metric and
    # warns below 95%; this harder threshold only aborts unambiguous failures.)
    COVERAGE_FAIL_FRAC = 0.90
    brain_mask = subject_brain.numpy() > 0
    covered = (warped_bigbrain_100um_staining.numpy() > 0) & brain_mask
    n_brain = int(brain_mask.sum())
    coverage = covered.sum() / n_brain if n_brain else 0.0
    print(f'BigBrain warp covers {coverage*100:.1f}% of the subject brain.')
    if coverage < COVERAGE_FAIL_FRAC:
        raise SystemExit(
            f'ERROR: BigBrain registration failed for sub-{subject}: warped staining '
            f'covers only {coverage*100:.1f}% of the brain (< {COVERAGE_FAIL_FRAC*100:.0f}%). '
            'The subject<->BigBrain affine likely locked into a bad local minimum; the '
            'neural density would be unreliable. Delete this subject\'s bigbrain_log.txt '
            'and re-run (center-of-mass init should fix most cases).')
