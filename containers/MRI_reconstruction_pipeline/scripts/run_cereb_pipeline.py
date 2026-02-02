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


def dilate_mask_by_distance(mask, radius_mm):
    """
    Dilates a binary mask by a specific physical distance (mm).
    """
    # 1. Compute the distance map (Maurer Distance)
    #    This calculates the physical distance from every voxel to the mask boundary.
    #    Standard ANTs convention: 
    #       Inside the mask = Negative values
    #       Outside the mask = Positive values
    dist_map = ants.iMath(mask, "MaurerDistance")
    
    # 2. Threshold the distance map
    #    We want to keep everything originally inside (negative values)
    #    PLUS everything outside up to 'radius_mm' (positive values <= radius_mm)
    dilated_mask = ants.threshold_image(
        dist_map, 
        low_thresh=dist_map.min()-1,
        high_thresh=radius_mm, 
    )
    
    # 3. Ensure it matches the original mask's type/header
    dilated_mask = ants.copy_image_info(mask, dilated_mask)
    
    return dilated_mask


if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Process subject data using a cerebellum template.",
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
        help='Path to the cerebellum template folder (e.g., /home/cerebellum_template/)'
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
    cerebellum_template_folder = args.template_dir


    ################## LOAD FILES ########################
    import ants

    # load brain stripped T1
    subject_brain = ants.image_read(os.path.join(subject_folder,'synthstrip/T1_stripped.nii.gz'))

    # apply bias field correction and reapply brain mask
    subject_brain = ants.n4_bias_field_correction(subject_brain)
    subject_brain = subject_brain * ants.image_read(os.path.join(subject_folder,'synthstrip/T1_stripped_mask.nii.gz'))

    # save bias field corrected brain
    ants.image_write(subject_brain, os.path.join(subject_folder,'cerebellum/T1_stripped_N4corrected.nii.gz'))

    # then load and resample cerebellum mask
    subject_cerebmask = ants.image_read(os.path.join(subject_folder,'fastsurfer/mri/cerebellum.CerebNet.nii.gz'))
    subject_cerebmask = ants.resample_image_to_target(subject_cerebmask, subject_brain, interp_type='genericLabel')
    subject_cerebmask = ants.threshold_image(subject_cerebmask, low_thresh=0.5)


    # load template files
    template_brain = ants.image_read(os.path.join(cerebellum_template_folder, 'brain.nii.gz'))
    template_cerebmask = ants.image_read(os.path.join(cerebellum_template_folder, 'mask.nii.gz'))
    template_cerebmask = ants.threshold_image(template_cerebmask, low_thresh=0.5)

    # Dilate by exactly 4.0 millimeters, regardless of pixel spacing
    # dilation is used to take advantage of edge features in the registration
    # not used anymore
    subject_cerebmask = dilate_mask_by_distance(subject_cerebmask, radius_mm=4.0)
    template_cerebmask = dilate_mask_by_distance(template_cerebmask, radius_mm=4.0)

    ############## RUN REGISTRATIONS #################
    # run affine+nonlinear transform
    os.mkdir(os.path.join(subject_folder, 'cerebellum/transform_files'))

    print('Running nonlinear registration...')
    
    # we only provide the fixed mask because we want ants to look everywhere in the subject brain for the best cerebellar alignment
    # (actually im afraid that the intersection of the two masks will make the registration very difficult in some edge cases)
    run_multistage_registration(fixed=template_brain, moving=subject_brain, outprefix=os.path.join(subject_folder,'cerebellum/transform_files/'))

    ################## APPLY TRANSFORMS to VOLUMES ###################

    ### cerebellum template
    warped_template = ants.apply_transforms(
        fixed=subject_brain,
        moving=template_brain,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')],
        whichtoinvert=[True]
    )

    ants.image_write(warped_template, os.path.join(subject_folder,"cerebellum/affine_template_brain.nii.gz"))

    warped_template = ants.apply_transforms(
	fixed=subject_brain,
	moving=template_brain,
	transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat'), os.path.join(subject_folder,'cerebellum/transform_files/1InverseWarp.nii.gz')],
	whichtoinvert=[True, False]
    )

    ants.image_write(warped_template, os.path.join(subject_folder,"cerebellum/nonlinear_template_brain.nii.gz"))

    ### cerebellum gray labels
    gray_labels = ants.image_read(os.path.join(cerebellum_template_folder, 'Cerebellum_GM_Labels.nii.gz'))

    warped_gray_labels = ants.apply_transforms(
        fixed=subject_brain,
        moving=gray_labels,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')],
        whichtoinvert=[True],
	interpolator='genericLabel'
    )

    ants.image_write(warped_gray_labels, os.path.join(subject_folder,"cerebellum/affine_gray_labels.nii.gz"))

    warped_gray_labels = ants.apply_transforms(
        fixed=subject_brain,
        moving=gray_labels,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat'), os.path.join(subject_folder,'cerebellum/transform_files/1InverseWarp.nii.gz')],
        whichtoinvert=[True, False],
	interpolator='genericLabel'
    )

    ants.image_write(warped_gray_labels, os.path.join(subject_folder,"cerebellum/nonlinear_gray_labels.nii.gz"))

    ### cerebellum white labels
    white_labels = ants.image_read(os.path.join(cerebellum_template_folder, 'Cerebellum_WM_Labels.nii.gz'))

    warped_white_labels = ants.apply_transforms(
        fixed=subject_brain,
        moving=white_labels,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')],
        whichtoinvert=[True],
	interpolator='genericLabel'
    )

    ants.image_write(warped_white_labels, os.path.join(subject_folder,"cerebellum/affine_white_labels.nii.gz"))

    warped_white_labels = ants.apply_transforms(
        fixed=subject_brain,
        moving=white_labels,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat'), os.path.join(subject_folder,'cerebellum/transform_files/1InverseWarp.nii.gz')],
        whichtoinvert=[True, False],
	interpolator='genericLabel'
    )

    ants.image_write(warped_white_labels, os.path.join(subject_folder,"cerebellum/nonlinear_white_labels.nii.gz"))

    ################## APPLY TRANSFORMS 1 ###################

    # load points
    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    # line 5 contains the header in this file
    points = mesh[5:int(mesh[4].split()[1])+5]
    points = np.array(list(map(lambda x: x.split(), points))).astype(float)
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # convert to LPS coordinates (needed for ants to work properly here)
    points = points * [-1,-1,1]

    ### warp affine
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]

    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])+5] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6f}'.format(y), x)))+'\n', warped_points.to_numpy()))

    with open(os.path.join(subject_folder,'cerebellum/affine_Cerebellum_Inner_Surf_With_Features.vtk'), 'w') as f:
        f.writelines(mesh)

    ### warp nonlinear
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/1Warp.nii.gz'), os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]


    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])+5] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6f}'.format(y), x)))+'\n', warped_points.to_numpy()))

    with open(os.path.join(subject_folder,'cerebellum/nonlinear_Cerebellum_Inner_Surf_With_Features.vtk'), 'w') as f:
        f.writelines(mesh)
        
    ################## APPLY TRANSFORMS 2 ###################

    # load points
    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_GM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    # line 5 contains the header in this file
    points = mesh[5:int(mesh[4].split()[1])//3+5+1]
    points = np.concatenate([np.array(list(map(lambda x: x.split(), points[:-1]))).astype(float).reshape((-1,3)), np.array(points[-1].split()).astype(float).reshape((2,3))])
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # convert to LPS coordinates (needed for ants to work properly here)
    points = points * [-1,-1,1]

    ### warp affine
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]

    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_GM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])//3+5+1] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6g}'.format(y), x)))+'\n', warped_points[:-2].to_numpy().reshape((-1,9)))) + [' '.join(list(map(lambda y: '{:.6g}'.format(y), warped_points[-2:].to_numpy().flatten())))+'\n']

    with open(os.path.join(subject_folder,'cerebellum/affine_Cerebellum_Surf_GM_Labels.vtk'), 'w') as f:
        f.writelines(mesh)

    ### warp nonlinear
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/1Warp.nii.gz'), os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]


    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_GM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])//3+5+1] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6g}'.format(y), x)))+'\n', warped_points[:-2].to_numpy().reshape((-1,9)))) + [' '.join(list(map(lambda y: '{:.6g}'.format(y), warped_points[-2:].to_numpy().flatten())))+'\n']
    
    with open(os.path.join(subject_folder,'cerebellum/nonlinear_Cerebellum_Surf_GM_Labels.vtk'), 'w') as f:
        f.writelines(mesh)
        
    ################## APPLY TRANSFORMS 3 ###################

    # load points
    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_WM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    # line 5 contains the header in this file
    points = mesh[5:int(mesh[4].split()[1])//3+5+1]
    points = np.concatenate([np.array(list(map(lambda x: x.split(), points[:-1]))).astype(float).reshape((-1,3)), np.array(points[-1].split()).astype(float).reshape((1,3))])
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # convert to LPS coordinates (needed for ants to work properly here)
    points = points * [-1,-1,1]

    ### warp affine
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]

    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_WM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])//3+5+1] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6g}'.format(y), x)))+'\n', warped_points[:-1].to_numpy().reshape((-1,9)))) + [' '.join(list(map(lambda y: '{:.6g}'.format(y), warped_points[-1:].to_numpy().flatten())))+'\n']
    
    with open(os.path.join(subject_folder,'cerebellum/affine_Cerebellum_Surf_WM_Labels.vtk'), 'w') as f:
        f.writelines(mesh)

    ### warp nonlinear
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/1Warp.nii.gz'), os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]


    with open(os.path.join(cerebellum_template_folder,'Cerebellum_Surf_WM_Labels.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])//3+5+1] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6g}'.format(y), x)))+'\n', warped_points[:-1].to_numpy().reshape((-1,9)))) + [' '.join(list(map(lambda y: '{:.6g}'.format(y), warped_points[-1:].to_numpy().flatten())))+'\n']
    
    with open(os.path.join(subject_folder,'cerebellum/nonlinear_Cerebellum_Surf_WM_Labels.vtk'), 'w') as f:
        f.writelines(mesh)

    ################## APPLY TRANSFORMS 4 ###################

    # load points
    with open(os.path.join(cerebellum_template_folder,'manifold_Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    # line 5 contains the header in this file
    points = mesh[5:int(mesh[4].split()[1])+5]
    points = np.array(list(map(lambda x: x.split(), points))).astype(float)
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])

    # convert to LPS coordinates (needed for ants to work properly here)
    points = points * [-1,-1,1]

    ### warp affine
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]

    with open(os.path.join(cerebellum_template_folder,'manifold_Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])+5] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6f}'.format(y), x)))+'\n', warped_points.to_numpy()))

    with open(os.path.join(subject_folder,'cerebellum/affine_manifold_Cerebellum_Inner_Surf_With_Features.vtk'), 'w') as f:
        f.writelines(mesh)

    ### warp nonlinear
    warped_points = ants.apply_transforms_to_points(
        dim=3,
        points=points,
        transformlist=[os.path.join(subject_folder,'cerebellum/transform_files/1Warp.nii.gz'), os.path.join(subject_folder,'cerebellum/transform_files/0GenericAffine.mat')]
    )

    # convert back to RAS
    warped_points = warped_points * [-1,-1,1]


    with open(os.path.join(cerebellum_template_folder,'manifold_Cerebellum_Inner_Surf_With_Features.vtk'), 'r') as f:
        mesh = f.readlines()

    mesh[5:int(mesh[4].split()[1])+5] = list(map(lambda x: ' '.join(list(map(lambda y: '{:.6f}'.format(y), x)))+'\n', warped_points.to_numpy()))

    with open(os.path.join(subject_folder,'cerebellum/nonlinear_manifold_Cerebellum_Inner_Surf_With_Features.vtk'), 'w') as f:
        f.writelines(mesh)