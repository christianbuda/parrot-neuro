import os
import nibabel as nib
import meshio
import numpy as np


if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Converts mesh file to vtu transforming according to nifti affine.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--reference_nifti',
        type=str,
        required=True,
        help='Path to reference nifti file'
    )

    parser.add_argument(
        '--mesh',
        type=str,
        required=True,
        help='Path to input mesh'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    nifti = args.reference_nifti
    input = args.mesh


    mesh = meshio.read(input)
    points = mesh.points

    # load original T1 to realign mesh to world space
    nifti = nib.load(nifti)

    affine = nifti.affine
    zooms = np.array(nifti.header.get_zooms())

    # scale back from mm to voxel size
    points /= zooms


    # Apply transform
    points = (affine @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T
    points = points[:,:3]

    mesh.points = points

    meshio.write(os.path.splitext(input)[0]+'.vtu', mesh)
