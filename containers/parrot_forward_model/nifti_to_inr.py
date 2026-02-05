import nibabel as nib
import numpy as np

def save_inr(vol, voxel_size: tuple[float, float, float], fname: str):
    """
    Save a volume (described as a numpy array) to INR format.
    Code inspired by iso2mesh (http://iso2mesh.sf.net) by Q. Fang
    INPUTS:
    - vol: volume as a 3D numpy array
    - h: voxel sizes
    - fname: filename for saving the inr file
    """
    with open(fname, "wb") as fid:

        btype, bitlen = {
            "uint8": ("unsigned fixed", 8),
            "uint16": ("unsigned fixed", 16),
            "float32": ("float", 32),
            "float64": ("float", 64),
        }[vol.dtype.name]
    
        xdim, ydim, zdim = vol.shape
        header = "\n".join(
            [
                "#INRIMAGE-4#{",
                f"XDIM={xdim}",
                f"YDIM={ydim}",
                f"ZDIM={zdim}",
                "VDIM=1",
                f"TYPE={btype}",
                f"PIXSIZE={bitlen} bits",
                "CPU=decm",
                f"VX={voxel_size[0]:f}",
                f"VY={voxel_size[1]:f}",
                f"VZ={voxel_size[2]:f}",
            ]
        )
        header += "\n"
    
        header = header + "\n" * (256 - 4 - len(header)) + "##}\n"
    
        fid.write(header.encode("ascii"))
        fid.write(vol.tobytes(order="F"))
    
    return

if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Converts nifti label field to inr format.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--nifti_path',
        type=str,
        required=True,
        help='Path to nifti file'
    )

    parser.add_argument(
        '--inr_path',
        type=str,
        required=True,
        help='Path to output inr file'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    input = args.nifti_path
    output = args.inr_path


    # read and convert file
    nii_file = nib.load(input)

    label_field_data = nii_file.get_fdata()
    label_field_data = label_field_data.astype(np.uint8)

    voxel_size = nii_file.header.get_zooms()

    save_inr(label_field_data, voxel_size, output)
