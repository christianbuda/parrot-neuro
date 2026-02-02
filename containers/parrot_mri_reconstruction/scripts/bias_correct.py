import ants
import sys
import os

# usage: python bias_correct.py input.nii.gz output.nii.gz

def run_n4(input_path, output_path):
    print(f"Loading: {input_path}")
    img = ants.image_read(input_path)

    # 1. Generate a rough mask of the brain
    # This prevents the bright scalp fat from confusing the bias correction
    print("Generating rough mask for calculation...")
    mask = ants.get_mask(img, low_thresh=None, high_thresh=None, cleanup=1)

    # 2. Run N4 Bias Field Correction
    # weight_mask=mask tells N4: "Calculate the lighting correction using only 
    # the brain pixels, but apply that correction to the whole image."
    print("Running N4 Bias Correction...")
    corrected_img = ants.n4_bias_field_correction(img, mask=mask)

    # 3. Save
    print(f"Saving to: {output_path}")
    ants.image_write(corrected_img, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing arguments.")
        print("Usage: python bias_correct.py <input_t1> <output_name>")
    else:
        run_n4(sys.argv[1], sys.argv[2])
