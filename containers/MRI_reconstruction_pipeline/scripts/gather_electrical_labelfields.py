import nibabel as nib
import numpy as np
import os
import argparse

# conductivities from https://simnibs.github.io/simnibs/build/html/documentation/conductivity.html
simnibs_electrical_conductivities = {'Background':0, 'White-Matter':0.126, 'Gray-Matter':0.275, 'CSF':1.654, 'Bone':0.01, 'Scalp':0.465, 'Eye_balls':0.5, 'Compact_bone':0.008, 'Spongy_bone':0.025, 'Blood':0.6, 'Muscle':0.16, 'Saline_or_gel':1.0}


def read_charm_labels(subj_dir):
    
    with open(os.path.join(subj_dir, 'simnibs_charm/final_tissues_LUT.txt'), 'r') as f:
        labels = f.readlines()
    
    # add background air
    labels[0] = '0\t  Background\t\t\t   0 0 0 0 \n'

    labels = list(map(lambda x: x.split(), labels))
    LUT = dict(map(lambda x: (int(x[0]), {'name':x[1], 'R':int(x[2]), 'G':int(x[3]), 'B': int(x[4]), 'A':int(x[5])}), labels))
    labels = {key:val['name'] for (key,val) in LUT.items()}
    
    return labels, LUT

def convert_simnibs_labelfield(subj_dir):
    labels, LUT = read_charm_labels(subj_dir)
    label_dict = {val:key for (key, val) in labels.items()}

    label_field = nib.load(os.path.join(subj_dir, 'simnibs_charm/final_tissues.nii.gz'))
    field_value = label_field.get_fdata().astype(int)

    original_labels = np.unique(field_value)
    final_tissues = [labels[i] for i in original_labels]

    # sort the new values and make them contiguous
    # this works because original_labels is sorted (because it comes out of np.unique), so i<=original_labels[i] for every i
    for i in range(len(original_labels)):
        field_value[field_value == original_labels[i]] = i

    # save the volume
    label_field = nib.Nifti1Image(field_value, label_field.affine, label_field.header)
    nib.save(label_field, os.path.join(subj_dir, 'tissue_labels/electrical/simnibs.nii.gz'))

    # save the corresponding label file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/simnibs_labels.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{tissue}\n')

    # save the corresponding LUT file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/simnibs_LUT.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            old_label = label_dict[tissue]
            f.write(f'{idx}\t{LUT[old_label]['name']}\t{LUT[old_label]['R']}\t{LUT[old_label]['G']}\t{LUT[old_label]['B']}\t{LUT[old_label]['A']}\n')
    
    # save the corresponding conductivities file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/simnibs_conductivities.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{simnibs_electrical_conductivities[tissue]}\n')
    return


# dictionary used to remove some of the tissues to reduce the number of labels (mainly done because we don't have electrical conductivity for each tissue)
tissue_conversion_dict = {'Ventricles':'Cerebrospinal_fluid', 'Cerebellum_white_matter':'Cerebrum_white_matter', 'Cerebellum_grey_matter':'Cerebrum_grey_matter', 'Caudate_nucleus':'Cerebrum_grey_matter', 'Putamen':'Cerebrum_grey_matter', 'Globus_pallidus':'Cerebrum_grey_matter', 'Brainstem':'Cerebrum_white_matter', 'Amygdala':'Cerebrum_grey_matter', 'Nucleus_accumbens':'Cerebrum_grey_matter', 'Muscle_ocular':'Muscle', 'Vein':'Artery', 'Nerve_cranial_II_optic':'Cerebrum_white_matter', 'Submandibular_gland':'Parotid_gland', 'Sublingual_gland':'Parotid_gland', 'Tendon_temporalis':'Tendon_galea_aponeurotica', 'Nasal_septum':'Cartilage', 'Vertebrae_cancellous':'Vertebrae_cortical'}

# dictionary used to convert from tissue label to electrical conductivity label
conductivity_conversion_dict = {'Cerebrum_white_matter':'Brain (White Matter)', # can employ anisotropy values, 'Brain (White Matter) across fibers' and 'Brain (White Matter) longitudinal (parallel to fibers)'
'Muscle':'Muscle', # can employ anisotropic values, 'Muscle Parallel' and 'Muscle Perpendicular'
'Eyes':'Eye (Aqueous Humor)',  # chosen as representative, Vitreous Humor makes up most of the eye, but Aqueous Humor has a more middle ground value
'Cerebrum_grey_matter':'Brain (Grey Matter)',
'Midbrain_ventral':'Midbrain',
'Air_internal':'Air',
'Artery':'Blood',
'Other_tissues':'SAT (Subcutaneous Fat)',
'Mucosa':'Mucous Membrane',
'Spinal_cord':'Spinal Cord',
'Skull_cortical':'Skull (Cortical)',
'Cerebrospinal_fluid':'Cerebrospinal Fluid',
'Vertebrae_cortical':'Vertebrae',
'Tendon_galea_aponeurotica':'Tendon\\Ligament',
'Parotid_gland':'Salivary Gland',
'Cartilage':'Cartilage',
'Intervertebral_disc':'Intervertebral Disc',
'Skull_cancellous':'Skull (Cancellous)'}

# values from https://itis.swiss/virtual-population/tissue-properties/database/low-frequency-conductivity/
electrical_conductivities = {'Background':	0.00E+0,'Air':	0.00E+0,'Blood':	6.62E-1,'Bone (Cancellous)':	8.05E-2,'Bone (Cortical)':	6.30E-3,'Brain (Grey Matter)':	4.19E-1,'Brain (Grey Matter) x - along':	7.33E-1,'Brain (Grey Matter) y - across':	2.31E-1,'Brain (White Matter)':	3.48E-1,'Brain (White Matter) across fibers':	2.31E-1,'Brain (White Matter) longitudinal (parallel to fibers)':	7.33E-1,'Cartilage':	7.39E-1,'Cerebellum':	5.77E-1,'Cerebrospinal Fluid':	1.88E+0,'Connective Tissue':	7.92E-2,'Dura':	6.00E-2,'Eye (Aqueous Humor)':	1.88E+0,'Eye (Choroid)':	6.62E-1,'Eye (Ciliary Body)':	4.61E-1,'Eye (Cornea)':	6.20E-1,'Eye (Iris)':	4.61E-1,'Eye (Lens)':	3.45E-1,'Eye (Retina)':	4.19E-1,'Eye (Sclera)':	6.20E-1,'Eye (Vitreous Humor)':	2.16E+0,'Eye Lens (Cortex)':	3.40E-1,'Eye Lens (Nucleus)':	1.90E-1,'Fat':	7.76E-2,'Hippocampus':	4.19E-1,'Hypophysis':	1.05E+0,'Hypothalamus':	4.19E-1,'Intervertebral Disc':	7.39E-1,'Midbrain':	3.50E-1,'Mucous Membrane':	4.61E-1,'Muscle':	4.61E-1,'Muscle Parallel':	4.47E-1,'Muscle Perpendicular':	1.21E-1,'Nerve':	3.48E-1,'Nerve - Across':	2.31E-1,'Nerve - Along':	7.33E-1,'SAT (Subcutaneous Fat)':	7.76E-2,'Salivary Gland':	5.59E-1,'Skin':	1.48E-1,'Skull':	1.79E-2,'Skull (Cancellous)':	9.98E-2,'Skull (Cortical)':	6.45E-3,'Skull (Suture Region)':	1.68E-2,'Spinal Cord':	6.11E-1,'Tendon\\Ligament':	3.68E-1,'Thalamus':	4.75E-1,'Thymus':	1.49E-1,'Thyroid Gland':	4.81E-1,'Tongue':	4.61E-1,'Tooth':	6.30E-3,'Tooth (Dentine)':	6.30E-3,'Tooth (Enamel)':	6.30E-3,'Trachea':	3.42E-1,'Vertebrae':	6.30E-3,'Water':	1.62E-3}

def read_Sim4Life_labels(subj_dir, raise_error = True):
    # this function reads the labels of the segmentation and checks if they are as expected (i.e. in the right order)
    # raise_error flag is used to raise an error when the labels are not as expected
    
    with open(os.path.join(subj_dir, 'tissue_labels/sim4life_raw/label_field.txt'), 'r') as f:
        labels = f.readlines()

    # remove header
    labels = labels[1:]
    
    # add background air
    labels[0] = 'C0.000000 0.000000 0.000000 0.000000 Background\n'
    
    # add numbers
    labels = [f'{idx} '+x.strip('C').strip() for idx,x in enumerate(labels)]
    
    labels = list(map(lambda x: x.split(), labels))
    LUT = dict(map(lambda x: (int(x[0]), {'name':x[5], 'R':int(255*float(x[1])), 'G':int(255*float(x[2])), 'B': int(255*float(x[3])), 'A':int(255*float(x[4]))}), labels))
    labels = {key:val['name'] for (key,val) in LUT.items()}
    
    if raise_error:
        # expected labels in Sim4Life segmentation
        sim4life_expected_labels = [['Background'], ['Cerebrum_white_matter'], ['Cerebrum_grey_matter'], ['Ventricles'], ['Cerebellum_white_matter'], ['Cerebellum_grey_matter'], ['Thalamus'], ['Caudate_nucleus'], ['Putamen'], ['Globus_pallidus'], ['Brainstem'], ['Hippocampus'], ['Amygdala'], ['Nucleus_accumbens'], ['Midbrain_ventral', '__unused__'], ['Air_internal'], ['Artery', '__unused__'], ['Eyes', '__unused__'], ['Other_tissues'], ['Muscle_ocular', '__unused__'], ['Mucosa', '__unused__'], ['Spinal_cord', '__unused__'], ['Vein', '__unused__'], ['Skull_cortical'], ['Cerebrospinal_fluid'], ['Nerve_cranial_II_optic', '__unused__'], ['Vertebrae_cortical', '__unused__'], ['Skin'], ['Muscle'], ['Tongue', '__unused__'], ['Tendon_galea_aponeurotica', '__unused__'], ['Parotid_gland', '__unused__'], ['Submandibular_gland', '__unused__'], ['Sublingual_gland', '__unused__'], ['Tendon_temporalis', '__unused__'], ['Nasal_septum', '__unused__'], ['Cartilage', '__unused__'], ['Intervertebral_disc', '__unused__'], ['Skull_cancellous'], ['Vertebrae_cancellous', '__unused__'], ['Dura', '__unused__']]
        
        correct = []
        for i in range(len(labels)):
            correct.append(labels[i] in sim4life_expected_labels[i])

        if not all(correct):
            raise ValueError(f'Sim4Life segmentation probably changed labels, in particular {[lab for idx, lab in enumerate(labels) if not correct[idx]]} do not agree with expected values (i.e. {[lab for idx, lab in enumerate(sim4life_expected_labels) if not correct[idx]]})')

    return labels, LUT

def convert_sim4life_labelfield(subj_dir):
    # this function converts the label field generated with sim4life by joining some of the tissues into just one (bringing the total to at most 20 tissues)
    
    labels, LUT = read_Sim4Life_labels(subj_dir, raise_error = True)
    label_dict = {val:key for (key, val) in labels.items()}
    
    label_field = nib.load(os.path.join(subj_dir, 'tissue_labels/sim4life_raw/label_field.nii.gz'))
    field_value = label_field.get_fdata().astype(np.uint8)
    
    # removes some of the labels in the volume by joining it to other tissue types
    for source_name, target_name in tissue_conversion_dict.items():
        if source_name in label_dict.keys():
            source_value = label_dict[source_name]
            target_value = label_dict[target_name]
        
            field_value[field_value == source_value] = target_value

    original_labels = np.unique(field_value)
    final_tissues = [labels[i] for i in original_labels]

    # sort the new values and make them contiguous
    # this works because original_labels is sorted (because it comes out of np.unique), so i<=original_labels[i] for every i
    for i in range(len(original_labels)):
        field_value[field_value == original_labels[i]] = i

    # save the volume
    label_field = nib.Nifti1Image(field_value, label_field.affine, label_field.header)
    nib.save(label_field, os.path.join(subj_dir, 'tissue_labels/electrical/sim4life.nii.gz'))
    
    # save the corresponding labels file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/sim4life_labels.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in conductivity_conversion_dict.keys():
                tissue = conductivity_conversion_dict[tissue]
            f.write(f'{idx},{tissue}\n')
    
    # save the corresponding LUT file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/sim4life_LUT.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            old_label = label_dict[tissue]
            tissue = LUT[old_label]['name']
            if tissue in conductivity_conversion_dict.keys():
                tissue = conductivity_conversion_dict[tissue]
            f.write(f'{idx}\t{tissue}\t{LUT[old_label]['R']}\t{LUT[old_label]['G']}\t{LUT[old_label]['B']}\t{LUT[old_label]['A']}\n')
    
    # save the corresponding conductivities file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/sim4life_conductivities.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in conductivity_conversion_dict.keys():
                tissue = conductivity_conversion_dict[tissue]
            f.write(f'{idx},{electrical_conductivities[tissue]}\n')
    
    return



if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Extracts and edits label fields.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 1. Define the Subject Folder Argument
    parser.add_argument(
        '--subject_dir',
        type=str,
        required=True,
        help='Path to the subject folder (e.g., /SUBJECTS/<subjectname>/)'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call your main processing function
    subj_dir = args.subject_dir

    convert_simnibs_labelfield(subj_dir)

    if os.path.isdir(os.path.join(subj_dir, 'tissue_labels/sim4life_raw')):
        print("Sim4Life tissue labels field folder detected, generating additional label field from it")
        convert_sim4life_labelfield(subj_dir)
    else:
        print("Sim4Life tissue labels field folder not detected, will only generate SimNibs label field.")

    # save the full ITIS conductivities file
    with open(os.path.join(subj_dir, 'tissue_labels/electrical/ITIS_conductivities.txt'), 'w') as f:
        for key,val in electrical_conductivities.items():
            f.write(f'{key},{val}\n')
