import nibabel as nib
import numpy as np
import os
import argparse

# acoustic parameters

# values from https://itis.swiss/virtual-population/tissue-properties/database/density/, accessed January 2026
density = {'Background': 1.0, 'Adrenal Gland': 1028.0, 'Air': 1.0, 'Bile': 928.0, 'Blood': 1050.0, 'Blood Plasma': 1020.0, 'Blood Serum': 1024.0, 'Blood Vessel Wall': 1102.0, 'Bone (Cancellous)': 1178.0, 'Bone (Cortical)': 1908.0, 'Bone Marrow (Red)': 1029.0, 'Bone Marrow (Yellow)': 980.0, 'Brain': 1046.0, 'Brain (Grey Matter)': 1045.0, 'Brain (White Matter)': 1041.0, 'Breast Fat': 911.0, 'Breast Gland': 1041.0, 'Bronchi': 1102.0, 'Bronchi lumen': 1.0, 'Cartilage': 1100.0, 'Cerebellum': 1045.0, 'Cerebrospinal Fluid': 1007.0, 'Cervix': 1105.0, 'Commissura Anterior': 1041.0, 'Commissura Posterior': 1041.0, 'Connective Tissue': 1027.0, 'Diaphragm': 1090.0, 'Ductus Deferens': 1102.0, 'Dura': 1174.0, 'Epididymis': 1082.0, 'Esophagus': 1040.0, 'Esophagus Lumen': 1.0, 'Extracellular Fluids': 1011.0, 'Eye (Aqueous Humor)': 994.0, 'Eye (Choroid)': 1050.0, 'Eye (Ciliary Body)': 1090.0, 'Eye (Cornea)': 1062.0, 'Eye (Iris)': 1090.0, 'Eye (Lens)': 1076.0, 'Eye (Retina)': 1036.0, 'Eye (Sclera)': 1032.0, 'Eye (Vitreous Humor)': 1005.0, 'Eye Lens (Cortex)': 1076.0, 'Eye Lens (Nucleus)': 1076.0, 'Fat': 911.0, 'Fat (Average Infiltrated)': 911.0, 'Fat (Not Infiltrated)': 911.0, 'Gallbladder': 1071.0, 'Heart Lumen': 1050.0, 'Heart Muscle': 1081.0, 'Hippocampus': 1045.0, 'Hypophysis': 1053.0, 'Hypothalamus': 1045.0, 'Intervertebral Disc': 1100.0, 'Kidney': 1066.0, 'Kidney (Cortex)': 1049.0, 'Kidney (Medulla)': 1044.0, 'Large Intestine': 1088.0, 'Large Intestine Lumen': 1045.0, 'Larynx': 1100.0, 'Liver': 1079.0, 'Lung': 394.0, 'Lung (Deflated)': 1050.0, 'Lung (Inflated)': 394.0, 'Lymph': 1019.0, 'Lymphnode': 1035.0, 'Mandible': 1908.0, 'Medulla Oblongata': 1046.0, 'Meniscus': 1100.0, 'Midbrain': 1046.0, 'Mucous Membrane': 1102.0, 'Muscle': 1090.0, 'Nerve': 1075.0, 'Ovary': 1048.0, 'Pancreas': 1087.0, 'Penis': 1102.0, 'Pharynx': 1.0, 'Pineal Body': 1053.0, 'Placenta': 1018.0, 'Pons': 1046.0, 'Prostate': 1045.0, 'SAT (Subcutaneous Fat)': 911.0, 'Salivary Gland': 1048.0, 'Scalp': 1109.0, 'Seminal vesicle': 1045.0, 'Skin': 1109.0, 'Skull': 1908.0, 'Skull (Cancellous)': 1178.0, 'Skull (Cortical)': 1908.0, 'Small Intestine': 1030.0, 'Small Intestine Lumen': 1045.0, 'Spinal Cord': 1075.0, 'Spleen': 1089.0, 'Stomach': 1088.0, 'Stomach Lumen': 1045.0, 'Tendon\\Ligament': 1142.0, 'Testis': 1082.0, 'Thalamus': 1045.0, 'Thymus': 1023.0, 'Thyroid Gland': 1050.0, 'Tongue': 1090.0, 'Tooth': 2180.0, 'Tooth (Dentine)': 2063.0, 'Tooth (Enamel)': 2958.0, 'Trachea': 1080.0, 'Trachea Lumen': 1.0, 'Ureter\\Urethra': 1102.0, 'Urinary Bladder Wall': 1086.0, 'Urine': 1024.0, 'Uterus': 1105.0, 'Vagina': 1088.0, 'Vertebrae': 1908.0, 'Water': 994.0}
# values from https://itis.swiss/virtual-population/tissue-properties/database/acoustic-properties/speed-of-sound/, accessed January 2026
speed_of_sound = {'Background': 343.0, 'Adrenal Gland': 1500.0, 'Air': 343.0, 'Bile': 1500.0, 'Blood': 1578.2, 'Blood Plasma': 1549.3, 'Blood Serum': 1500.0, 'Blood Vessel Wall': 1569.1, 'Bone (Cancellous)': 2117.5, 'Bone (Cortical)': 3514.9, 'Bone Marrow (Red)': 1450.0, 'Bone Marrow (Yellow)': 1371.9, 'Brain': 1546.3, 'Brain (Grey Matter)': 1500.0, 'Brain (White Matter)': 1552.5, 'Breast Fat': 1440.2, 'Breast Gland': 1505.0, 'Bronchi': 1569.1, 'Bronchi lumen': 343.0, 'Cartilage': 1639.6, 'Cerebellum': 1537.0, 'Cerebrospinal Fluid': 1504.5, 'Cervix': 1629.0, 'Commissura Anterior': 1552.5, 'Commissura Posterior': 1552.5, 'Connective Tissue': 1545.0, 'Diaphragm': 1588.4, 'Ductus Deferens': 1569.1, 'Dura': 1500.0, 'Epididymis': 1595.0, 'Esophagus': 1500.0, 'Esophagus Lumen': 343.0, 'Extracellular Fluids': 1554.2, 'Eye (Aqueous Humor)': 1537.0, 'Eye (Choroid)': 1530.8, 'Eye (Ciliary Body)': 1554.0, 'Eye (Cornea)': 1565.8, 'Eye (Iris)': 1542.0, 'Eye (Lens)': 1643.3, 'Eye (Retina)': 1576.5, 'Eye (Sclera)': 1630.4, 'Eye (Vitreous Humor)': 1525.8, 'Eye Lens (Cortex)': 1643.3, 'Eye Lens (Nucleus)': 1643.3, 'Fat': 1440.2, 'Fat (Average Infiltrated)': 1440.2, 'Fat (Not Infiltrated)': 1440.2, 'Gallbladder': 1583.6, 'Heart Lumen': 1578.2, 'Heart Muscle': 1561.3, 'Hippocampus': 1500.0, 'Hypophysis': 1500.0, 'Hypothalamus': 1500.0, 'Intervertebral Disc': 1639.6, 'Kidney': 1554.3, 'Kidney (Cortex)': 1571.3, 'Kidney (Medulla)': 1564.0, 'Large Intestine': 1500.0, 'Large Intestine Lumen': 1535.4, 'Larynx': 1639.6, 'Liver': 1585.7, 'Lung': 949.3, 'Lung (Deflated)': 1500.0, 'Lung (Inflated)': 949.3, 'Lymph': 1500.0, 'Lymphnode': 1586.0, 'Mandible': 3514.9, 'Medulla Oblongata': 1546.3, 'Meniscus': 1639.6, 'Midbrain': 1546.3, 'Mucous Membrane': 1620.7, 'Muscle': 1588.4, 'Nerve': 1629.5, 'Ovary': 1595.0, 'Pancreas': 1591.0, 'Penis': 1569.1, 'Pharynx': 343.0, 'Pineal Body': 1500.0, 'Placenta': 1500.0, 'Pons': 1546.3, 'Prostate': 1559.5, 'SAT (Subcutaneous Fat)': 1477.0, 'Salivary Gland': 1559.5, 'Scalp': 1624.0, 'Seminal vesicle': 1559.5, 'Skin': 1624.0, 'Skull': 2770.3, 'Skull (Cancellous)': 2117.5, 'Skull (Cortical)': 2813.7, 'Small Intestine': 1500.0, 'Small Intestine Lumen': 1535.4, 'Spinal Cord': 1542.0, 'Spleen': 1567.6, 'Stomach': 1500.0, 'Stomach Lumen': 1535.4, 'Tendon\\Ligament': 1750.0, 'Testis': 1595.0, 'Thalamus': 1500.0, 'Thymus': 1513.1, 'Thyroid Gland': 1500.0, 'Tongue': 1588.4, 'Tooth': 4565.9, 'Tooth (Dentine)': 3639.8, 'Tooth (Enamel)': 5491.9, 'Trachea': 1639.6, 'Trachea Lumen': 343.0, 'Ureter\\Urethra': 1569.1, 'Urinary Bladder Wall': 1500.0, 'Urine': 1537.7, 'Uterus': 1629.0, 'Vagina': 1500.0, 'Vertebrae': 3514.9, 'Water': 1482.3}
# values from https://itis.swiss/virtual-population/tissue-properties/database/acoustic-properties/attenuation-constant/ at 500kHz, accessed January 2026
attenuation_constant = {'Background': 0.00978598664525, 'Adrenal Gland': 7.746, 'Air': 0.00978598664525, 'Bile': 0.1028468243008211, 'Blood': 1.1436339565230036, 'Blood Plasma': 0.4812093766987019, 'Blood Serum': 0.0, 'Blood Vessel Wall': 3.51, 'Bone (Cancellous)': 20.45793823745892, 'Bone (Cortical)': 27.2765, 'Bone Marrow (Red)': 6.25, 'Bone Marrow (Yellow)': 2.0526675477076903, 'Brain': 2.762957751445371, 'Brain (Grey Matter)': 0.5187223387847191, 'Brain (White Matter)': 3.1512532531387163, 'Breast Fat': 2.0526675477076903, 'Breast Gland': 3.052933527772919, 'Bronchi': 3.51, 'Bronchi lumen': 0.00978598664525, 'Cartilage': 0.21985, 'Cerebellum': 1.2818218166756143, 'Cerebrospinal Fluid': 0.05, 'Cervix': 4.05785, 'Commissura Anterior': 3.1512532531387163, 'Commissura Posterior': 3.1512532531387163, 'Connective Tissue': 6.438291282094524, 'Diaphragm': 3.356379961658642, 'Ductus Deferens': 3.51, 'Dura': 6.701031, 'Epididymis': 0.8212601132195042, 'Esophagus': 2.87145, 'Esophagus Lumen': 0.00978598664525, 'Extracellular Fluids': 0.05, 'Eye (Aqueous Humor)': 0.05, 'Eye (Choroid)': 1.1436339565230036, 'Eye (Ciliary Body)': 3.356379961658642, 'Eye (Cornea)': 0.13010764773832476, 'Eye (Iris)': 3.356379961658642, 'Eye (Lens)': 0.333335, 'Eye (Retina)': 0.5187223387847191, 'Eye (Sclera)': 0.39816337010589165, 'Eye (Vitreous Humor)': 2.5, 'Eye Lens (Cortex)': 0.333335, 'Eye Lens (Nucleus)': 0.333335, 'Fat': 2.0526675477076903, 'Fat (Average Infiltrated)': 2.0526675477076903, 'Fat (Not Infiltrated)': 2.0526675477076903, 'Gallbladder': 0.755, 'Heart Lumen': 1.1436339565230036, 'Heart Muscle': 1.6357464370893975, 'Hippocampus': 0.5187223387847191, 'Hypophysis': 7.746, 'Hypothalamus': 0.5187223387847191, 'Intervertebral Disc': 0.21985, 'Kidney': 1.382112322637471, 'Kidney (Cortex)': 6.0, 'Kidney (Medulla)': 1.382112322637471, 'Large Intestine': 2.87145, 'Large Intestine Lumen': 0.20614860214699238, 'Larynx': 0.21985, 'Liver': 3.4575, 'Lung': 115.47, 'Lung (Deflated)': 80.94, 'Lung (Inflated)': 115.47, 'Lymph': 0.0, 'Lymphnode': 14.391, 'Mandible': 27.2765, 'Medulla Oblongata': 5.7645, 'Meniscus': 0.21985, 'Midbrain': 2.762957751445371, 'Mucous Membrane': 3.356379961658642, 'Muscle': 3.356379961658642, 'Nerve': 6.6175, 'Ovary': 3.91625, 'Pancreas': 4.7751, 'Penis': 3.51, 'Pharynx': 0.00978598664525, 'Pineal Body': 7.746, 'Placenta': 3.371693064615947, 'Pons': 2.762957751445371, 'Prostate': 4.65955, 'SAT (Subcutaneous Fat)': 3.5, 'Salivary Gland': 4.65955, 'Scalp': 10.579, 'Seminal vesicle': 4.65955, 'Skin': 10.579, 'Skull': 27.2765, 'Skull (Cancellous)': 20.45793823745892, 'Skull (Cortical)': 27.2765, 'Small Intestine': 2.87145, 'Small Intestine Lumen': 0.20614860214699238, 'Spinal Cord': 2.762957751445371, 'Spleen': 1.6763127970153082, 'Stomach': 2.87145, 'Stomach Lumen': 0.20614860214699238, 'Tendon\\Ligament': 6.438291282094524, 'Testis': 0.8212601132195042, 'Thalamus': 0.5187223387847191, 'Thymus': 5.434690996344573, 'Thyroid Gland': 7.746, 'Tongue': 1.4187788745827103, 'Tooth': 32.222, 'Tooth (Dentine)': 25.555555, 'Tooth (Enamel)': 38.88889, 'Trachea': 0.21985, 'Trachea Lumen': 0.00978598664525, 'Ureter\\Urethra': 3.51, 'Urinary Bladder Wall': 2.87145, 'Urine': 0.022574095509546616, 'Uterus': 4.05785, 'Vagina': 2.87145, 'Vertebrae': 27.2765, 'Water': 0.0126642180115}
# values from https://itis.swiss/virtual-population/tissue-properties/database/acoustic-properties/acoustic-non-linearity/, accessed January 2026
nonlinearity_parameter = {'Background': np.nan, 'Adrenal Gland': np.nan, 'Air': np.nan, 'Bile': 6.0, 'Blood': 6.11, 'Blood Plasma': 5.74, 'Blood Serum': np.nan, 'Blood Vessel Wall': np.nan, 'Bone (Cancellous)': np.nan, 'Bone (Cortical)': np.nan, 'Bone Marrow (Red)': np.nan, 'Bone Marrow (Yellow)': np.nan, 'Brain': 6.72, 'Brain (Grey Matter)': np.nan, 'Brain (White Matter)': np.nan, 'Breast Fat': 10.07, 'Breast Gland': 9.63, 'Bronchi': np.nan, 'Bronchi lumen': np.nan, 'Cartilage': np.nan, 'Cerebellum': np.nan, 'Cerebrospinal Fluid': np.nan, 'Cervix': np.nan, 'Commissura Anterior': np.nan, 'Commissura Posterior': np.nan, 'Connective Tissue': np.nan, 'Diaphragm': 7.17, 'Ductus Deferens': np.nan, 'Dura': np.nan, 'Epididymis': np.nan, 'Esophagus': np.nan, 'Esophagus Lumen': np.nan, 'Extracellular Fluids': np.nan, 'Eye (Aqueous Humor)': np.nan, 'Eye (Choroid)': 6.11, 'Eye (Ciliary Body)': 7.17, 'Eye (Cornea)': np.nan, 'Eye (Iris)': 7.17, 'Eye (Lens)': np.nan, 'Eye (Retina)': np.nan, 'Eye (Sclera)': np.nan, 'Eye (Vitreous Humor)': np.nan, 'Eye Lens (Cortex)': np.nan, 'Eye Lens (Nucleus)': np.nan, 'Fat': 10.07, 'Fat (Average Infiltrated)': 10.07, 'Fat (Not Infiltrated)': 10.07, 'Gallbladder': 6.22, 'Heart Lumen': 6.11, 'Heart Muscle': 6.76, 'Hippocampus': np.nan, 'Hypophysis': np.nan, 'Hypothalamus': np.nan, 'Intervertebral Disc': np.nan, 'Kidney': 7.44, 'Kidney (Cortex)': 7.44, 'Kidney (Medulla)': 7.44, 'Large Intestine': np.nan, 'Large Intestine Lumen': 6.06, 'Larynx': np.nan, 'Liver': 7.28, 'Lung': np.nan, 'Lung (Deflated)': np.nan, 'Lung (Inflated)': np.nan, 'Lymph': np.nan, 'Lymphnode': 8.21, 'Mandible': np.nan, 'Medulla Oblongata': 6.72, 'Meniscus': np.nan, 'Midbrain': 6.72, 'Mucous Membrane': np.nan, 'Muscle': 7.17, 'Nerve': np.nan, 'Ovary': np.nan, 'Pancreas': np.nan, 'Penis': np.nan, 'Pharynx': np.nan, 'Pineal Body': np.nan, 'Placenta': np.nan, 'Pons': 6.72, 'Prostate': np.nan, 'SAT (Subcutaneous Fat)': np.nan, 'Salivary Gland': np.nan, 'Scalp': np.nan, 'Seminal vesicle': np.nan, 'Skin': np.nan, 'Skull': np.nan, 'Skull (Cancellous)': np.nan, 'Skull (Cortical)': np.nan, 'Small Intestine': np.nan, 'Small Intestine Lumen': 6.06, 'Spinal Cord': np.nan, 'Spleen': 7.62, 'Stomach': np.nan, 'Stomach Lumen': 6.06, 'Tendon\\Ligament': np.nan, 'Testis': np.nan, 'Thalamus': np.nan, 'Thymus': 9.14, 'Thyroid Gland': np.nan, 'Tongue': 7.17, 'Tooth': np.nan, 'Tooth (Dentine)': np.nan, 'Tooth (Enamel)': np.nan, 'Trachea': np.nan, 'Trachea Lumen': np.nan, 'Ureter\\Urethra': np.nan, 'Urinary Bladder Wall': np.nan, 'Urine': 6.14, 'Uterus': np.nan, 'Vagina': np.nan, 'Vertebrae': np.nan, 'Water': 4.96}

# dictionary to obtain itis labels from simnibs labels
simnibs_to_itis = {
    'Background': 'Background',
    'White-Matter': 'Brain (White Matter)',
    'Gray-Matter': 'Brain (Grey Matter)',
    'CSF': 'Cerebrospinal Fluid',
    'Bone': 'Bone (Cortical)',
    'Scalp': 'Scalp',
    'Eye_balls': 'Eye (Vitreous Humor)',
    'Compact_bone': 'Skull (Cortical)',
    'Spongy_bone': 'Skull (Cancellous)',
    'Blood': 'Blood',
    'Muscle': 'Muscle',
    'Saline_or_gel': 'Water'
}

# dictionary used to remove some of the tissues to reduce the number of labels in sim4life segmentation (mainly done because we don't have parameter values for each tissue)
tissue_conversion_dict = {'Ventricles':'Cerebrospinal_fluid', 'Cerebellum_white_matter':'Cerebrum_white_matter', 'Cerebellum_grey_matter':'Cerebrum_grey_matter', 'Caudate_nucleus':'Cerebrum_grey_matter', 'Putamen':'Cerebrum_grey_matter', 'Globus_pallidus':'Cerebrum_grey_matter', 'Brainstem':'Cerebrum_white_matter', 'Amygdala':'Cerebrum_grey_matter', 'Nucleus_accumbens':'Cerebrum_grey_matter', 'Muscle_ocular':'Muscle', 'Vein':'Artery', 'Nerve_cranial_II_optic':'Cerebrum_white_matter', 'Submandibular_gland':'Parotid_gland', 'Sublingual_gland':'Parotid_gland', 'Tendon_temporalis':'Tendon_galea_aponeurotica', 'Nasal_septum':'Cartilage', 'Vertebrae_cancellous':'Vertebrae_cortical'}

# dictionary used to obtain itis labels from sim4life labels
sim4life_to_itis = {'Cerebrum_white_matter':'Brain (White Matter)',
'Muscle':'Muscle',
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
    nib.save(label_field, os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs.nii.gz'))

    # save the corresponding label file
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_labels.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{tissue}\n')

    # save the corresponding LUT file
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_LUT.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            old_label = label_dict[tissue]
            f.write(f'{idx}\t{LUT[old_label]['name']}\t{LUT[old_label]['R']}\t{LUT[old_label]['G']}\t{LUT[old_label]['B']}\t{LUT[old_label]['A']}\n')
    
    # save the corresponding acoustic parameters
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_density.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{density[simnibs_to_itis[tissue]]}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_speed_of_sound.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{speed_of_sound[simnibs_to_itis[tissue]]}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_attenuation_constant.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{attenuation_constant[simnibs_to_itis[tissue]]}\n')
            
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/simnibs_nonlinearity_parameter.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            f.write(f'{idx},{nonlinearity_parameter[simnibs_to_itis[tissue]]}\n')
    
    return


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
    nib.save(label_field, os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life.nii.gz'))
    
    # save the corresponding labels file
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_labels.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx},{tissue}\n')
    
    # save the corresponding LUT file
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_LUT.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            old_label = label_dict[tissue]
            tissue = LUT[old_label]['name']
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx}\t{tissue}\t{LUT[old_label]['R']}\t{LUT[old_label]['G']}\t{LUT[old_label]['B']}\t{LUT[old_label]['A']}\n')
    
    # save the corresponding acoustic parameters
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_density.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx},{density[tissue]}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_speed_of_sound.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx},{speed_of_sound[tissue]}\n')

    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_attenuation_constant.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx},{attenuation_constant[tissue]}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/sim4life_nonlinearity_parameter.txt'), 'w') as f:
        for idx, tissue in enumerate(final_tissues):
            if tissue in sim4life_to_itis.keys():
                tissue = sim4life_to_itis[tissue]
            f.write(f'{idx},{nonlinearity_parameter[tissue]}\n')
    
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

    # save the full ITIS acoustic parameters files
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/ITIS_density.txt'), 'w') as f:
        for key,val in density.items():
            f.write(f'{key},{val}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/ITIS_speed_of_sound.txt'), 'w') as f:
        for key,val in speed_of_sound.items():
            f.write(f'{key},{val}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/ITIS_attenuation_constant.txt'), 'w') as f:
        for key,val in attenuation_constant.items():
            f.write(f'{key},{val}\n')
    
    with open(os.path.join(subj_dir, 'tissue_labels/acoustic/ITIS_nonlinearity_parameter.txt'), 'w') as f:
        for key,val in nonlinearity_parameter.items():
            f.write(f'{key},{val}\n')
