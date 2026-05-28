#!/bin/bash
source /scripts/source_env.sh

check_step() {
    local exit_code=$1    # The exit code of the command you just ran
    local description=$2  # Text description
    local log_file=$3     # Where the logs are stored
    local cleanup_path=$4 # Optional, path to remove if error is detected

    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] $description failed! (Exit Code: $exit_code)"
        echo "Check log file for more info: $log_file"

        # Check if the cleanup path was provided and if it actually exists
        if [ -n "$cleanup_path" ] && [ -d "$cleanup_path" ]; then
            echo "Cleaning up directory: $cleanup_path"
            rm -rf "$cleanup_path"
        fi

        echo
        exit 1
    fi
}

######################################

# define subject name and file paths
subj=""
T1_file=""
T2_file=""

# true if you want to use T2 images (false if you dont have it)
USE_T2=false

# true if the T2 image is FLAIR
IS_FLAIR=false

# true if T2 is already registered to T1
USE_SKIPREGISTER_T2=false

# true if anatomical images do not contain neck
USE_NONECK=false

# number of threads to use for freesurfer reconstruction
N_THREADS=8


# Function to display usage message
usage() {
  echo "Usage: $0 [--subject SUBJECT] [--T1 T1_PATH] [--T2 T2_PATH] [--FLAIR] [--skip-register-T2] [--no-neck] [--threads N_THREADS]"
  echo ""
  echo "Options:"
  echo "  --subject SUBJECT          Set the subject name (mandatory)"
  echo "  --T1 T1_PATH               Path to the T1 image (mandatory)"
  echo "  --T2 T2_PATH               Path to the T2 image (optional)"
  echo "  --FLAIR                    If the T2 image is FLAIR, set this flag"
  echo "  --skip-register-T2         Skip the registration of the T2 image to T1"
  echo "  --no-neck                  If the anatomical images do not contain neck, set this flag"
  echo "  --threads N_THREADS        Number of threads to use for freesurfer reconstruction (default: 8)"
  exit 1
}


# If no arguments are provided, print usage
if [[ $# -eq 0 ]]; then
  usage
fi


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) subj="$2"; shift 2 ;;
    --T1) T1_file="$2"; shift 2 ;;
    --T2) T2_file="$2"; USE_T2=true; shift 2 ;;  # If --T2 is passed, set USE_T2 to true
    --FLAIR) IS_FLAIR=true; shift ;;  # Set IS_FLAIR to true if --FLAIR is passed
    --skip-register-T2) USE_SKIPREGISTER_T2=true; shift ;;
    --no-neck) USE_NONECK=true; shift ;;
    --threads) N_THREADS="$2"; shift 2 ;;
    *) usage ;;  # If an unrecognized argument is encountered, show usage
  esac
done

# Check if mandatory arguments are provided
if [[ -z "$subj" || -z "$T1_file" ]]; then
  echo "Error: --subject and --T1-file are mandatory."
  echo ""
  usage
fi

# Check if --FLAIR or --skip-register-T2 is provided without --T2
if [[ "$IS_FLAIR" == true || "$USE_SKIPREGISTER_T2" == true ]]; then
  if [[ -z "$T2_file" || "$USE_T2" == false ]]; then
    echo "Error: --T2 must be provided if --FLAIR or --skip-register-T2 is specified."
    echo ""
    usage
  fi
fi

##########################################

#### handling optional arguments ####

# optional arguments to freesurfer
fs_args=()
simnibs_args=()

if [ "$USE_T2" = true ]; then
	if [ "$IS_FLAIR" = true ]; then
	    fs_args+=(-FLAIR "$T2_file" -FLAIRpial)
	else
	    fs_args+=(-T2 "$T2_file" -T2pial)
	fi
	simnibs_args+=("$T2_file")
        if [ "$USE_SKIPREGISTER_T2" = true ]; then
            simnibs_args+=(--skipregisterT2)
        fi
fi

if [ "$USE_NONECK" = true ]; then
    simnibs_args+=(--noneck)
fi
######################################
LOG_FILE="/SUBJECTS/$subj/reconstruction_logs/reconstruction_log.txt"
######################################
# check whether GPUs are available or not
if nvidia-smi -L &> /dev/null; then
    echo "GPU check inside reconstruction container: OK" >> "$LOG_FILE"
    export USE_GPU=true
else
    echo "GPU check inside reconstruction container: FAILED (Falling back to CPU)" >> "$LOG_FILE"
    export USE_GPU=false
fi
######################################
# Define a function to clean up temporary files/links
cleanup() {
    # Check if the path exists AND is a symlink (-L)
    if [ -L "$SUBJECTS_DIR"/"$subj" ]; then
        rm "$SUBJECTS_DIR"/"$subj"
    fi
}

# run the cleanup function on EXIT
trap cleanup EXIT
######################################
# --- FREESURFER PATH HANDLING ---
# If recon-all was previously completed and moved, subsequent FreeSurfer commands 
# (like segment_subregions) might fail because they expect the subject folder to be in $SUBJECTS_DIR.
# This creates a temporary symlink if the folder was already moved to your custom directory.
if [ ! -d "$SUBJECTS_DIR"/"$subj" ] && [ -f /SUBJECTS/"$subj"/reconstruction_logs/freesurfer.txt ]; then
        ln -s /SUBJECTS/"$subj"/freesurfer "$SUBJECTS_DIR"/"$subj"
fi
######################################

# Freesurfer Recon-All
if [ ! -f /SUBJECTS/"$subj"/reconstruction_logs/freesurfer.txt ]; then
        echo "Running Freesurfer reconstruction..."

        # run recon all
        start=$(date +%s)
        recon-all -subject "$subj" -i "$T1_file" "${fs_args[@]}" -all -threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer.txt 2>&1
        check_step $? "Freesurfer reconstruction" "$subj"/reconstruction_logs/freesurfer.txt
        end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        rm -rf /SUBJECTS/"$subj"/freesurfer # remove if it already exists
        mv "$SUBJECTS_DIR"/"$subj" /SUBJECTS/"$subj"/freesurfer
        ln -s /SUBJECTS/"$subj"/freesurfer "$SUBJECTS_DIR"/"$subj"
	cp $FREESURFER_HOME/FreeSurferColorLUT.txt /SUBJECTS/"$subj"/freesurfer/FreeSurferColorLUT.txt
	cp -r /home/Schaefer2018_LocalGlobal/Parcellations/project_to_individual /SUBJECTS/"$subj"/freesurfer/Schaefer_LUT

        echo "Freesurfer reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
        echo "Freesurfer reconstruction log detected, skipping recon-all..." | tee -a "$LOG_FILE"
fi
echo

# BEM Surfaces
if [ ! -f /SUBJECTS/"$subj"/reconstruction_logs/mne.txt ]; then
        echo "Running BEM surfaces reconstruction..."

        # make bem surfaces
        start=$(date +%s)
        micromamba run -n neuro python /scripts/make_bem_surfaces.py --subject "$subj" --subjects_dir "$SUBJECTS_DIR"  > /SUBJECTS/"$subj"/reconstruction_logs/mne.txt 2>&1
        check_step $? "Make BEM surfaces" "$subj"/reconstruction_logs/mne.txt
        end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "MNE BEM surfaces reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
        echo "MNE BEM surfaces log detected, skipping MNE BEM reconstruction..." | tee -a "$LOG_FILE"
fi
echo

# Schaefer Atlases
if [ ! -f /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt ]; then
        echo "Registering user to Schaefer atlases..."
        start=$(date +%s)
	echo "------------------------------------------------------------------------------------" > /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt
	for n_parcels in {100..1000..100}; do
		ATLAS_NAME="Schaefer2018_${n_parcels}Parcels_17Networks_order"

		mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject "$subj" --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/lh.${ATLAS_NAME}.annot --tval "$SUBJECTS_DIR"/"$subj"/label/lh.${ATLAS_NAME}.annot >> /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt 2>&1
	        check_step $? "Registration of Schaefer $n_parcels atlas to left hemisphere" "$subj"/reconstruction_logs/schaefer.txt
		mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject "$subj" --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/rh.${ATLAS_NAME}.annot --tval "$SUBJECTS_DIR"/"$subj"/label/rh.${ATLAS_NAME}.annot >> /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt 2>&1
		check_step $? "Registration of Schaefer $n_parcels atlas to right hemisphere" "$subj"/reconstruction_logs/schaefer.txt
		mri_aparc2aseg --s "$subj" --o "$SUBJECTS_DIR"/"$subj"/mri/schaefer${n_parcels}_aparc+aseg.mgz --annot "$ATLAS_NAME" >> /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt 2>&1
		check_step $? "Creation of Schaefer $n_parcels volumetric atlas" "$subj"/reconstruction_logs/schaefer.txt
		echo "------------------------------------------------------------------------------------" >> /SUBJECTS/"$subj"/reconstruction_logs/schaefer.txt
        done
        end=$(date +%s)
        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        echo "Registration to Schaefer atlases completed in ${minutes} minutes." | tee -a "$LOG_FILE"
else
        echo "Schaefer atlases log detected, skipping registration..." | tee -a "$LOG_FILE"
fi
echo

# Subcortical Stream
if [ ! -f /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt ]; then
        echo "Running Freesurfer subcortical stream..."

        start=$(date +%s)

        echo -e "Starting thalamus subsegmentation..." > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        segment_subregions thalamus --cross "$subj" --threads "$N_THREADS" >> /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        check_step $? "Freesurfer thalamus reconstruction" "$subj"/reconstruction_logs/freesurfer_subcortical.txt
        
        echo -e "\n\n\n\n\n\n\n\n\n\nStarting hippo-amygdala subsegmentation..." >> /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        segment_subregions hippo-amygdala --cross "$subj" --threads "$N_THREADS" >> /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        check_step $? "Freesurfer hippo-amygdala reconstruction" "$subj"/reconstruction_logs/freesurfer_subcortical.txt
        
        echo -e "\n\n\n\n\n\n\n\n\n\nStarting brainstem subsegmentation..." >> /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        segment_subregions brainstem --cross "$subj" --threads "$N_THREADS" >> /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_subcortical.txt 2>&1
        check_step $? "Freesurfer brainstem reconstruction" "$subj"/reconstruction_logs/freesurfer_subcortical.txt

        end=$(date +%s)
        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        echo "Freesurfer subcortical stream completed in ${minutes} minutes." | tee -a "$LOG_FILE"
else
        echo "Subcortical stream log detected, skipping subcortical segmentation..." | tee -a "$LOG_FILE"
fi

# if not already done, run simnibs charm
if [ ! -f "$subj"/reconstruction_logs/simnibs_charm.txt ]; then
	echo "Running Simnibs charm reconstruction..."

	start=$(date +%s)
	cd /home/simnibs_reconstructions
	/root/SimNIBS-4.5/bin/charm subject "$T1_file" "${simnibs_args[@]}" --forcerun --fs-dir /SUBJECTS/"$subj"/freesurfer --forcesform > /SUBJECTS/"$subj"/reconstruction_logs/simnibs_charm.txt 2>&1
        check_step $? "Simnibs charm reconstruction" "$subj"/reconstruction_logs/simnibs_charm.txt
	cd /
	/root/SimNIBS-4.5/bin/simnibs_python /scripts/extract_charm_surf.py --charm_dir "/home/simnibs_reconstructions/m2m_subject/" >> /SUBJECTS/"$subj"/reconstruction_logs/simnibs_charm.txt 2>&1
	check_step $? "Simnibs charm surface extraction" "$subj"/reconstruction_logs/simnibs_charm.txt
        cp /scripts/simnibs_conductivities.txt /home/simnibs_reconstructions/m2m_subject/conductivities.txt
        cp /scripts/simnibs_labels.txt /home/simnibs_reconstructions/m2m_subject/labels.txt

        rm -rf /SUBJECTS/"$subj"/simnibs_charm # remove if it already exists
	mv /home/simnibs_reconstructions/m2m_subject /SUBJECTS/"$subj"/simnibs_charm
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Simnibs charm reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
	echo "Simnibs charm log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo


# if not already done, run fsl first
if [ ! -f "$subj"/reconstruction_logs/fsl_first.txt ]; then
	echo "Running FSL first reconstruction..."
        rm -rf /SUBJECTS/"$subj"/fsl_first # remove if it already exists
	mkdir /SUBJECTS/"$subj"/fsl_first
	cp "$T1_file" /SUBJECTS/"$subj"/fsl_first/T1.nii.gz

	start=$(date +%s)
	# bias field correct image before running first
	micromamba run -n neuro python /scripts/bias_correct.py "$T1_file" /SUBJECTS/"$subj"/fsl_first/T1.nii.gz > /SUBJECTS/"$subj"/reconstruction_logs/fsl_first.txt 2>&1
	check_step $? "N4 bias correction" "$subj"/reconstruction_logs/fsl_first.txt /SUBJECTS/"$subj"/fsl_first
	/scripts/run_first_all_sequential -i /SUBJECTS/"$subj"/fsl_first/T1.nii.gz -o /SUBJECTS/"$subj"/fsl_first/FSL -v >> /SUBJECTS/"$subj"/reconstruction_logs/fsl_first.txt 2>&1
	check_step $? "FSL first reconstruction" "$subj"/reconstruction_logs/fsl_first.txt /SUBJECTS/"$subj"/fsl_first
        end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "FSL first reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
	echo "FSL first log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, run synthstrip
if [ ! -f "$subj"/reconstruction_logs/synthstrip.txt ]; then
	echo "Running SynthStrip reconstruction..."
        rm -rf /SUBJECTS/"$subj"/synthstrip # remove if it already exists
        mkdir /SUBJECTS/"$subj"/synthstrip

	synt_flag=()
	if [ "$USE_GPU" = true ] ; then
	     synt_flag="--gpu"
	fi

	start=$(date +%s)
	mri_synthstrip -i "$T1_file" -o /SUBJECTS/"$subj"/synthstrip/T1_stripped.nii.gz -m /SUBJECTS/"$subj"/synthstrip/T1_stripped_mask.nii.gz "${synt_flag[@]}" > /SUBJECTS/"$subj"/reconstruction_logs/synthstrip.txt 2>&1
	check_step $? "SynthStrip reconstruction" "$subj"/reconstruction_logs/synthstrip.txt /SUBJECTS/"$subj"/synthstrip
	mri_synthstrip -i "$T1_file" -o /SUBJECTS/"$subj"/synthstrip/T1_noCSF_stripped.nii.gz -m /SUBJECTS/"$subj"/synthstrip/T1_noCSF_stripped_mask.nii.gz "${synt_flag[@]}" --no-csf >> /SUBJECTS/"$subj"/reconstruction_logs/synthstrip.txt 2>&1
	check_step $? "SynthStrip no-CSF reconstruction" "$subj"/reconstruction_logs/synthstrip.txt /SUBJECTS/"$subj"/synthstrip
        end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "SynthStrip reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
	echo "SynthStrip log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, run ANTs registration to align cerebellum surface
if [ ! -f "$subj"/reconstruction_logs/cerebellum.txt ]; then
	echo "Running Cerebellum reconstruction..."
        rm -rf /SUBJECTS/"$subj"/cerebellum # remove if it already exists
        mkdir /SUBJECTS/"$subj"/cerebellum
	cp /home/cerebellum_template/Cerebellar_Regions.csv /SUBJECTS/"$subj"/cerebellum/LABELS.csv

	start=$(date +%s)
	micromamba run -n neuro python /scripts/run_cereb_pipeline.py --subject_dir /SUBJECTS/"$subj"/ --template_dir "/home/cerebellum_template/" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/cerebellum.txt 2>&1
	check_step $? "Cerebellum reconstruction" "$subj"/reconstruction_logs/cerebellum.txt /SUBJECTS/"$subj"/cerebellum
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Cerebellum reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
	echo "Cerebellum reconstruction log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, run ANTs registration to align bigbrain scans
if [ ! -f "$subj"/reconstruction_logs/bigbrain.txt ]; then
	echo "Running bigbrain registration..."
        rm -rf /SUBJECTS/"$subj"/bigbrain # remove if it already exists
        mkdir /SUBJECTS/"$subj"/bigbrain

	start=$(date +%s)
	micromamba run -n neuro python /scripts/run_bigbrain_pipeline.py --subject_dir /SUBJECTS/"$subj"/ --template_dir "/home/bigbrain_scans/" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/bigbrain.txt 2>&1
	check_step $? "BigBrain registration" "$subj"/reconstruction_logs/bigbrain.txt /SUBJECTS/"$subj"/bigbrain
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "BigBrain registration completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
	echo "BigBrain registration log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# convert all relevant meshes to world space and gather them in one place
if [ ! -f "$subj"/reconstruction_logs/surfaces.txt ]; then
	echo "Converting all relevant meshes to world space and gathering them in one place..."
        rm -rf /SUBJECTS/"$subj"/surfaces # remove if it already exists
        mkdir /SUBJECTS/"$subj"/surfaces

	start=$(date +%s)
	micromamba run -n neuro python /scripts/gather_surfaces.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/surfaces.txt 2>&1
	check_step $? "Surfaces gathering" "$subj"/reconstruction_logs/surfaces.txt /SUBJECTS/"$subj"/surfaces
	end=$(date +%s)

	duration=$(( end - start ))
	minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

	echo "Surface gathering completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
        echo "Surfaces log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make atlases from various tools' segmentations
if [ ! -f "$subj"/reconstruction_logs/atlas.txt ]; then
        echo "Making atlases from various tools' segmentations..."
        rm -rf /SUBJECTS/"$subj"/atlas # remove if it already exists
        mkdir /SUBJECTS/"$subj"/atlas

        start=$(date +%s)
        micromamba run -n neuro python /scripts/make_atlas.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/atlas.txt 2>&1
	check_step $? "Atlas generation" "$subj"/reconstruction_logs/atlas.txt /SUBJECTS/"$subj"/atlas
        end=$(date +%s)

        duration=$(( end - start ))
	minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "Atlases completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
        echo "Atlases log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make electrical label fields using simnibs reconstruction and sim4life (optional)
if [ ! -f "$subj"/reconstruction_logs/electrical_labelfields.txt ]; then
        echo "Making electrical label fields using simnibs reconstruction and Sim4Life (optional)..."
        rm -rf /SUBJECTS/"$subj"/tissue_labels/electrical # remove if it already exists
        mkdir -p /SUBJECTS/"$subj"/tissue_labels/electrical
        cp /scripts/simnibs_mesher_parameters.txt /SUBJECTS/"$subj"/tissue_labels/electrical/
        cp /scripts/sim4life_mesher_parameters.txt /SUBJECTS/"$subj"/tissue_labels/electrical/

        start=$(date +%s)
        micromamba run -n neuro python /scripts/gather_electrical_labelfields.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/electrical_labelfields.txt 2>&1
        check_step $? "Electrical label fields generation" "$subj"/reconstruction_logs/electrical_labelfields.txt /SUBJECTS/"$subj"/tissue_labels/electrical
        end=$(date +%s)

        duration=$(( end - start ))

        echo "Electrical label fields completed in ${duration} seconds." | tee -a "$LOG_FILE"
else
        echo "Electrical label fields log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make acoustic label fields using simnibs reconstruction and sim4life (optional)
if [ ! -f "$subj"/reconstruction_logs/acoustic_labelfields.txt ]; then
        echo "Making acoustic label fields using simnibs reconstruction and Sim4Life (optional)..."
        rm -rf /SUBJECTS/"$subj"/tissue_labels/acoustic # remove if it already exists
        mkdir -p /SUBJECTS/"$subj"/tissue_labels/acoustic

        start=$(date +%s)
        micromamba run -n neuro python /scripts/gather_acoustic_labelfields.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/acoustic_labelfields.txt
        check_step $? "Acoustic label fields generation" "$subj"/reconstruction_logs/acoustic_labelfields.txt /SUBJECTS/"$subj"/tissue_labels/acoustic
        end=$(date +%s)

        duration=$(( end - start ))

        echo "Acoustic label fields completed in ${duration} seconds." | tee -a "$LOG_FILE"
else
        echo "Acoustic label fields log detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo
