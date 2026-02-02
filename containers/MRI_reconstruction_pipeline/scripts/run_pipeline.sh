#!/bin/bash
source /scripts/source_env.sh

check_step() {
    local exit_code=$1    # The exit code of the command you just ran
    local description=$2  # Text description
    local log_file=$3     # Where the logs are stored

    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] $description failed! (Exit Code: $exit_code)"
        echo "Check log file for more info: $log_file"
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

# if not already done, run freesurfer recon
if [ ! -d /SUBJECTS/"$subj"/freesurfer ]; then
	echo "Running Freesurfer reconstruction..."

	# run recon-all
	start=$(date +%s)
	recon-all -subject "$subj" -i "$T1_file" "${fs_args[@]}" -all -threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer.txt 2>&1
	check_step $? "Freesurfer reconstruction" "$subj"/reconstruction_logs/freesurfer.txt
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Freesurfer reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"

	echo "Running MNE bem surfaces reconstruction..."
	# make bem surfaces
	start=$(date +%s)
	micromamba run -n neuro mne watershed_bem -s "$subj"  > /SUBJECTS/"$subj"/reconstruction_logs/mne.txt 2>&1
        check_step $? "MNE watershed bem" "$subj"/reconstruction_logs/mne.txt
	micromamba run -n neuro mne make_scalp_surfaces --force --overwrite --subject "$subj" >> /SUBJECTS/"$subj"/reconstruction_logs/mne.txt 2>&1
        check_step $? "MNE make scalp surfaces" "$subj"/reconstruction_logs/mne.txt
	end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "MNE bem surfaces reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"

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

	echo "Running Freesurfer subcortical stream..."
	# run subcortical stream
	start=$(date +%s)
	segment_subregions thalamus --cross "$subj" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_thalamus.txt 2>&1
	check_step $? "Freesurfer thalamus reconstruction" "$subj"/reconstruction_logs/freesurfer_thalamus.txt
	segment_subregions hippo-amygdala --cross "$subj" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_hippoamygdala.txt 2>&1
        check_step $? "Freesurfer hippo-amygdala reconstruction" "$subj"/reconstruction_logs/freesurfer_hippoamygdala.txt
	segment_subregions brainstem --cross "$subj" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/freesurfer_brainstem.txt 2>&1
        check_step $? "Freesurfer brainstem reconstruction" "$subj"/reconstruction_logs/freesurfer_brainstem.txt
	end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))

        echo "Freesurfer subcortical stream completed in ${minutes} minutes." | tee -a "$LOG_FILE"

	mv "$SUBJECTS_DIR"/"$subj" /SUBJECTS/"$subj"/freesurfer
	cp $FREESURFER_HOME/FreeSurferColorLUT.txt /SUBJECTS/"$subj"/freesurfer/FreeSurferColorLUT.txt
	cp -r /home/Schaefer2018_LocalGlobal/Parcellations/project_to_individual /SUBJECTS/"$subj"/freesurfer/Schaefer_LUT
else
        echo "Freesurfer reconstruction detected in subject's folder, skipping all associated steps..." | tee -a "$LOG_FILE"
fi
echo


# if not already done, run simnibs charm
if [ ! -d /SUBJECTS/"$subj"/simnibs_charm ]; then
	echo "Running Simnibs charm reconstruction..."

	start=$(date +%s)
	cd /home/simnibs_reconstructions
	/root/SimNIBS-4.5/bin/charm subject "$T1_file" "${simnibs_args[@]}" --forcerun --fs-dir /SUBJECTS/"$subj"/freesurfer --forcesform > /SUBJECTS/"$subj"/reconstruction_logs/simnibs_charm.txt 2>&1
        check_step $? "Simnibs charm reconstruction" "$subj"/reconstruction_logs/simnibs_charm.txt
	cd /
	/root/SimNIBS-4.5/bin/simnibs_python /scripts/extract_charm_surf.py --charm_dir "/home/simnibs_reconstructions/m2m_subject/" >> /SUBJECTS/"$subj"/reconstruction_logs/simnibs_charm.txt 2>&1
	check_step $? "Simnibs charm surface extraction" "$subj"/reconstruction_logs/simnibs_charm.txt
	mv /home/simnibs_reconstructions/m2m_subject /SUBJECTS/"$subj"/simnibs_charm
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Simnibs charm reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
	echo "Simnibs charm reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo


# if not already done, run fsl first
if [ ! -d /SUBJECTS/"$subj"/fsl_first ]; then
	echo "Running FSL first reconstruction..."
	mkdir /SUBJECTS/"$subj"/fsl_first
	cp "$T1_file" /SUBJECTS/"$subj"/fsl_first/T1.nii.gz

	start=$(date +%s)
	# bias field correct image before running first
	micromamba run -n neuro python /scripts/bias_correct.py "$T1_file" /SUBJECTS/"$subj"/fsl_first/T1.nii.gz > /SUBJECTS/"$subj"/reconstruction_logs/fsl_first.txt 2>&1
	check_step $? "N4 bias correction" "$subj"/reconstruction_logs/fsl_first.txt
	/scripts/run_first_all_sequential -i /SUBJECTS/"$subj"/fsl_first/T1.nii.gz -o /SUBJECTS/"$subj"/fsl_first/FSL -v >> /SUBJECTS/"$subj"/reconstruction_logs/fsl_first.txt 2>&1
	check_step $? "FSL first reconstruction" "$subj"/reconstruction_logs/fsl_first.txt
        end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "FSL first reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
	echo "FSL first reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, run synthstrip
if [ ! -d /SUBJECTS/"$subj"/synthstrip ]; then
	echo "Running SynthStrip reconstruction..."
        mkdir /SUBJECTS/"$subj"/synthstrip

	synt_flag=()
	if [ "$USE_GPU" = true ] ; then
	     synt_flag="--gpu"
	fi

	start=$(date +%s)
	mri_synthstrip -i "$T1_file" -o /SUBJECTS/"$subj"/synthstrip/T1_stripped.nii.gz -m /SUBJECTS/"$subj"/synthstrip/T1_stripped_mask.nii.gz "${synt_flag[@]}" > /SUBJECTS/"$subj"/reconstruction_logs/synthstrip.txt 2>&1
	check_step $? "SynthStrip reconstruction" "$subj"/reconstruction_logs/synthstrip.txt
	mri_synthstrip -i "$T1_file" -o /SUBJECTS/"$subj"/synthstrip/T1_noCSF_stripped.nii.gz -m /SUBJECTS/"$subj"/synthstrip/T1_noCSF_stripped_mask.nii.gz "${synt_flag[@]}" --no-csf >> /SUBJECTS/"$subj"/reconstruction_logs/synthstrip.txt 2>&1
	check_step $? "SynthStrip no-CSF reconstruction" "$subj"/reconstruction_logs/synthstrip.txt
        end=$(date +%s)

        duration=$(( end - start ))
        minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "SynthStrip reconstruction completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
	echo "SynthStrip reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, run ANTs registration to align cerebellum surface
if [ ! -d /SUBJECTS/"$subj"/cerebellum ]; then
	echo "Running Cerebellum reconstruction..."
        mkdir /SUBJECTS/"$subj"/cerebellum
	cp /home/cerebellum_template/Cerebellar_Regions.csv /SUBJECTS/"$subj"/cerebellum/LABELS.csv

	start=$(date +%s)
	micromamba run -n neuro python /scripts/run_cereb_pipeline.py --subject_dir /SUBJECTS/"$subj"/ --template_dir "/home/cerebellum_template/" --threads "$N_THREADS" > /SUBJECTS/"$subj"/reconstruction_logs/cerebellum.txt 2>&1
	check_step $? "Cerebellum reconstruction" "$subj"/reconstruction_logs/cerebellum.txt
	end=$(date +%s)

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Cerebellum reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
	echo "Cerebellum reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# convert all relevant meshes to world space and gather them in one place
if [ ! -d /SUBJECTS/"$subj"/surfaces ]; then
	echo "Converting all relevant meshes to world space and gathering them in one place..."
        mkdir /SUBJECTS/"$subj"/surfaces

	start=$(date +%s)
	micromamba run -n neuro python /scripts/gather_surfaces.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/surfaces.txt 2>&1
	check_step $? "Surfaces gathering" "$subj"/reconstruction_logs/surfaces.txt
	end=$(date +%s)

	duration=$(( end - start ))
	minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

	echo "Surface gathering completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
        echo "Surfaces detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make atlases from various tools' segmentations
if [ ! -d /SUBJECTS/"$subj"/atlas ]; then
        echo "Making atlases from various tools' segmentations..."
        mkdir /SUBJECTS/"$subj"/atlas

        start=$(date +%s)
        micromamba run -n neuro python /scripts/make_atlas.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/atlas.txt 2>&1
	check_step $? "Atlas generation" "$subj"/reconstruction_logs/atlas.txt
        end=$(date +%s)

        duration=$(( end - start ))
	minutes=$(( duration / 60 ))
        seconds=$(( duration % 60 ))

        echo "Atlases completed in ${minutes} minutes and ${seconds} seconds." | tee -a "$LOG_FILE"
else
        echo "Atlases detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make electrical label fields using simnibs reconstruction and sim4life (optional)
if [ ! -d /SUBJECTS/"$subj"/tissue_labels/electrical ]; then
        echo "Making electrical label fields using simnibs reconstruction and Sim4Life (optional)..."
        mkdir -p /SUBJECTS/"$subj"/tissue_labels/electrical

        start=$(date +%s)
        micromamba run -n neuro python /scripts/gather_electrical_labelfields.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/electrical_labelfields.txt 2>&1
        check_step $? "Electrical label fields generation" "$subj"/reconstruction_logs/electrical_labelfields.txt
        end=$(date +%s)

        duration=$(( end - start ))

        echo "Electrical label fields completed in ${duration} seconds." | tee -a "$LOG_FILE"
else
        echo "Electrical label fields detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# make acoustic label fields using simnibs reconstruction and sim4life (optional)
if [ ! -d /SUBJECTS/"$subj"/tissue_labels/acoustic ]; then
        echo "Making acoustic label fields using simnibs reconstruction and Sim4Life (optional)..."
        mkdir -p /SUBJECTS/"$subj"/tissue_labels/acoustic

        start=$(date +%s)
        micromamba run -n neuro python /scripts/gather_acoustic_labelfields.py --subject_dir /SUBJECTS/"$subj"/ > /SUBJECTS/"$subj"/reconstruction_logs/acoustic_labelfields.txt
        check_step $? "Acoustic label fields generation" "$subj"/reconstruction_logs/acoustic_labelfields.txt
        end=$(date +%s)

        duration=$(( end - start ))

        echo "Acoustic label fields completed in ${duration} seconds." | tee -a "$LOG_FILE"
else
        echo "Acoustic label fields detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo
