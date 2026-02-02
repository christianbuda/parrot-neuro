#!/bin/bash

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

############## variables ################################
# MANDATORY: directory where subject data lives
SUBJECTS_DIR=""

# MANDATORY: subject ID
SUBJECT=""

## other pipeline variables

# define file paths
T1_PATH=""
T2_PATH=""

# tissue labels field reconstructed with Sim4Life
SIM4LIFE_VOL=""
SIM4LIFE_LUT=""
S4L_AVAILABLE=false

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

# whether or not to use GPUs for the tools that can use it
USE_GPU="--gpus all"

# help function
# Help function
usage() {
    echo "Usage: $0 -s SUBJECT -d SUBJECTS_DIR -t1 PATH_TO_T1 [OPTIONS]"
    echo ""
    echo "Mandatory:"
    echo "  -s, --subject                Subject ID"
    echo "  -d, --subjects-dir           directory where subject data lives"
    echo ""
    echo "Pipeline Options (Only used on first run):"
    echo "  -t1, --t1-path      Path to T1 nifti (Mandatory for new subjects)"
    echo "  -t2, --t2-path      Path to T2 nifti"
    echo "  --s4l-vol           Path to Sim4Life tissue labels field"
    echo "  --s4l-lut           Path to Sim4Life tissue labels LUT"
    echo "  --flair             Flag: T2 image is FLAIR"
    echo "  --skip-reg          Flag: Skip T2 registration"
    echo "  --no-neck           Flag: Anatomical images have no neck"
    echo "  --threads           Number of threads (Default: 8)"
    echo "  --no-gpu            Flag: don't use a GPU, even if available"
    echo "                      NOTE: this flag is not saved in the configuration file, so you need to specify it at every run."
    exit 1
}


####################################################################
# PARSE COMMAND LINE ARGUMENTS

# We store flags to check if the user provided them later
USER_PROVIDED_OPTS=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--subject)
            SUBJECT="$2"
            shift 2
            ;;
        -d|--subjects-dir)
            SUBJECTS_DIR="$2"
            shift 2
            ;;
        -t1|--t1-path)
            T1_PATH="$2"
            USER_PROVIDED_OPTS=true
            shift 2
            ;;
        -t2|--t2-path)
            T2_PATH="$2"
            USE_T2=true
            USER_PROVIDED_OPTS=true
            shift 2
            ;;
        --s4l-vol)
            SIM4LIFE_VOL="$2"
	    S4L_AVAILABLE=true
	    USER_PROVIDED_OPTS=true
            shift 2
            ;;
        --s4l-lut)
            SIM4LIFE_LUT="$2"
	    S4L_AVAILABLE=true
	    USER_PROVIDED_OPTS=true
            shift 2
            ;;
        --flair)
            IS_FLAIR=true
            USER_PROVIDED_OPTS=true
            shift
            ;;
        --skip-reg)
            USE_SKIPREGISTER_T2=true
            USER_PROVIDED_OPTS=true
            shift
            ;;
        --no-neck)
            USE_NONECK=true
            USER_PROVIDED_OPTS=true
            shift
            ;;
        --threads)
            N_THREADS="$2"
            shift 2
            ;;
	--no-gpu)
	    USE_GPU=""
	    shift
	    ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


#######################################################
# check mandatory variables

if [ -z "$SUBJECT" ]; then
    echo "ERROR: Subject ID (-s) is required."
    usage
fi

if [ -z "$SUBJECTS_DIR" ]; then
    echo "ERROR: Subjects directory (-d) is required."
    usage
fi

#############################################################

# Define Config File Path
CONFIG_FILE="${SUBJECTS_DIR}/${SUBJECT}/pipeline_config.env"

# Define internal destination paths immediately so they can be used in config
RAW_DIR="${SUBJECTS_DIR}/${SUBJECT}/raw"
DEST_T1="${RAW_DIR}/T1.nii.gz"
DEST_T2="${RAW_DIR}/T2.nii.gz"

if [ -f "$CONFIG_FILE" ]; then
    echo "Found existing configuration for $SUBJECT."

    # CHECK: Did the user try to change parameters on a locked subject?
    if [ "$USER_PROVIDED_OPTS" = true ]; then
        echo "----------------------------------------------------------------"
        echo "CRITICAL ERROR: Arguments provided for an existing subject."
        echo "----------------------------------------------------------------"
        echo "This subject has already been initialized. To ensure consistency,"
        echo "you cannot change pipeline parameters (like T1, T2, paths) via command line for a rerun."
        echo ""
        echo "To rerun this subject, use ONLY:"
        echo "  $0 -s $SUBJECT -d $SUBJECTS_DIR [optional --threads $N_THREADS] [optional --no-gpu]"
        echo ""
        echo "If you really want to change settings, you can delete the config file:"
        echo "  rm $CONFIG_FILE"
	echo "However, only do this if you know what you're doing. Otherwise, just delete the subject's folder and rerun the entire pipeline from scratch with the desired options"
        exit 1
    fi

    # LOAD: Source the config file to overwrite current variables
    echo "Loading parameters from $CONFIG_FILE..."
    source "$CONFIG_FILE"

    if [ ! -d "$RAW_DIR" ]; then
	echo "ERROR: Critical error: raw/ folder not found in ${SUBJECTS_DIR}/${SUBJECT}. Either regenerate it manually (if you know what you're doing), or rerun the reconstruction from scratch."
        exit 1
    fi

else
    echo "Initializing NEW subject: $SUBJECT"

    # VALIDATION: T1 is mandatory for new runs
    if [ -z "$T1_PATH" ]; then
        echo "ERROR: This is a new subject. You must provide at least a T1 path (-t1)."
        exit 1
    fi

    # Create directory structure
    mkdir -p "${SUBJECTS_DIR}/${SUBJECT}/raw"

    # Determine what T2 path to save in the config
    # We want to save the internal path, not the user provided source
    SAVED_T2_PATH=""
    if [ "$USE_T2" = true ]; then
        SAVED_T2_PATH="$DEST_T2"
    fi

    # SAVE: Write variables to config file
    # Note: We save DEST_T1 and SAVED_T2_PATH as the paths, effectively
    # "forgetting" the original source paths in future runs.
    cat <<EOF > "$CONFIG_FILE"

# Auto-generated pipeline config
# Created on $(date)
SUBJECT="$SUBJECT"
SUBJECTS_DIR="$SUBJECTS_DIR"
T1_PATH="$DEST_T1"
T2_PATH="$SAVED_T2_PATH"
S4L_AVAILABLE="$S4L_AVAILABLE"
USE_T2=$USE_T2
IS_FLAIR=$IS_FLAIR
USE_SKIPREGISTER_T2=$USE_SKIPREGISTER_T2
USE_NONECK=$USE_NONECK
N_THREADS=$N_THREADS
EOF

    echo "Configuration saved to $CONFIG_FILE"
fi


###################################################################################
# PREPARE FILES (Idempotency Check)

# DEST_T1 and DEST_T2 are already defined above.

# Ensure T1 exists in raw folder
if [ ! -f "$DEST_T1" ]; then
    echo "Copying T1 to raw folder..."
    # On first run: T1_PATH is the user source.
    # On rerun: T1_PATH is DEST_T1. If file is missing, this copy will fail (copying self to self),
    # which is acceptable behavior as the raw data was deleted.
    cp "$T1_PATH" "$DEST_T1"
else
    if [ -n "$T1_PATH" ]; then
        echo "WARNING: T1 already exists in raw folder, using the already existing one."
    fi
fi

# Ensure T2 exists if used
if [ "$USE_T2" = true ]; then
    if [ ! -f "$DEST_T2" ]; then
        echo "Copying T2 to raw folder..."
        if [ -z "$T2_PATH" ]; then
             echo "ERROR: Config says USE_T2=true, but T2_PATH is empty and file is missing."
             exit 1
        fi
        cp "$T2_PATH" "$DEST_T2"
    else
	if [ -n "$T1_PATH" ]; then
             echo "WARNING: T2 already exists in raw folder, using the already existing one."
        fi
    fi
fi

if [ "$S4L_AVAILABLE" = true ]; then
	mkdir -p "${SUBJECTS_DIR}/${SUBJECT}/tissue_labels/sim4life_raw"
	if [ ! -f "${SUBJECTS_DIR}/${SUBJECT}/tissue_labels/sim4life_raw/label_field.nii.gz" ]; then
		if [ -n "$SIM4LIFE_VOL" ]; then
			cp "$SIM4LIFE_VOL" "${SUBJECTS_DIR}/${SUBJECT}/tissue_labels/sim4life_raw/label_field.nii.gz"
		else
			echo "Config says that Sim4Life label field is available but the corresponding volume cannot be found, rerun the script supplying the file with --s4l-vol '$SIM4LIFE_VOL_PATH'"
			exit 1
		fi
	fi
	if [ ! -f "${SUBJECTS_DIR}/${SUBJECT}/tissue_labels/sim4life_raw/label_field.txt" ]; then
		if [ -n "$SIM4LIFE_LUT" ]; then
                        cp "$SIM4LIFE_LUT" "${SUBJECTS_DIR}/${SUBJECT}/tissue_labels/sim4life_raw/label_field.txt"
                else
                        echo "Config says that Sim4Life label LUT is available but the corresponding text file cannot be found, rerun the script supplying the file with --s4l-lut '$SIM4LIFE_LUT_PATH'"
                        exit 1
                fi
	fi
fi
###########################################################

#### handling optional arguments ####

# optional arguments to freesurfer
recon_args=()

if [ "$USE_T2" = true ]; then
	recon_args+=(--T2 /SUBJECTS/"$SUBJECT"/raw/T2.nii.gz)
        if [ "$IS_FLAIR" = true ]; then
            recon_args+=(--FLAIR)
        fi
fi

if [ "$USE_SKIPREGISTER_T2" = true ]; then
    recon_args+=(--skip-register-T2)
fi

if [ "$USE_NONECK" = true ]; then
    recon_args+=(--no-neck)
fi

recon_args+=(--threads "$N_THREADS")

##################### GPU usage ############################

if [[ -n "$USE_GPU" ]]; then
    if ! nvidia-smi &> /dev/null; then
	USE_GPU=""
	echo "No NVIDIA GPU found (or drivers missing). Falling back to CPU mode."
    fi
fi

##################### make folder structure ################################

# temporary directory with T1 file
TMP_DIR=$(mktemp -d "$PWD/.temp_pipeline.XXXXXX")


# Check if the directory was created successfully
if [[ ! -d "$TMP_DIR" ]]; then
    echo "Failed to create temporary directory." >&2
    exit 1
fi

# ensure temporary directory will be deleted at the end of the script
trap "echo 'Cleaning up temporary dir $TMP_DIR...'; rm -rf \"$TMP_DIR\"" EXIT

mkdir -p "$TMP_DIR"/input
cp "$DEST_T1" "$TMP_DIR/input/subject.nii.gz"

# reconstruction logs folder
mkdir -p "${SUBJECTS_DIR}/${SUBJECT}/reconstruction_logs"

# reconstruction log file
LOG_FILE="$SUBJECTS_DIR/$SUBJECT/reconstruction_logs/reconstruction_log.txt"

############################ run reconstruction ###################################
echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"
echo "Script run on [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$LOG_FILE"

echo
echo "----------------------------------------------------------------------------------------"
echo "Running reconstruction for $SUBJECT..."
echo "----------------------------------------------------------------------------------------"
echo

start_time=$(date +%s)

# if not already done, run fastsurfer
if [ ! -d "$SUBJECTS_DIR/$SUBJECT/fastsurfer" ]; then
	echo "Running FastSurfer reconstruction..."
	mkdir -p "$TMP_DIR"/tmp_fastsurfer

	start=$(date +%s)
	docker run -v "$TMP_DIR":/data -v "$SUBJECTS_DIR":/fs_license $USE_GPU --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest --fs_license /fs_license/license.txt --t1 /data/input/subject.nii.gz --sid "$SUBJECT" --sd /data/tmp_fastsurfer --3T --threads "$N_THREADS" --seg_only > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/fastsurfer.txt 2>&1
	check_step $? "FastSurfer reconstruction" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/fastsurfer.txt
	end=$(date +%s)
	# move output
	mv "$TMP_DIR"/tmp_fastsurfer/"$SUBJECT" "$SUBJECTS_DIR"/"$SUBJECT"/fastsurfer

	duration=$(( end - start ))
	minutes=$(( duration / 60 ))

	echo "FastSurfer reconstruction completed in ${minutes} minutes." | tee -a "$LOG_FILE"
else
        echo "FastSurfer reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo


# if not already done, run hippunfold
if [ ! -d "$SUBJECTS_DIR/$SUBJECT/hippunfold" ]; then
        echo "Running HippUnfold reconstruction..."
        mkdir -p "$TMP_DIR"/tmp_hippunfold

	start=$(date +%s)
	docker run -it --rm -v "$TMP_DIR":/data khanlab/hippunfold:latest /data/input /data/tmp_hippunfold participant --modality T1w --cores "$N_THREADS" --path_T1w /data/input/{subject}.nii.gz > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/hippunfold.txt 2>&1
	check_step $? "HippUnfold reconstruction" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/hippunfold.txt
	end=$(date +%s)

        # move output
        mv "$TMP_DIR"/tmp_hippunfold/hippunfold/sub-subject "$SUBJECTS_DIR"/"$SUBJECT"/hippunfold

        # make label file
        cat <<EOL > "$SUBJECTS_DIR"/"$SUBJECT"/hippunfold/LABELS.txt
index, name, abbreviation
1, subiculum, Sub
2, CA1, CA1
3, CA2, CA2
4, CA3, CA3
5, CA4, CA4
6, dentate_gyrus, DG
7, SRLM, SRLM
8, cysts, Cyst
EOL

        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "HippUnfold reconstruction completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
        echo "HippUnfold reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo


# OLD #####################
# if not already done, run dbsegment
#if [ ! -d "$SUBJECTS_DIR/$SUBJECT/dbsegment" ]; then
#        echo "Running dbSegment reconstruction..."
#        mkdir -p "$TMP_DIR"/tmp_dbsegment
#        mkdir -p "$TMP_DIR"/tmp_dbsegment_models
#
#	start=$(date +%s)
#	docker run --rm $USE_GPU -v "$TMP_DIR"/input:/input -v "$TMP_DIR"/tmp_dbsegment:/output -v "$TMP_DIR"/tmp_dbsegment_models:/models imagingai/dbsegment:latest > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/dbsegment.txt 2>&1
#	check_step $? "dbSegment reconstruction" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/dbsegment.txt
#	end=$(date +%s)
#
#        # move output
#        mv "$TMP_DIR"/tmp_dbsegment "$SUBJECTS_DIR"/"$SUBJECT"/dbsegment
#
#        duration=$(( end - start ))
#        minutes=$(( duration / 60 ))
#
#        echo "dbSegment reconstruction completed in ${minutes} minutes." | tee -a "$LOG_FILE"
#else
#	echo "dbSegment reconstruction detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
#fi
#echo
############################

# start reconstruction in docker container
docker run --rm $USE_GPU -v "$SUBJECTS_DIR":/SUBJECTS christianbuda/parrot_MRI_reconstruction:latest --subject "$SUBJECT" --T1 /SUBJECTS/"$SUBJECT"/raw/T1.nii.gz "${recon_args[@]}"

end_time=$(date +%s)

duration=$(( end_time - start_time ))
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))

echo
echo "----------------------------------------------------------------------------------------"
echo "Done! Full script execution took ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
echo "----------------------------------------------------------------------------------------"
echo
