#!/bin/bash

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

############## variables ################################
# MANDATORY: directory where subject data lives
SUBJECTS_DIR=""

# MANDATORY: subject ID
SUBJECT=""

## other pipeline variables

SPACING_LIST=(4 3 2)

# number of threads to use for cgal meshing step
N_THREADS=8

# help function
# Help function
usage() {
    echo "Usage: $0 -s SUBJECT -d SUBJECTS_DIR [OPTIONS]"
    echo ""
    echo "Mandatory:"
    echo "  -s, --subject                Subject ID"
    echo "  -d, --subjects-dir           directory where subject data (from previous reconstruction) lives"
    echo ""
    echo "Pipeline Options:"
    echo "  --spacing           Array of spacings (in mm) between the dipoles (Default: 4, 3, 2)"
    echo "                      NOTE: The script automatically uses all three default spacings for computation, if you change this, make sure to adapt the rest of the script to accomodate for this change."
    echo "  --threads           Number of threads, you can go up to about 30 without having diminishing returns (Default: 8)"
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
        --threads)
            N_THREADS="$2"
            shift 2
            ;;
        --spacing)
            # Capture all following numeric arguments into the array
            shift
            SPACING_LIST=()
            while [[ $1 =~ ^[0-9.]+$ ]]; do
                spacings+=("$1")
                shift
            done
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
# check previous reconstruction
if [ ! -d "$SUBJECTS_DIR"/"$SUBJECT"/surfaces ] || [ ! -d "$SUBJECTS_DIR"/"$SUBJECT"/atlas ] || [ ! -d "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical ]; then
	echo "ERROR: Critical error: MRI reconstruction not found, please run it before launching this script."
    exit 1
fi

# check fiducials
if [ ! -f "$SUBJECTS_DIR"/"$SUBJECT"/scalp_landmarks/fiducials.json ]; then
	echo "ERROR: Critical error: Fiducial positions (NAS,IN,RPA,LPA) not found in $SUBJECTS_DIR/$SUBJECT/scalp_landmarks/fiducials.json, either create it manually or use the pick_landmarks.ipynb notebook in the parrot repository."
    exit 1
fi

# check whether sim4life reconstruction is available or not
VOLUME_TO_MESH="simnibs"
if [ -f "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/sim4life.nii.gz ]; then
    VOLUME_TO_MESH="sim4life"
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

# reconstruction logs folder
mkdir -p "${SUBJECTS_DIR}/${SUBJECT}/reconstruction_logs"

# reconstruction log file
LOG_FILE="$SUBJECTS_DIR/$SUBJECT/reconstruction_logs/forward_model_pipeline_log.txt"

############################ run reconstruction ###################################
echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"
echo "Script run on [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$LOG_FILE"

echo
echo "----------------------------------------------------------------------------------------"
echo "Running forward model pipeline for $SUBJECT..."
echo "----------------------------------------------------------------------------------------"
echo


start_time=$(date +%s)



# if not already done, compute electrodes positions
if [ ! -d "$SUBJECTS_DIR/$SUBJECT/electrodes" ]; then
	echo "Computing electrodes positions on the reconstructed scalp..."

	start=$(date +%s)
    docker run --rm -v $SUBJECTS_DIR/$SUBJECT:/subject parrot_forward_model python place_electrodes.py > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/electrodes.txt 2>&1
	check_step $? "Electrodes computation" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/electrodes.txt "$SUBJECTS_DIR"/"$SUBJECT"/electrodes
	end=$(date +%s)
	# move output

	duration=$(( end - start ))
	minutes=$(( duration / 60 ))

	echo "Electrodes computation completed in ${minutes} minutes." | tee -a "$LOG_FILE"
else
    echo "Electrodes positions detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo


# if not already done, compute dipoles positions at specified spacings
mkdir -p "$SUBJECTS_DIR/$SUBJECT/dipoles"
echo "Each folder contains the sampled dipoles at the specified spacing." > "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "For each sample, four kinds of files are generated:" >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "dipole_positions.npy: (N,3) array that contains the position of each dipole in world space." >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "dipole_volume.npy: (N,) array that contains the estimated volume of gray matter associated to each dipole." >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "dipole_directions.npy: (N,3) array that contains the preferential direction of each dipole in world space." >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "orient_type.npy: (N,) array that indicates the kind of procedure used to choose a preferential direction (one of ['U', 'N', 'G', 'P', 'R'], which mean, respectively, ['Unassigned', 'Normal to a surface', 'Gradient of smoothed structure (to mimick normal to surface)', 'Principal axis of the structure', 'Randomly generated (uniformly on unit sphere)'])." >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
echo "*_dipole_volume.npy: (N,) array that contains the label associated to each dipole in various atlases." >> "$SUBJECTS_DIR/$SUBJECT/dipoles/legend.txt"
for s in "${SPACING_LIST[@]}"; do
    # Ensure 1 decimal point formatting
    spacing=$(printf "%.1f" "$s")

    if [ ! -d "$SUBJECTS_DIR"/"$SUBJECT"/dipoles/spacing"$spacing"mm ]; then
        echo "Placing dipoles at $spacing mm spacing..."

        start=$(date +%s)
        docker run --rm -v $SUBJECTS_DIR/$SUBJECT:/subject parrot_forward_model python place_dipoles.py --dipole_spacing $spacing > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/dipoles"$spacing"mm.txt 2>&1
        check_step $? "Dipole placing at $spacing mm spacing" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/dipoles"$spacing"mm.txt "$SUBJECTS_DIR"/"$SUBJECT"/dipoles/spacing"$spacing"mm
        end=$(date +%s)


        duration=$(( end - start ))
        hours=$(( duration / 3600 ))
        minutes=$(( (duration % 3600) / 60 ))

        echo "Dipoles placing (with $spacing mm spacing) completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
    else
        echo "Dipoles placed at $spacing mm spacing detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
    fi
    echo
done


# if not already done, mesh the label fields
if [ ! -d "$SUBJECTS_DIR/$SUBJECT/tetmesh" ]; then
    # load config for current mesh
    CONFIG_FILE="$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/"$VOLUME_TO_MESH"_mesher_parameters.txt
    ANGLE="30.0"; DIST="2.0"; DEF_SURF="2.0"; DEF_VOL="5.0"; RATIO="3.0"; SMOOTH="2"; OPT_TIME="1800"

    # parse parameters file
    if [ ! -f "$CONFIG_FILE" ]; then echo "Error: $CONFIG_FILE not found!"; exit 1; fi
    eval $(awk '/^(ANGLE|DIST|DEF_SURF|DEF_VOL|RATIO|SMOOTH|OPT_TIME)/ {print $1"="$2}' "$CONFIG_FILE")
    TISSUE_ARGS=($(awk '/^[[:space:]]*[0-9]/ {print $1":"$2":"$3}' "$CONFIG_FILE"))

    # copy label field in local directory
    mkdir -p "$TMP_DIR"/tetmesh
    cp "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/"$VOLUME_TO_MESH".nii.gz "$TMP_DIR/tetmesh/label_field.nii.gz"

	start=$(date +%s)
    echo "Converting nifti label field to inr..."
    docker run --rm -v "$TMP_DIR/tetmesh":/data parrot_forward_model python nifti_to_inr.py --nifti_path /data/label_field.nii.gz --inr_path /data/label_field.inr > "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt 2>&1
	check_step $? "Nifti to inr" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt

    echo "Meshing label field using CGAL..."
    docker run --rm -v "$TMP_DIR/tetmesh":/data parrot_forward_model mesher $N_THREADS /data/label_field.inr /data/tetrahedral_mesh.mesh "$ANGLE" "$DIST" "$DEF_SURF" "$DEF_VOL" "$RATIO" "$SMOOTH" "$OPT_TIME" "${TISSUE_ARGS[@]}" >> "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt 2>&1
	check_step $? "CGAL meshing" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt

    echo "Converting tetrahedral mesh back to world space..."
    docker run --rm -v "$TMP_DIR/tetmesh":/data parrot_forward_model python mesh_postprocessing.py --reference_nifti /data/label_field.nii.gz --mesh /data/tetrahedral_mesh.mesh --output /data/transformed_tetrahedral_mesh.mesh --export_vtu >> "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt 2>&1
	check_step $? "Mesh to world space" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/tetmesh.txt
	end=$(date +%s)
	
    # move outputs
    mkdir "$SUBJECTS_DIR/$SUBJECT/tetmesh"
    cp "$TMP_DIR/tetmesh/transformed_tetrahedral_mesh.mesh" "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/tetrahedral_mesh.mesh
    cp "$TMP_DIR/tetmesh/transformed_tetrahedral_mesh.vtu" "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/tetrahedral_mesh.vtu
    cp "$CONFIG_FILE" "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/mesher_parameters.txt
	cp "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/"$VOLUME_TO_MESH"_labels.txt "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/labels.txt
    cp "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/"$VOLUME_TO_MESH"_LUT.txt "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/LUT.txt
    cp "$SUBJECTS_DIR"/"$SUBJECT"/tissue_labels/electrical/"$VOLUME_TO_MESH"_conductivities.txt "$SUBJECTS_DIR"/"$SUBJECT"/tetmesh/conductivities.txt

    duration=$(( end - start ))
    hours=$(( duration / 3600 ))
    minutes=$(( (duration % 3600) / 60 ))

	echo "Tetrahedral meshing completed in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
    echo "Tetrahedral mesh detected in subject's folder, skipping step..." | tee -a "$LOG_FILE"
fi
echo

# if not already done, solve the forward problem
if [ ! -d "$SUBJECTS_DIR/$SUBJECT/forward_solvers" ]; then
    mkdir -p "$SUBJECTS_DIR/$SUBJECT/forward_solvers"

    start=$(date +%s)
    
	spacing=$(printf "%.1f" "${SPACING_LIST[0]}")
    echo "Solving forward problem with OpenMEEG at $spacing mm dipole spacing"
    docker run --rm -v $DATA/SUBJECTS/mni_nlin_asym_09b:/subject parrot_forward_solvers /scripts/make_leadfield_openmeeg.py --dipole_spacing "$spacing" >> "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt 2>&1
	check_step $? "OpenMEEG $spacing mm" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt "$SUBJECTS_DIR/$SUBJECT/forward_solvers"

    spacing=$(printf "%.1f" "${SPACING_LIST[1]}")
    echo "Solving forward problem with DUNEuro using SimNIBS charm mesh, at $spacing mm dipole spacing"
    docker run --rm -v $DATA/SUBJECTS/mni_nlin_asym_09b:/subject parrot_forward_solvers /scripts/make_leadfield_duneuro.py --dipole_spacing "$spacing" --mesh_path simnibs_charm/subject.msh --tissue_names simnibs_charm/labels.txt --conductivities_path simnibs_charm/conductivities.txt --label simnibs --valid_tissues "Gray-Matter" >> "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt 2>&1
	check_step $? "DUNEuro SimNIBS $spacing mm" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt "$SUBJECTS_DIR/$SUBJECT/forward_solvers"

    spacing=$(printf "%.1f" "${SPACING_LIST[2]}")
    echo "Solving forward problem with DUNEuro using CGAL mesh, at $spacing mm dipole spacing"
    docker run --rm -v $DATA/SUBJECTS/mni_nlin_asym_09b:/subject parrot_forward_solvers /scripts/make_leadfield_duneuro.py --dipole_spacing "$spacing" --mesh_path tetmesh/tetrahedral_mesh.mesh --tissue_names tetmesh/labels.txt --conductivities_path tetmesh/conductivities.txt --label CGAL --valid_tissues "Brain (Grey Matter)" Thalamus Hippocampus >> "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt 2>&1
	check_step $? "DUNEuro CGAL $spacing mm" "$SUBJECTS_DIR"/"$SUBJECT"/reconstruction_logs/forward_solvers.txt "$SUBJECTS_DIR/$SUBJECT/forward_solvers"
	end=$(date +%s)
	
    duration=$(( end - start ))
    hours=$(( duration / 3600 ))
    minutes=$(( (duration % 3600) / 60 ))

	echo "All three forward solvers run in ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
else
    echo "Forward solvers folder detected, skipping step..." | tee -a "$LOG_FILE"
fi
echo

end_time=$(date +%s)

duration=$(( end_time - start_time ))
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))

echo
echo "----------------------------------------------------------------------------------------"
echo "Done! Full script execution took ${hours} hours and ${minutes} minutes." | tee -a "$LOG_FILE"
echo "----------------------------------------------------------------------------------------"
echo
