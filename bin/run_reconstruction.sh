#!/bin/bash

###############################################################################
# PARROT MRI RECONSTRUCTION - BIDS APP
###############################################################################

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
            echo "Cleaning up incomplete directory: $cleanup_path"
            rm -rf "$cleanup_path"
        fi

        echo
        exit 1
    fi
}

usage() {
    echo "Usage: $0 bids_dir output_dir participant [OPTIONS]"
    echo ""
    echo "Positional Arguments (Mandatory BIDS format):"
    echo "  bids_dir              The root directory of the BIDS dataset."
    echo "  output_dir            The root derivatives directory (e.g., /bids/derivatives)."
    echo "  analysis_level        Level of analysis (must be 'participant')."
    echo ""
    echo "Options:"
    echo "  --participant-label        List of subject IDs to process (e.g., 01 02). If omitted, processes all."
    echo "  --threads                  Number of threads to use for software that support it (Default: 8)."
    echo "  --gpus                     GPU configuration: 'all' (default), 'none', or specific devices (e.g., 'device=0,1' or '2')."
    echo "  --spacing-openmeeg         Dipole spacing (mm) for the OpenMEEG BEM solver (Default: 4)."
    echo "  --spacing-duneuro-simnibs  Dipole spacing (mm) for DUNEuro FEM with SimNIBS mesh (Default: 3)."
    echo "  --spacing-duneuro-cgal     Dipole spacing (mm) for DUNEuro FEM with CGAL mesh (Default: 2)."
    echo "  --dipole-seed              Integer seed for reproducible dipole sampling (Default: unset = random)."
    echo "  --dwi-preprocessed         Treat the BIDS dwi/ data as already corrected and skip QSIPrep (e.g. HCP)."
    exit 1
}

# Get the absolute directory of parrot
PARROT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

# Docker image definitions (single source of truth, shared with bin/build.sh)
source "$PARROT_SCRIPT_DIR/bin/images.sh"

# =============================================================================
# 1. PARSE POSITIONAL BIDS ARGUMENTS
# =============================================================================
if [ $# -lt 3 ]; then
    usage
fi

BIDS_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath "$2")
ANALYSIS_LEVEL="$3"
shift 3

if [ "$ANALYSIS_LEVEL" != "participant" ]; then
    echo "ERROR: Analysis level must be 'participant'."
    usage
fi

# =============================================================================
# 2. DEFAULT VARIABLES & PARSE OPTIONAL ARGUMENTS
# =============================================================================
PARTICIPANTS=()
N_THREADS=8
GPU_OPT="all"
SPACING_OPENMEEG=4
SPACING_DUNEURO_SIMNIBS=3
SPACING_DUNEURO_CGAL=2
DIPOLE_SEED=""
DWI_PREPROCESSED=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --participant-label)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                PARTICIPANTS+=("$1")
                shift
            done
            ;;
        --threads)
            N_THREADS="$2"
            shift 2
            ;;
        --gpus)
            GPU_OPT="$2"
            shift 2
            ;;
        --spacing-openmeeg)
            SPACING_OPENMEEG="$2"
            shift 2
            ;;
        --spacing-duneuro-simnibs)
            SPACING_DUNEURO_SIMNIBS="$2"
            shift 2
            ;;
        --spacing-duneuro-cgal)
            SPACING_DUNEURO_CGAL="$2"
            shift 2
            ;;
        --dipole-seed)
            DIPOLE_SEED="$2"
            shift 2
            ;;
        --dwi-preprocessed)
            DWI_PREPROCESSED=true
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

# Internal list of all spacings needed for dipole precomputation (one per solver)
SPACING_LIST=("$SPACING_OPENMEEG" "$SPACING_DUNEURO_SIMNIBS" "$SPACING_DUNEURO_CGAL")

# =============================================================================
# 3. PRE-FLIGHT CHECKS
# =============================================================================

# GPU Configuration Logic
if [ "$GPU_OPT" == "none" ]; then
    DOCKER_GPU=""
    echo "Notice: GPU disabled by user. Running in CPU-only mode."
else
    # Check if nvidia-smi exists and can talk to the driver
    if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        DOCKER_GPU=""
        echo "WARNING: nvidia-smi not found or driver missing. Falling back to CPU-only mode."
    else
        DOCKER_GPU="--gpus $GPU_OPT"
        echo "GPU Configuration: $DOCKER_GPU"
    fi
fi

# Ensure required Docker images are present (pull any that are missing).
# Image list comes from bin/images.sh; this replaces the old standalone setup.sh.
ALL_IMAGES=("${EXTERNAL_IMAGES[@]}")
for entry in "${PARROT_IMAGES[@]}"; do
    ALL_IMAGES+=("${entry%%|*}")
done

echo "Checking required Docker images..."
for img in "${ALL_IMAGES[@]}"; do
    if [[ -z "$(docker images -q "$img" 2> /dev/null)" ]]; then
        echo "  Missing $img - pulling (this may take a while)..."
        if ! docker pull "$img"; then
            echo "ERROR: Failed to pull $img"
            exit 1
        fi
    else
        echo "  Found $img"
    fi
done

# Scratch work directory for BIDS apps that need one (e.g. QSIPrep/nipype, which
# can balloon to tens of GB). Placed inside the output dir so it shares the large
# derivatives filesystem rather than a possibly RAM-backed /tmp, and removed when
# the script exits (success, error, or interrupt). Ephemeral by design: a failed
# QSIPrep run therefore restarts from scratch on re-run (no nipype resume cache).
WORK_DIR=$(mktemp -d "$OUTPUT_DIR/.parrot_work.XXXXXX")
WORK_DIR_DOCKER="/derivatives/$(basename "$WORK_DIR")"
cleanup_work_dir() { [ -n "${WORK_DIR:-}" ] && rm -rf "$WORK_DIR"; }
trap cleanup_work_dir EXIT INT TERM

# Persistent TemplateFlow cache (NOT swept on exit): QSIPrep fetches templates at
# runtime and the image ships none, so caching here downloads them once and reuses
# them across subjects/runs.
TEMPLATEFLOW_DIR="$OUTPUT_DIR/.templateflow"
mkdir -p "$TEMPLATEFLOW_DIR"

# Auto-discover participants if none were provided
if [ ${#PARTICIPANTS[@]} -eq 0 ]; then
    echo "No participant labels provided. Scanning $BIDS_DIR for subjects..."
    for sub_dir in "$BIDS_DIR"/sub-*; do
        if [ -d "$sub_dir" ]; then
            sub_id=$(basename "$sub_dir" | sed 's/sub-//')
            PARTICIPANTS+=("$sub_id")
        fi
    done
fi

if [ ${#PARTICIPANTS[@]} -eq 0 ]; then
    echo "ERROR: No subjects found in $BIDS_DIR."
    exit 1
fi

echo "Subjects to process: ${PARTICIPANTS[*]}"
echo "====================================================================="

# =============================================================================
# DOCKER WRAPPER FUNCTION (The Magic Ingredient)
# =============================================================================
# This dynamically spins up the container, sources the environment, and runs your command
run_in_docker_MRI() {
    local step_name=$1
    local log_file=$2
    local cmd=$3
    
    echo "Running $step_name..."
    # --entrypoint /bin/bash overrides any internal entrypoints so we can run raw commands
    docker run --rm $DOCKER_GPU --entrypoint /bin/bash \
        -v "$BIDS_DIR":/bids:ro \
        -v "$OUTPUT_DIR":/derivatives \
        "$IMG_MRI_RECONSTRUCTION" \
        -c "source /scripts/source_env.sh && $cmd" > "$log_file" 2>&1
        
    check_step $? "$step_name" "$log_file"
}

run_in_docker_FWD() {
    local step_name=$1
    local log_file=$2
    local image=$3
    local cmd=$4
    
    echo "Running $step_name..."
    # --entrypoint /bin/bash overrides any internal entrypoints so we can run raw commands
    docker run --rm $DOCKER_GPU --entrypoint /bin/bash \
        -v "$BIDS_DIR":/bids:ro \
        -v "$OUTPUT_DIR":/derivatives \
        "$image" -c "$cmd" > "$log_file" 2>&1
        
    check_step $? "$step_name" "$log_file"
}

# =============================================================================
# 4. MAIN PROCESSING LOOP
# =============================================================================

for SUBJECT in "${PARTICIPANTS[@]}"; do
    echo ""
    echo "====================================================================="
    echo " Processing sub-${SUBJECT}"
    echo "====================================================================="
    
    # Subject BIDS input directory
    SUB_BIDS_DIR="$BIDS_DIR/sub-${SUBJECT}"
       
    # Subject-specific log directory (stored inside Parrot's folder)
    LOG_DIR="$OUTPUT_DIR/logs/sub-${SUBJECT}"
    LOG_FILE="$LOG_DIR/parrot-reconstruction_log.txt"
    
    mkdir -p "$LOG_DIR"
    echo "Run started on [$(date '+%Y-%m-%d %H:%M:%S')]" >> "$LOG_FILE"

    # ---------------------------------------------------------
    # Auto-Discover Anatomical Files
    # ---------------------------------------------------------
    T1_PATH=$(find "$SUB_BIDS_DIR/anat" -name "sub-${SUBJECT}*_T1w.nii.gz" | head -n 1)
    T2_PATH=$(find "$SUB_BIDS_DIR/anat" -name "sub-${SUBJECT}*_T2w.nii.gz" | head -n 1)
    FLAIR_PATH=$(find "$SUB_BIDS_DIR/anat" -name "sub-${SUBJECT}*_FLAIR.nii.gz" | head -n 1)

    if [ -z "$T1_PATH" ]; then
        echo "[ERROR] No T1w image found for sub-${SUBJECT}. Skipping..." | tee -a "$LOG_FILE"
        continue
    fi
    echo "Found T1w: $T1_PATH" | tee -a "$LOG_FILE"

    # Map paths for inside the container
    T1_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$T1_PATH")"

    fs_args=()
    simnibs_args=()
    if [ -n "$T2_PATH" ]; then
        T2_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$T2_PATH")"
        fs_args=("-T2" "$T2_DOCKER" "-T2pial")
        simnibs_args=("$T2_DOCKER")
    elif [ -n "$FLAIR_PATH" ]; then
        FLAIR_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$FLAIR_PATH")"
        fs_args=("-FLAIR" "$FLAIR_DOCKER" "-FLAIRpial")
    fi

    # ---------------------------------------------------------
    # Auto-Discover Diffusion (DWI) — optional
    # ---------------------------------------------------------
    # DWI drives subject-specific structural connectivity. It is optional: with
    # no usable DWI the pipeline degrades gracefully to the template connectome.
    # Single user-facing input location is the BIDS dwi/ folder (the same whether
    # or not --dwi-preprocessed is set); the flag only toggles whether QSIPrep
    # correction runs.
    DWI_PATH=$(find "$SUB_BIDS_DIR/dwi" -name "sub-${SUBJECT}*_dwi.nii.gz" 2>/dev/null | head -n 1)
    HAS_DWI=false
    if [ -n "$DWI_PATH" ]; then
        BVAL_PATH="${DWI_PATH%.nii.gz}.bval"
        BVEC_PATH="${DWI_PATH%.nii.gz}.bvec"
        if [ -f "$BVAL_PATH" ] && [ -f "$BVEC_PATH" ]; then
            HAS_DWI=true
            DWI_DOCKER="/bids/sub-${SUBJECT}/dwi/$(basename "$DWI_PATH")"
            if [ "$DWI_PREPROCESSED" = true ]; then
                echo "Found DWI (flagged already-preprocessed): $DWI_PATH" | tee -a "$LOG_FILE"
            else
                echo "Found DWI: $DWI_PATH" | tee -a "$LOG_FILE"
            fi
        else
            echo "[WARN] DWI found but .bval/.bvec missing alongside it; treating sub-${SUBJECT} as no-DWI." | tee -a "$LOG_FILE"
        fi
    fi

    if [ "$DWI_PREPROCESSED" = true ] && [ "$HAS_DWI" = false ]; then
        echo "[WARN] --dwi-preprocessed set but no usable DWI found for sub-${SUBJECT}; falling back to template connectome." | tee -a "$LOG_FILE"
    fi

    # TSV Overrides
    if [ -f "$BIDS_DIR/participants.tsv" ]; then
        SUB_ROW=$(grep "^sub-${SUBJECT}" "$BIDS_DIR/participants.tsv" 2>/dev/null)
        if [ -n "$SUB_ROW" ]; then
            if [ "$(echo "$SUB_ROW" | awk '{print tolower($4)}')" == "true" ]; then 
                simnibs_args+=("--skipregisterT2")
            fi
            if [ "$(echo "$SUB_ROW" | awk '{print tolower($5)}')" == "true" ]; then 
                simnibs_args+=("--noneck")
            fi
        fi
    fi

    fs_args+=(--threads "$N_THREADS")

    # =========================================================================
    # 5. EXECUTE PIPELINE STEPS (With Robust Idempotency)
    # =========================================================================

    start_time=$(date +%s)

    # ---------------------------------------------------------
    # FASTSURFER
    # ---------------------------------------------------------
    NAME="fastsurfer"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)
        docker run $DOCKER_GPU --rm --user $(id -u):$(id -g) \
            -v "$BIDS_DIR":/data:ro \
            -v "$OUTPUT_DIR/$NAME":/output \
            "$IMG_FASTSURFER" \
            --fs_license /data/license.txt \
            --t1 "/data/sub-${SUBJECT}/anat/$(basename "$T1_PATH")" \
            --sid "sub-${SUBJECT}" \
            --sd /output \
            --3T --threads "$N_THREADS" --seg_only > "$LOG_DIR/${NAME}_log.txt" 2>&1
            
        check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
        
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # HIPPUNFOLD
    # ---------------------------------------------------------
    NAME="hippunfold"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)
        docker run -it --rm \
            -v "$BIDS_DIR":/bids:ro \
            -v "$OUTPUT_DIR/$NAME":/output \
            "$IMG_HIPPUNFOLD" \
            /bids /output participant \
            --participant_label "$SUBJECT" \
            --modality T1w --cores "$N_THREADS" > "$LOG_DIR/${NAME}_log.txt" 2>&1
            
        check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
        
        cp "$PARROT_SCRIPT_DIR/template_data/hippunfold_labels.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/LABELS.txt"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # PARROT MRI RECONSTRUCTION
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # FREESURFER RECON-ALL
    # ---------------------------------------------------------
    NAME="freesurfer"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME"
        
        step_start=$(date +%s)
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" \
            "export SUBJECTS_DIR=/derivatives/freesurfer && \
             recon-all -subject sub-${SUBJECT} -i $T1_DOCKER ${fs_args[@]} -all -threads $N_THREADS && \
             cp \$FREESURFER_HOME/FreeSurferColorLUT.txt /derivatives/freesurfer/sub-${SUBJECT}/FreeSurferColorLUT.txt && \
             cp -r /home/Schaefer2018_LocalGlobal/Parcellations/project_to_individual /derivatives/freesurfer/sub-${SUBJECT}/Schaefer_LUT"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # MNE BEM SURFACES
    # ---------------------------------------------------------
    NAME="mne"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        
        step_start=$(date +%s)
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" \
            "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && \
             micromamba run -n neuro python /scripts/make_bem_surfaces.py --subject $SUBJECT --subjects_dir \$FREESURFER_HOME/subjects"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # SCHAEFER ATLASES
    # ---------------------------------------------------------
    NAME="schaefer"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        
        step_start=$(date +%s)

        for n_parcels in {100..1000..100}; do
            ATLAS_NAME="Schaefer2018_${n_parcels}Parcels_17Networks_order"

            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/lh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/lh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/rh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/rh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_aparc2aseg --s $SUBJECT --o \$SUBJECTS_DIR/$SUBJECT/mri/schaefer${n_parcels}_aparc+aseg.mgz --annot $ATLAS_NAME"
        done

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # SUBCORTICAL FREESURFER
    # ---------------------------------------------------------
    NAME="freesurfersubcortical"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        
        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions thalamus --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions hippo-amygdala --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/freesurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions brainstem --cross $SUBJECT --threads $N_THREADS"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # SIMNIBS CHARM
    # ---------------------------------------------------------
    NAME="simnibscharm"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cd /home/simnibs_reconstructions && \
                                                        /root/SimNIBS-4.5/bin/charm subject $T1_DOCKER ${simnibs_args[@]} --forcerun --fs-dir /derivatives/freesurfer/sub-${SUBJECT} --forcesform && \
                                                        cd / && \
                                                        /root/SimNIBS-4.5/bin/simnibs_python /scripts/extract_charm_surf.py --charm_dir "/home/simnibs_reconstructions/m2m_subject/" && \
                                                        cp /scripts/simnibs_conductivities.txt /home/simnibs_reconstructions/m2m_subject/conductivities.txt && \
                                                        cp /scripts/simnibs_labels.txt /home/simnibs_reconstructions/m2m_subject/labels.txt && \
                                                        mv /home/simnibs_reconstructions/m2m_subject $OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # FSL FIRST
    # ---------------------------------------------------------
    NAME="fslfirst"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        # bias field correct image and then run FSL first
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/bias_correct.py $T1_DOCKER $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1.nii.gz && \
	                                                    /scripts/run_first_all_sequential -i $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1.nii.gz -o $OUTPUT_DIR/$NAME/sub-${SUBJECT}/FSL -v"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # SYNTHSTRIP
    # ---------------------------------------------------------
    NAME="synthstrip"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        synth_flag=()
        if [ -n "$DOCKER_GPU" ] ; then
            synth_flag="--gpu"
        fi

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "mri_synthstrip -i "$T1_DOCKER" -o $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1_stripped.nii.gz -m $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1_stripped_mask.nii.gz ${synth_flag[@]} && \
	                                                    mri_synthstrip -i "$T1_DOCKER" -o $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1_noCSF_stripped.nii.gz -m $OUTPUT_DIR/$NAME/sub-${SUBJECT}/T1_noCSF_stripped_mask.nii.gz ${synth_flag[@]} --no-csf"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi
	

    # ---------------------------------------------------------
    # CEREBELLUM
    # ---------------------------------------------------------
    NAME="cerebellum"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cp /home/cerebellum_template/Cerebellar_Regions.csv $OUTPUT_DIR/$NAME/sub-${SUBJECT}/LABELS.csv && \
                                                        micromamba run -n neuro python /scripts/run_cereb_pipeline.py --output_dir /derivatives --subject $SUBJECT --template_dir /home/cerebellum_template/ --threads $N_THREADS"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi
	
    # ---------------------------------------------------------
    # BIGBRAIN
    # ---------------------------------------------------------
    NAME="bigbrain"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/run_bigbrain_pipeline.py --output_dir /derivatives --subject $SUBJECT --template_dir /home/bigbrain_scans/ --threads $N_THREADS"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # SURFACES
    # ---------------------------------------------------------
    NAME="surfaces"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/gather_surfaces.py --output_dir /derivatives --subject $SUBJECT"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # ATLASES
    # ---------------------------------------------------------
    NAME="atlas"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/make_atlas.py --T1_path $T1_DOCKER --output_dir /derivatives --subject $SUBJECT"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # TISSUE LABELS
    # ---------------------------------------------------------
    NAME="tissuelabels"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/electrical"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/acoustic"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cp /scripts/simnibs_mesher_parameters.txt $OUTPUT_DIR/$NAME/sub-${SUBJECT}/electrical/ && \
                                                        cp /scripts/sim4life_mesher_parameters.txt $OUTPUT_DIR/$NAME/sub-${SUBJECT}/electrical/ && \
                                                        micromamba run -n neuro python /scripts/gather_electrical_labelfields.py --T1_path $T1_DOCKER --output_dir /derivatives --subject $SUBJECT && \
                                                        micromamba run -n neuro python /scripts/gather_acoustic_labelfields.py --T1_path $T1_DOCKER --output_dir /derivatives --subject $SUBJECT"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

        
    # ---------------------------------------------------------
    # QSIPREP (DWI PREPROCESSING)
    # ---------------------------------------------------------
    # Optional. Runs only when usable raw DWI is present and not flagged as
    # already preprocessed. Produces corrected DWI (+ DTI) in subject anatomical
    # space under derivatives/qsiprep/, the input contract for the recon stage.
    # QSIPrep runs its own anatomical workflow (SynthStrip/TemplateFlow) and does
    # not consume our FreeSurfer dir -- FreeSurfer reuse belongs to the recon stage.
    NAME="qsiprep"
    if [ "$HAS_DWI" = true ] && [ "$DWI_PREPROCESSED" = false ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            echo "Running $NAME (DWI preprocessing)..." | tee -a "$LOG_FILE"
            mkdir -p "$OUTPUT_DIR/$NAME"

            step_start=$(date +%s)

            # Output resolution: --output-resolution is mandatory and forces an
            # ISOTROPIC grid. Policy = min(native smallest axis, 1.25): upsample
            # coarse DWI toward the MRtrix-recommended ~1.25 mm for tractography,
            # but never downsample finer-than-1.25 acquisitions. Read the native
            # voxel size with nibabel from our MRI image (it has the env).
            OUTPUT_RES=$(docker run --rm --entrypoint micromamba \
                -v "$BIDS_DIR":/bids:ro \
                "$IMG_MRI_RECONSTRUCTION" \
                run -n neuro python -c "import nibabel as nib; z=nib.load('$DWI_DOCKER').header.get_zooms()[:3]; print(round(min(min(z),1.25),2))" 2>/dev/null)
            if [ -z "$OUTPUT_RES" ]; then
                echo "[ERROR] Could not read DWI voxel size for sub-${SUBJECT}." | tee -a "$LOG_FILE"
                exit 1
            fi
            echo "Using --output-resolution $OUTPUT_RES mm." | tee -a "$LOG_FILE"

            # --user avoids root-owned outputs in the NFS derivatives tree; the
            # image's HOME (/home/qsiprep) is world-writable so we keep it as-is.
            # TemplateFlow is cached persistently across runs.
            docker run --rm $DOCKER_GPU --user "$(id -u):$(id -g)" \
                -e TEMPLATEFLOW_HOME=/templateflow \
                -v "$TEMPLATEFLOW_DIR":/templateflow \
                -v "$BIDS_DIR":/bids:ro \
                -v "$OUTPUT_DIR":/derivatives \
                "$IMG_QSIPREP" \
                /bids "/derivatives/$NAME" participant \
                --participant-label "$SUBJECT" \
                --fs-license-file /bids/license.txt \
                --output-resolution "$OUTPUT_RES" \
                --nprocs "$N_THREADS" \
                --skip-bids-validation \
                -w "$WORK_DIR_DOCKER" > "$LOG_DIR/${NAME}_log.txt" 2>&1

            check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    elif [ "$DWI_PREPROCESSED" = true ]; then
        echo "$NAME skipped for sub-${SUBJECT} (--dwi-preprocessed): using DWI as-is." | tee -a "$LOG_FILE"
    else
        echo "$NAME skipped for sub-${SUBJECT} (no usable DWI)." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # QSIRECON (TRACTOGRAPHY)
    # ---------------------------------------------------------
    # Optional. Selects an MRtrix recon spec adaptively from the .bval shell
    # scheme and exports the tractogram + SIFT2 weights for tck2connectome (run
    # in the connectivity stage). A too-sparse scheme -> skip (template fallback).
    # The --dwi-preprocessed (HCP, --input-type hcpya) path is not wired yet.
    NAME="qsirecon"
    if [ "$HAS_DWI" = true ] && [ "$DWI_PREPROCESSED" = false ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            echo "Running $NAME (tractography)..." | tee -a "$LOG_FILE"
            step_start=$(date +%s)

            if [ ! -d "$OUTPUT_DIR/qsiprep/sub-${SUBJECT}" ]; then
                echo "[WARN] $NAME: no QSIPrep output for sub-${SUBJECT}; skipping (template fallback)." | tee -a "$LOG_FILE"
            else
                # Adaptive recon-spec selection from the acquisition shell scheme:
                # >=2 non-zero shells -> MSMT; 1 shell with >=28 dirs -> SS3T; else skip.
                BVAL_DOCKER="${DWI_DOCKER%.nii.gz}.bval"
                RECON_CHOICE=$(docker run --rm --entrypoint micromamba \
                    -v "$BIDS_DIR":/bids:ro \
                    "$IMG_MRI_RECONSTRUCTION" \
                    run -n neuro python -c "
import numpy as np
b=np.atleast_1d(np.loadtxt('$BVAL_DOCKER'))
nz=np.sort(b[b>=100]); sh=[]
for v in nz:
    if not sh or v-sh[-1][0]>100: sh.append([v])
    else: sh[-1].append(v)
print('msmt' if len(sh)>=2 else ('ss3t' if (len(sh)==1 and len(sh[0])>=28) else 'none'))
" 2>/dev/null)

                case "$RECON_CHOICE" in
                    msmt) SPEC=parrot_multishell_msmt.yaml ;;
                    ss3t) SPEC=parrot_singleshell_ss3t.yaml ;;
                    *)    SPEC="" ;;
                esac

                if [ -z "$SPEC" ]; then
                    echo "[WARN] $NAME: shell scheme insufficient for tractography (choice='$RECON_CHOICE'); skipping (template fallback)." | tee -a "$LOG_FILE"
                    echo "Skipped: insufficient shell scheme (choice='$RECON_CHOICE')." > "$LOG_DIR/${NAME}_log.txt"
                else
                    echo "Selected recon spec: $SPEC (shell choice '$RECON_CHOICE')." | tee -a "$LOG_FILE"
                    mkdir -p "$WORK_DIR/qsirecon_out"

                    # QSIRecon reuses our FreeSurfer (ACT-hsvs); persistent TemplateFlow
                    # cache; --user avoids root-owned outputs. Output lands under the
                    # ephemeral work dir, then we relocate the results into place.
                    docker run --rm $DOCKER_GPU --user "$(id -u):$(id -g)" \
                        -e TEMPLATEFLOW_HOME=/templateflow \
                        -v "$TEMPLATEFLOW_DIR":/templateflow \
                        -v "$PARROT_SCRIPT_DIR/template_data/qsirecon_specs":/specs:ro \
                        -v "$BIDS_DIR":/bids:ro \
                        -v "$OUTPUT_DIR":/derivatives \
                        "$IMG_QSIRECON" \
                        /derivatives/qsiprep "$WORK_DIR_DOCKER/qsirecon_out" participant \
                        --participant-label "$SUBJECT" \
                        --recon-spec "/specs/$SPEC" \
                        --input-type qsiprep \
                        --fs-subjects-dir /derivatives/freesurfer \
                        --fs-license-file /bids/license.txt \
                        --nprocs "$N_THREADS" \
                        -w "$WORK_DIR_DOCKER" > "$LOG_DIR/${NAME}_log.txt" 2>&1
                    check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME"

                    # QSIRecon writes to <out>/derivatives/qsirecon-Parrot/; relocate the
                    # results into derivatives/qsirecon/ (the rest -- logs, nested
                    # derivatives -- stays in the ephemeral work dir and is swept on exit).
                    rm -rf "$OUTPUT_DIR/$NAME"
                    mv "$WORK_DIR/qsirecon_out/derivatives/qsirecon-Parrot" "$OUTPUT_DIR/$NAME"
                    check_step $? "$NAME relocation" "$LOG_DIR/${NAME}_log.txt"
                fi
            fi

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    elif [ "$HAS_DWI" = true ] && [ "$DWI_PREPROCESSED" = true ]; then
        echo "$NAME skipped for sub-${SUBJECT} (--dwi-preprocessed HCP/hcpya path not yet implemented; template fallback)." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # CONNECTIVITY
    # ---------------------------------------------------------
    # No DWI  -> copy the group-average template connectome (no container).
    # With DWI -> build the per-subject connectivity atlases here; tractography
    #             (QSIRecon) and the connectome matrices (tck2connectome) are
    #             produced by the following stages, each with its own log guard.
    NAME="connectivity"
    mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
    if [ "$HAS_DWI" = false ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            echo "Running $NAME (no DWI -> template connectome)..." | tee -a "$LOG_FILE"
            step_start=$(date +%s)

            cp "$PARROT_SCRIPT_DIR"/template_data/connectivity/* \
               "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/" 2> "$LOG_DIR/${NAME}_log.txt"
            check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt"
            echo "Copied template connectome to sub-${SUBJECT}." >> "$LOG_DIR/${NAME}_log.txt"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    else
        # Atlas preparation (subject atlas already in T1w space -> no registration).
        if [ ! -f "$LOG_DIR/${NAME}-atlas_log.txt" ]; then
            echo "Running $NAME atlas preparation..." | tee -a "$LOG_FILE"
            step_start=$(date +%s)

            run_in_docker_MRI "$NAME-atlas" "$LOG_DIR/${NAME}-atlas_log.txt" \
                "micromamba run -n neuro python /scripts/prepare_connectivity_atlas.py --output_dir /derivatives --subject $SUBJECT"

            step_end=$(date +%s)
            echo "$NAME atlas preparation completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME atlas log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    fi

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # PARROT FORWARD MODEL
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    DOCKER_IMAGE="$IMG_FORWARD_MODEL"

    # check whether sim4life reconstruction is available or not
    VOLUME_TO_MESH="simnibs"
    if [ -f "$OUTPUT_DIR/tissue_labels/sub-${SUBJECT}/electrical/sim4life.nii.gz" ]; then
        VOLUME_TO_MESH="sim4life"
    fi

    # ---------------------------------------------------------
    # PLACE ELECTRODES
    # ---------------------------------------------------------
    NAME="electrodes"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python place_electrodes.py --subject $SUBJECT --output_dir /derivatives"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # PLACE DIPOLES
    # ---------------------------------------------------------
    NAME="dipoles"
    echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
    mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

    echo "Each folder contains the sampled dipoles at the specified spacing." > "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "For each sample, four kinds of files are generated:" >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "dipole_positions.npy: (N,3) array that contains the position of each dipole in world space." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "dipole_volume.npy: (N,) array that contains the estimated volume of gray matter associated to each dipole." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "dipole_directions.npy: (N,3) array that contains the preferential direction of each dipole in world space." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "dipole_traceback.npy: (N,) boolean array, available only in the surfaces and volumetric subfolders. It traces the sub arrays back to the aggregated one. E.g. if A is the array at dipoles/dipole_positions.npy, B is the array at dipoles/volumetric/dipole_positions.npy, and M is the array at dipoles/volumetric/dipole_traceback.npy, then it is true that B = A[M]." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "orient_type.npy: (N,) array that indicates the kind of procedure used to choose a preferential direction (one of ['U', 'N', 'G', 'P', 'R'], which mean, respectively, ['Unassigned', 'Normal to a surface', 'Gradient of smoothed structure (to mimick normal to surface)', 'Principal axis of the structure', 'Randomly generated (uniformly on unit sphere)'])." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    echo "*_dipole_labels.npy: (N,) array that contains the label associated to each dipole in various atlases." >> "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/legend.txt"
    
    for s in "${SPACING_LIST[@]}"; do
        # Ensure 1 decimal point formatting
        spacing=$(printf "%.1f" "$s")

        if [ ! -f "$LOG_DIR/${NAME}-${spacing}mm_log.txt" ]; then
            echo "Placing dipoles at $spacing mm spacing..."

            step_start=$(date +%s)
            run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}-${spacing}mm_log.txt" "$DOCKER_IMAGE" "python place_dipoles.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing${DIPOLE_SEED:+ --seed $DIPOLE_SEED}"
            step_end=$(date +%s)


            echo "$NAME at $spacing mm spacing completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file at $spacing mm spacing detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
        echo
    done

    # ---------------------------------------------------------
    # MESH LABEL FIELD
    # ---------------------------------------------------------
    NAME="tetmesh"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        CONFIG_FILE="$OUTPUT_DIR"/tissuelabels/sub-${SUBJECT}/electrical/"$VOLUME_TO_MESH"_mesher_parameters.txt
        ANGLE="30.0"; DIST="2.0"; DEF_SURF="2.0"; DEF_VOL="5.0"; RATIO="3.0"; SMOOTH="2"; OPT_TIME="1800"

        # parse parameters file
        if [ ! -f "$CONFIG_FILE" ]; then echo "Error: $CONFIG_FILE not found!"; exit 1; fi
        ANGLE=$(awk    '/^ANGLE[[:space:]]/    {print $2; exit}' "$CONFIG_FILE")
        DIST=$(awk     '/^DIST[[:space:]]/     {print $2; exit}' "$CONFIG_FILE")
        DEF_SURF=$(awk '/^DEF_SURF[[:space:]]/ {print $2; exit}' "$CONFIG_FILE")
        DEF_VOL=$(awk  '/^DEF_VOL[[:space:]]/  {print $2; exit}' "$CONFIG_FILE")
        RATIO=$(awk    '/^RATIO[[:space:]]/    {print $2; exit}' "$CONFIG_FILE")
        SMOOTH=$(awk   '/^SMOOTH[[:space:]]/   {print $2; exit}' "$CONFIG_FILE")
        OPT_TIME=$(awk '/^OPT_TIME[[:space:]]/ {print $2; exit}' "$CONFIG_FILE")
        TISSUE_ARGS=($(awk '/^[[:space:]]*[0-9]/ {print $1":"$2":"$3}' "$CONFIG_FILE"))

        step_start=$(date +%s)

        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python nifti_to_inr.py --nifti_path /derivatives/tissuelabels/sub-${SUBJECT}/electrical/$VOLUME_TO_MESH.nii.gz --inr_path /derivatives/tetmesh/sub-${SUBJECT}/label_field.inr"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "mesher $N_THREADS /derivatives/tetmesh/sub-${SUBJECT}/label_field.inr /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh $ANGLE $DIST $DEF_SURF $DEF_VOL $RATIO $SMOOTH $OPT_TIME ${TISSUE_ARGS[@]}"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python mesh_postprocessing.py --reference_nifti /derivatives/tissuelabels/sub-${SUBJECT}/electrical/$VOLUME_TO_MESH.nii.gz --mesh /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --output /derivatives/tetmesh/sub-${SUBJECT}/transformed_tetrahedral_mesh.mesh --export_vtu"
 
        mv "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/transformed_tetrahedral_mesh.mesh" "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh"
        mv "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/transformed_tetrahedral_mesh.vtu" "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.vtu"
        cp "$CONFIG_FILE" "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/mesher_parameters.txt"
        cp $OUTPUT_DIR/tissuelabels/sub-${SUBJECT}/electrical/"$VOLUME_TO_MESH"_labels.txt "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/labels.txt"
        cp $OUTPUT_DIR/tissuelabels/sub-${SUBJECT}/electrical/"$VOLUME_TO_MESH"_LUT.txt "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/LUT.txt"
        cp $OUTPUT_DIR/tissuelabels/sub-${SUBJECT}/electrical/"$VOLUME_TO_MESH"_conductivities.txt "$OUTPUT_DIR/tetmesh/sub-${SUBJECT}/conductivities.txt"

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi
    
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # PARROT FORWARD SOLVERS
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    DOCKER_IMAGE="$IMG_FORWARD_SOLVERS"

    # ---------------------------------------------------------
    # SOLVE FORWARD PROBLEM
    # ---------------------------------------------------------
    NAME="forwardsolvers"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        echo "Running $NAME reconstruction..." | tee -a "$LOG_FILE"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
        mkdir -p "$OUTPUT_DIR/leadfields/sub-${SUBJECT}"

        step_start=$(date +%s)

        spacing=$(printf "%.1f" "$SPACING_OPENMEEG")
        echo "Solving forward problem with OpenMEEG at $spacing mm dipole spacing"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "/scripts/make_leadfield_openmeeg.py --dipole_spacing $spacing"

        spacing=$(printf "%.1f" "$SPACING_DUNEURO_SIMNIBS")
        echo "Solving forward problem with DUNEuro using SimNIBS charm mesh, at $spacing mm dipole spacing"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "/scripts/make_leadfield_duneuro.py --dipole_spacing $spacing --mesh_path /derivatives/simnibscharm/sub-${SUBJECT}/subject.msh --tissue_names /derivatives/simnibscharm/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/simnibscharm/sub-${SUBJECT}/conductivities.txt --label simnibs --valid_tissues \"Gray-Matter\""

        spacing=$(printf "%.1f" "$SPACING_DUNEURO_CGAL")
        echo "Solving forward problem with DUNEuro using CGAL mesh, at $spacing mm dipole spacing"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "/scripts/make_leadfield_duneuro.py --dipole_spacing $spacing --mesh_path /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --tissue_names /derivatives/tetmesh/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/tetmesh/sub-${SUBJECT}/conductivities.txt --label CGAL --valid_tissues \"Brain (Grey Matter)\" Thalamus Hippocampus"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi    


done

echo ""
echo "====================================================================="
echo "ALL SUBJECTS PROCESSED SUCCESSFULLY!"
echo "====================================================================="