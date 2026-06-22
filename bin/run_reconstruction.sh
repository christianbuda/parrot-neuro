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

log_step() {
    # Announce a pipeline step with a wall-clock start timestamp. Prints to the console
    # and appends to the per-subject summary log; keeping the timestamp format here means
    # it lives in one place instead of being repeated at every call site.
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
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
    echo "  --threads                  Number of threads to use for software that support it (Default: 32)."
    echo "  --gpus                     GPU configuration: 'all' (default), 'none', or specific devices (e.g., 'device=0,1' or '2')."
    echo "  --spacing-openmeeg         Dipole spacing (mm) for the OpenMEEG BEM solver (Default: 4)."
    echo "  --spacing-duneuro-simnibs  Dipole spacing (mm) for DUNEuro FEM with SimNIBS mesh (Default: 3)."
    echo "  --spacing-duneuro-cgal     Dipole spacing (mm) for DUNEuro FEM with CGAL mesh (Default: 2)."
    echo "  --dipole-seed              Integer seed for reproducible dipole sampling (Default: unset = random)."
    echo "  --dwi-preprocessed         Treat the BIDS dwi/ data as already corrected and skip QSIPrep (e.g. HCP)."
    echo "  --fix-inputs               Auto-repair flagged input issues (squeeze singleton 4D, snap voxel-size artifacts). Default: flag only, never mutate."
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
N_THREADS=32
GPU_OPT="all"
SPACING_OPENMEEG=4
SPACING_DUNEURO_SIMNIBS=3
SPACING_DUNEURO_CGAL=2
DIPOLE_SEED=""
DWI_PREPROCESSED=false
FIX_INPUTS=false

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
        --fix-inputs)
            FIX_INPUTS=true
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

# FreeSurfer license: required by FastSurfer (and every downstream FreeSurfer tool)
# as well as QSIPrep/QSIRecon. It must live at the BIDS dataset root as license.txt;
# the containers reach it via the /bids mount. Fail fast with a clear message rather
# than letting a stage die deep in processing with a cryptic "license not found".
if [ ! -f "$BIDS_DIR/license.txt" ]; then
    echo "ERROR: FreeSurfer license not found at $BIDS_DIR/license.txt"
    echo "       Place a valid FreeSurfer license file there (free: https://surfer.nmr.mgh.harvard.edu/registration.html)."
    exit 1
fi

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

# Re-own a subject's container-created outputs back to the host user, from inside a
# root container (so no host sudo is needed). The Parrot MRI/forward images and
# HippUnfold run as root, so without this they leave root-owned files in the
# derivatives tree -- a pain to manage or delete later. Every root-owned output lives
# under /derivatives/<stage>/sub-<ID>, so that one glob covers them all (logs/ and
# leadfields/ included); nullglob makes a subject with no outputs yet a clean no-op.
normalize_ownership() {
    local subject=$1
    [ -n "$subject" ] || return 0
    docker run --rm --entrypoint bash \
        -v "$OUTPUT_DIR":/derivatives \
        "$IMG_MRI_RECONSTRUCTION" \
        -c "shopt -s nullglob; t=(/derivatives/*/sub-${subject}); [ \${#t[@]} -gt 0 ] && chown -R $(id -u):$(id -g) \"\${t[@]}\"" \
        || echo "[WARN] ownership normalization failed for sub-${subject} (some files may remain root-owned)."
}

# Cleanup on exit: sweep the scratch work dir and re-own the in-flight subject's
# outputs (covers aborts/errors mid-subject; cleanly completed subjects are re-owned
# at the end of their loop iteration). CURRENT_SUBJECT is set at the top of each loop.
CURRENT_SUBJECT=""
cleanup() {
    [ -n "${WORK_DIR:-}" ] && rm -rf "$WORK_DIR"
    normalize_ownership "$CURRENT_SUBJECT"
}
trap cleanup EXIT INT TERM

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
    
    # --entrypoint /bin/bash overrides any internal entrypoints so we can run raw commands.
    # FS_LICENSE override: the image bakes FS_LICENSE=/SUBJECTS/license.txt, but we only
    # mount /bids and /derivatives. Point FreeSurfer at the license shipped in the BIDS
    # dataset (same file QSIPrep uses) so all recon steps can find it.
    docker run --rm $DOCKER_GPU --entrypoint /bin/bash \
        -e FS_LICENSE=/bids/license.txt \
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

    CURRENT_SUBJECT="$SUBJECT"

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
    # MP2RAGE INV2: consumed by the ingest stage (below) to MPRAGEise the UNI when the
    # 'mp2rage' tsv column is set.
    INV2_PATH=$(find "$SUB_BIDS_DIR/anat" -name "sub-${SUBJECT}*_inv-2*.nii.gz" 2>/dev/null | head -n 1)

    if [ -z "$T1_PATH" ]; then
        echo "[ERROR] No T1w image found for sub-${SUBJECT}. Skipping..." | tee -a "$LOG_FILE"
        continue
    fi
    echo "Found T1w: $T1_PATH" | tee -a "$LOG_FILE"

    # BIDS inputs (read-only) the ingest stage consumes...
    T1_BIDS_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$T1_PATH")"
    [ -n "$T2_PATH" ] && T2_BIDS_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$T2_PATH")"
    [ -n "$INV2_PATH" ] && INV2_BIDS_DOCKER="/bids/sub-${SUBJECT}/anat/$(basename "$INV2_PATH")"

    # ...and the standardized working inputs the ingest stage WRITES, which every later
    # stage reads. T1_DOCKER is MPRAGEised for MP2RAGE, a clean copy otherwise.
    RAW_DIR_DOCKER="/derivatives/raw/sub-${SUBJECT}"
    T1_DOCKER="$RAW_DIR_DOCKER/T1.nii.gz"

    # T2 (if present) feeds SimNIBS charm. FastSurfer surfaces are T1-only, so the old
    # recon-all T2/FLAIR pial refinement is gone -- FLAIR is no longer consulted at all.
    simnibs_args=()
    if [ -n "$T2_PATH" ]; then
        echo "Found T2w: $T2_PATH (used by SimNIBS charm)" | tee -a "$LOG_FILE"
        T2_DOCKER="$RAW_DIR_DOCKER/T2.nii.gz"
        simnibs_args=("$T2_DOCKER")
    else
        echo "No T2w found for sub-${SUBJECT}." | tee -a "$LOG_FILE"
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

    # TSV Overrides (positional columns: 4=skip-T2-reg, 5=no-neck, 6=mp2rage)
    MP2RAGE_SUBJECT=false
    if [ -f "$BIDS_DIR/participants.tsv" ]; then
        SUB_ROW=$(grep "^sub-${SUBJECT}" "$BIDS_DIR/participants.tsv" 2>/dev/null)
        if [ -n "$SUB_ROW" ]; then
            if [ "$(echo "$SUB_ROW" | awk '{print tolower($4)}')" == "true" ]; then
                simnibs_args+=("--skipregisterT2")
            fi
            if [ "$(echo "$SUB_ROW" | awk '{print tolower($5)}')" == "true" ]; then
                simnibs_args+=("--noneck")
            fi
            # col6 mp2rage: T1 is an MP2RAGE UNI -> MPRAGEise it (FastSurfer/charm can't
            # consume the raw UNI's high-intensity background). Handled by ingest below.
            if [ "$(echo "$SUB_ROW" | awk '{print tolower($6)}')" == "true" ]; then
                MP2RAGE_SUBJECT=true
            fi
        fi
    fi

    # =========================================================================
    # 5. EXECUTE PIPELINE STEPS (With Robust Idempotency)
    # =========================================================================

    start_time=$(date +%s)

    # ---------------------------------------------------------
    # INGEST (stage 0): validate inputs + standardize into raw/
    # ---------------------------------------------------------
    # Runs FIRST, for EVERY subject. Validates each anatomical input (loadable .nii.gz,
    # genuinely 3D, sane voxel size) and writes the standardized working inputs every
    # later stage reads: /derivatives/raw/sub-X/{T1,T2}.nii.gz + per-volume JSON
    # provenance sidecars. T1 is MPRAGEised for MP2RAGE (N3-corrected INV2 soft-weight),
    # a clean copy otherwise. --fix-inputs lets it auto-repair flagged issues (squeeze a
    # singleton 4D, snap a float32 voxel-size artifact); without it those are flagged,
    # and a bad shape is fatal. (DWI stays on the BIDS dwi/ path for QSIPrep.)
    NAME="ingest"
    if [ "$MP2RAGE_SUBJECT" = true ] && [ -z "$INV2_PATH" ]; then
        echo "[ERROR] sub-${SUBJECT} flagged mp2rage but INV2 not found in anat/. Skipping subject." | tee -a "$LOG_FILE"
        continue
    fi
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        log_step "Running $NAME (validate + standardize inputs -> raw/)..."
        mkdir -p "$OUTPUT_DIR/raw/sub-${SUBJECT}"
        step_start=$(date +%s)
        # Under FreeSurfer's fspython: ingest shells out to mri_nu_correct.mni (N3, for
        # MPRAGEise) and imports nibabel, both in FreeSurfer's env.
        INGEST_CMD="fspython /scripts/ingest.py --out-dir $RAW_DIR_DOCKER --t1 $T1_BIDS_DOCKER"
        [ -n "$T2_PATH" ] && INGEST_CMD="$INGEST_CMD --t2 $T2_BIDS_DOCKER"
        [ "$MP2RAGE_SUBJECT" = true ] && INGEST_CMD="$INGEST_CMD --mp2rage --inv2 $INV2_BIDS_DOCKER"
        [ "$FIX_INPUTS" = true ] && INGEST_CMD="$INGEST_CMD --fix-inputs"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "$INGEST_CMD"
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # FASTSURFER (segmentation + surfaces; replaces recon-all)
    # ---------------------------------------------------------
    # Full seg+surf run: the CNN segmentation gives the CerebNet/HypVINN subsegs AND a
    # FreeSurfer-format SUBJECTS_DIR (surf/, label/, mri/, ?h.sphere.reg) that every
    # downstream stage (MNE BEM, Schaefer projection, segment_subregions, charm,
    # QSIRecon) consumes -- so this single stage replaces both the old seg-only run and
    # recon-all. NOTE: inputs must be cleanly <=1mm; FastSurfer's surf-stage conform
    # rejects vox_size > 1.0, so a float32 voxel-size header artifact silently kills the
    # surfaces (it still exits 0). The LEMON staging cleans this; see the surface guard.
    NAME="fastsurfer"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        log_step "Running $NAME reconstruction (seg + surfaces)..."
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
            --3T --threads "$N_THREADS" --parallel > "$LOG_DIR/${NAME}_log.txt" 2>&1
        fastsurfer_rc=$?

        # FastSurfer can exit 0 even when the surf stage dies (e.g. a rejected vox_size),
        # so $? alone is not enough -- assert the surfaces were actually produced.
        if [ "$fastsurfer_rc" -eq 0 ] && [ ! -f "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/surf/lh.white" ]; then
            echo "[ERROR] FastSurfer exited 0 but produced no surfaces (surf/lh.white missing)." | tee -a "$LOG_FILE"
            echo "        Most likely the input voxel size is > 1mm (header artifact); it must be <= 1mm." | tee -a "$LOG_FILE"
            fastsurfer_rc=1
        fi
        check_step "$fastsurfer_rc" "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        # LUTs the atlas stage later reads from the subject dir (recon-all used to copy
        # these in). Run in the MRI image -- the Schaefer LUT ships there, not in the
        # FastSurfer image. Append to the same log; FREESURFER_HOME is set in that image.
        docker run --rm --entrypoint /bin/bash -v "$OUTPUT_DIR":/derivatives \
            "$IMG_MRI_RECONSTRUCTION" -c \
            "cp \$FREESURFER_HOME/FreeSurferColorLUT.txt /derivatives/fastsurfer/sub-${SUBJECT}/FreeSurferColorLUT.txt && \
             cp -r /home/Schaefer2018_LocalGlobal/Parcellations/project_to_individual /derivatives/fastsurfer/sub-${SUBJECT}/Schaefer_LUT" \
            >> "$LOG_DIR/${NAME}_log.txt" 2>&1 \
            || { echo "[ERROR] failed to copy LUTs into fastsurfer/sub-${SUBJECT}" | tee -a "$LOG_FILE"; exit 1; }

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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)
        # HippUnfold (:latest) writes its derivatives to <out>/hippunfold/sub-XXX and
        # also litters <out> with work/, .snakemake/, config/. Pointing it straight at
        # $OUTPUT_DIR/hippunfold would double-nest (hippunfold/hippunfold/sub-XXX) and
        # leave scratch behind. So run it into a throwaway temp dir, then lift just the
        # sub-XXX tree into $OUTPUT_DIR/hippunfold/sub-XXX where the pipeline expects it.
        # No -it: a TTY isn't available in non-interactive/background runs and breaks
        # with "the input device is not a TTY"; this batch BIDS app doesn't need one.
        HIPPUNFOLD_TMP=$(mktemp -d "$OUTPUT_DIR/.hippunfold_tmp.XXXXXX")
        docker run --rm \
            -v "$BIDS_DIR":/bids:ro \
            -v "$HIPPUNFOLD_TMP":/output \
            "$IMG_HIPPUNFOLD" \
            /bids /output participant \
            --participant_label "$SUBJECT" \
            --modality T1w --cores "$N_THREADS" > "$LOG_DIR/${NAME}_log.txt" 2>&1
        hippunfold_rc=$?

        # Lift the real derivatives out of the nested hippunfold/ subdir, drop the scratch.
        if [ -d "$HIPPUNFOLD_TMP/hippunfold/sub-${SUBJECT}" ]; then
            rm -rf "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
            mkdir -p "$OUTPUT_DIR/$NAME"
            mv "$HIPPUNFOLD_TMP/hippunfold/sub-${SUBJECT}" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
        fi
        rm -rf "$HIPPUNFOLD_TMP"

        check_step $hippunfold_rc "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

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
    # MNE BEM SURFACES
    # ---------------------------------------------------------
    NAME="mne"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        log_step "Running $NAME reconstruction..."

        step_start=$(date +%s)
        # MP2RAGE: run watershed on INV2, not the UNI/MPRAGEised T1. INV2 keeps the
        # whole-head contrast (bright scalp/fat, dark skull, dark air) that mri_watershed
        # needs and the UNI optimizes away -> on the UNI, watershed collapses to a cube.
        # Conform INV2 onto the subject's mri grid first; the dense scalp still comes from
        # T1.mgz (its head/air boundary is clean). No-op for non-MP2RAGE subjects.
        bem_prep=""
        bem_vol=""
        if [ "$MP2RAGE_SUBJECT" = true ]; then
            bem_prep="mri_convert --conform $INV2_BIDS_DOCKER \$FREESURFER_HOME/subjects/$SUBJECT/mri/INV2.mgz && "
            bem_vol="--volume INV2"
        fi
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" \
            "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && \
             ${bem_prep}micromamba run -n neuro python /scripts/make_bem_surfaces.py --subject $SUBJECT --subjects_dir \$FREESURFER_HOME/subjects $bem_vol"

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
        log_step "Running $NAME reconstruction..."
        
        step_start=$(date +%s)

        for n_parcels in {100..1000..100}; do
            ATLAS_NAME="Schaefer2018_${n_parcels}Parcels_17Networks_order"

            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/lh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/lh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/rh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/rh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_aparc2aseg --s $SUBJECT --o \$SUBJECTS_DIR/$SUBJECT/mri/schaefer${n_parcels}_aparc+aseg.mgz --annot $ATLAS_NAME"
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
        log_step "Running $NAME reconstruction..."
        
        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions thalamus --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions hippo-amygdala --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/fastsurfer/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions brainstem --cross $SUBJECT --threads $N_THREADS"

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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cd /home/simnibs_reconstructions && \
                                                        /root/SimNIBS-4.5/bin/charm subject $T1_DOCKER ${simnibs_args[*]} --forcerun --fs-dir /derivatives/fastsurfer/sub-${SUBJECT} --forcesform && \
                                                        cd / && \
                                                        /root/SimNIBS-4.5/bin/simnibs_python /scripts/extract_charm_surf.py --charm_dir "/home/simnibs_reconstructions/m2m_subject/" && \
                                                        cp /scripts/simnibs_conductivities.txt /home/simnibs_reconstructions/m2m_subject/conductivities.txt && \
                                                        cp /scripts/simnibs_labels.txt /home/simnibs_reconstructions/m2m_subject/labels.txt && \
                                                        mv /home/simnibs_reconstructions/m2m_subject /derivatives/$NAME/sub-${SUBJECT}"

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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        # bias field correct image and then run FSL first. Paths are /derivatives/...
        # (the in-container mount), not $OUTPUT_DIR (host path), which doesn't exist
        # inside the container and made the writes fail with "cannot open output file".
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/bias_correct.py $T1_DOCKER /derivatives/$NAME/sub-${SUBJECT}/T1.nii.gz && \
	                                                    /scripts/run_first_all_sequential -i /derivatives/$NAME/sub-${SUBJECT}/T1.nii.gz -o /derivatives/$NAME/sub-${SUBJECT}/FSL -v"

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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        synth_flag=()
        if [ -n "$DOCKER_GPU" ] ; then
            synth_flag=("--gpu")
        fi

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "mri_synthstrip -i "$T1_DOCKER" -o /derivatives/$NAME/sub-${SUBJECT}/T1_stripped.nii.gz -m /derivatives/$NAME/sub-${SUBJECT}/T1_stripped_mask.nii.gz ${synth_flag[*]} && \
	                                                    mri_synthstrip -i "$T1_DOCKER" -o /derivatives/$NAME/sub-${SUBJECT}/T1_noCSF_stripped.nii.gz -m /derivatives/$NAME/sub-${SUBJECT}/T1_noCSF_stripped_mask.nii.gz ${synth_flag[*]} --no-csf"

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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cp /home/cerebellum_template/Cerebellar_Regions.csv /derivatives/$NAME/sub-${SUBJECT}/LABELS.csv && \
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
        log_step "Running $NAME reconstruction..."
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
        log_step "Running $NAME reconstruction..."
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
        log_step "Running $NAME reconstruction..."
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
        log_step "Running $NAME reconstruction..."
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/electrical"
        mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/acoustic"

        step_start=$(date +%s)

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cp /scripts/simnibs_mesher_parameters.txt /derivatives/$NAME/sub-${SUBJECT}/electrical/ && \
                                                        cp /scripts/simnibs_mesher_parameters.txt /derivatives/$NAME/sub-${SUBJECT}/electrical/simnibs_itis_mesher_parameters.txt && \
                                                        cp /scripts/sim4life_mesher_parameters.txt /derivatives/$NAME/sub-${SUBJECT}/electrical/ && \
                                                        micromamba run -n neuro python /scripts/gather_electrical_labelfields.py --output_dir /derivatives --subject $SUBJECT && \
                                                        micromamba run -n neuro python /scripts/gather_acoustic_labelfields.py --output_dir /derivatives --subject $SUBJECT"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi

        
    # ---------------------------------------------------------
    # QSIPREP (DWI PREPROCESSING)
    # ---------------------------------------------------------
    # Optional. Runs only when usable raw DWI is present and not flagged as
    # already preprocessed. Produces the corrected/aligned DWI (no diffusion
    # model fit -- modeling belongs to the recon stage) in subject anatomical
    # space under derivatives/qsiprep/, the input contract for the recon stage.
    # QSIPrep runs its own anatomical workflow (SynthStrip/TemplateFlow) and does
    # not consume our FreeSurfer dir -- FreeSurfer reuse belongs to the recon stage.
    NAME="qsiprep"
    if [ "$HAS_DWI" = true ] && [ "$DWI_PREPROCESSED" = false ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (DWI preprocessing)..."
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

            # Expose only a RAW view of /bids -- dataset metadata + this
            # subject's raw folder -- never the derivatives tree. qsiprep's
            # pybids indexer (qsiprep/config.py) is built with an explicit
            # `ignore` list that does NOT skip derivatives; that explicit list
            # replaces pybids' default (which would). So if $OUTPUT_DIR is
            # nested inside $BIDS_DIR (the BIDS-standard derivatives/ layout)
            # and we mount the whole $BIDS_DIR, pybids walks our derivatives and
            # crashes on Parrot JSON that isn't a sidecar -- e.g.
            # atlas_to_aggregated.json is a top-level JSON array, so
            # dict.update(list) -> "unhashable type: list". Binding only raw
            # inputs sidesteps this regardless of where the output dir lives.
            QSIPREP_BIDS_MOUNTS=(
                -v "$BIDS_DIR/dataset_description.json":/bids/dataset_description.json:ro
                -v "$BIDS_DIR/license.txt":/bids/license.txt:ro
                -v "$BIDS_DIR/sub-${SUBJECT}":/bids/sub-${SUBJECT}:ro
            )
            # participants.tsv is optional; bind it only if present -- a missing
            # bind source makes Docker silently create an empty dir at /bids.
            if [ -f "$BIDS_DIR/participants.tsv" ]; then
                QSIPREP_BIDS_MOUNTS+=( -v "$BIDS_DIR/participants.tsv":/bids/participants.tsv:ro )
            fi

            # --user avoids root-owned outputs in the NFS derivatives tree; the
            # image's HOME (/home/qsiprep) is world-writable so we keep it as-is.
            # TemplateFlow is cached persistently across runs.
            docker run --rm $DOCKER_GPU --user "$(id -u):$(id -g)" \
                -e TEMPLATEFLOW_HOME=/templateflow \
                -v "$TEMPLATEFLOW_DIR":/templateflow \
                "${QSIPREP_BIDS_MOUNTS[@]}" \
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
            log_step "Running $NAME (tractography)..."
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
                    # qsirecon's pybids input is /derivatives/qsiprep (a clean
                    # tree), so it doesn't hit the derivatives-walk crash above.
                    # We still bind only license.txt from $BIDS_DIR (not the whole
                    # dataset) to avoid needlessly exposing the derivatives tree.
                    docker run --rm $DOCKER_GPU --user "$(id -u):$(id -g)" \
                        -e TEMPLATEFLOW_HOME=/templateflow \
                        -v "$TEMPLATEFLOW_DIR":/templateflow \
                        -v "$PARROT_SCRIPT_DIR/template_data/qsirecon_specs":/specs:ro \
                        -v "$BIDS_DIR/license.txt":/bids/license.txt:ro \
                        -v "$OUTPUT_DIR":/derivatives \
                        "$IMG_QSIRECON" \
                        /derivatives/qsiprep "$WORK_DIR_DOCKER/qsirecon_out" participant \
                        --participant-label "$SUBJECT" \
                        --recon-spec "/specs/$SPEC" \
                        --input-type qsiprep \
                        --fs-subjects-dir /derivatives/fastsurfer \
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
    # DWI present and tractography succeeded -> subject connectome:
    #   atlas preparation  +  tck2connectome (run in the QSIRecon image).
    # No DWI, or DWI too sparse for tractography -> group-average template
    # connectome. Each sub-step has its own log guard.
    NAME="connectivity"
    mkdir -p "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

    # Did QSIRecon produce a tractogram for this subject?
    HAVE_TRACKS=false
    if [ "$HAS_DWI" = true ] && \
       compgen -G "$OUTPUT_DIR/qsirecon/sub-${SUBJECT}/dwi/"*streamlines.tck.gz > /dev/null 2>&1; then
        HAVE_TRACKS=true
    fi

    if [ "$HAS_DWI" = true ]; then
        # Atlas preparation (subject atlas already in T1w space -> no registration).
        if [ ! -f "$LOG_DIR/${NAME}-atlas_log.txt" ]; then
            log_step "Running $NAME atlas preparation..."
            step_start=$(date +%s)

            run_in_docker_MRI "$NAME-atlas" "$LOG_DIR/${NAME}-atlas_log.txt" \
                "micromamba run -n neuro python /scripts/prepare_connectivity_atlas.py --output_dir /derivatives --subject $SUBJECT"

            step_end=$(date +%s)
            echo "$NAME atlas preparation completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME atlas log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    fi

    if [ "$HAVE_TRACKS" = true ]; then
        # Connectome matrices via tck2connectome, run in the QSIRecon image (same
        # MRtrix3 that generated the tracks). --user keeps outputs user-owned.
        if [ ! -f "$LOG_DIR/${NAME}-matrices_log.txt" ]; then
            log_step "Running $NAME matrices (tck2connectome)..."
            step_start=$(date +%s)

            docker run --rm --user "$(id -u):$(id -g)" \
                -v "$OUTPUT_DIR":/derivatives \
                -v "$PARROT_SCRIPT_DIR/bin/make_connectomes.sh":/make_connectomes.sh:ro \
                --entrypoint bash "$IMG_QSIRECON" \
                /make_connectomes.sh "$SUBJECT" > "$LOG_DIR/${NAME}-matrices_log.txt" 2>&1
            check_step $? "$NAME matrices" "$LOG_DIR/${NAME}-matrices_log.txt"

            step_end=$(date +%s)
            echo "$NAME matrices completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME matrices log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    else
        # No DWI, or DWI too sparse -> group-average template connectome.
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (template connectome fallback)..."
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
    fi

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # PARROT FORWARD MODEL
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    DOCKER_IMAGE="$IMG_FORWARD_MODEL"

    # Pick the volume meshed by the CGAL stream (and the valid generator tissues for its FEM
    # leadfield). Prefer Sim4Life if present; otherwise fall back to "simnibs_itis" -- the SimNIBS
    # segmentation carrying ITIS conductivities -- so the CGAL leadfield uses ITIS values while the
    # SimNIBS-charm leadfield keeps native SimNIBS conductivities. Unlike Sim4Life, the SimNIBS
    # segmentation has no separate thalamus/hippocampus, so grey matter is the only generator tissue.
    VOLUME_TO_MESH="simnibs_itis"
    CGAL_VALID_TISSUES='"Brain (Grey Matter)"'
    if [ -f "$OUTPUT_DIR/tissuelabels/sub-${SUBJECT}/electrical/sim4life.nii.gz" ]; then
        VOLUME_TO_MESH="sim4life"
        CGAL_VALID_TISSUES='"Brain (Grey Matter)" Thalamus Hippocampus'
    fi

    # ---------------------------------------------------------
    # PLACE ELECTRODES
    # ---------------------------------------------------------
    NAME="electrodes"
    if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
        log_step "Running $NAME reconstruction..."
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
    log_step "Running $NAME reconstruction..."
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
        log_step "Running $NAME reconstruction..."
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
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "mesher $N_THREADS /derivatives/tetmesh/sub-${SUBJECT}/label_field.inr /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh $ANGLE $DIST $DEF_SURF $DEF_VOL $RATIO $SMOOTH $OPT_TIME ${TISSUE_ARGS[*]}"
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
        log_step "Running $NAME reconstruction..."
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
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "/scripts/make_leadfield_duneuro.py --dipole_spacing $spacing --mesh_path /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --tissue_names /derivatives/tetmesh/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/tetmesh/sub-${SUBJECT}/conductivities.txt --label CGAL --valid_tissues $CGAL_VALID_TISSUES"
 
        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi    


    # All stages done for this subject -- re-own its outputs now (don't wait for the
    # whole batch). Clear CURRENT_SUBJECT so the exit trap won't redundantly re-chown
    # a subject that already completed cleanly.
    normalize_ownership "$SUBJECT"
    CURRENT_SUBJECT=""
done

echo ""
echo "====================================================================="
echo "ALL SUBJECTS PROCESSED SUCCESSFULLY!"
echo "====================================================================="