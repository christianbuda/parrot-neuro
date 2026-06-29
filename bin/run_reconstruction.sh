#!/bin/bash

###############################################################################
# PARROT MRI RECONSTRUCTION - BIDS APP
###############################################################################

# Idempotency model (robust to ANY stop, incl. SIGKILL/crash/OOM/power-loss):
#   - A step's idempotency marker is its final "<name>_log.txt". It must exist IFF the
#     step completed successfully, so a step's live output is written to "<log>.partial"
#     and only RENAMED to "<log>" by check_step on success (see below). An interruption
#     therefore leaves only ".partial" (never the marker), so the step re-runs.
#   - begin_step() is called when a step is (re-)run: it clears any stale ".partial" and
#     WIPES the step's output dir, so a partial left by a previously-killed attempt cannot
#     survive into the rerun (some steps don't overwrite old files -- e.g. per-spacing
#     dipoles). Pass the output path(s) the step produces.
begin_step() {
    local log_file=$1; shift
    rm -f "${log_file}.partial"
    local p
    for p in "$@"; do
        [ -n "$p" ] && rm -rf "$p"
    done
}

check_step() {
    local exit_code=$1    # The exit code of the command you just ran
    local description=$2  # Text description
    local log_file=$3     # Where the logs are stored
    local cleanup_path=$4 # Optional, path to remove if error is detected
    local partial="${log_file}.partial"  # live output lives here until finalised

    if [ "$exit_code" -ne 0 ]; then
        echo "[ERROR] $description failed! (Exit Code: $exit_code)"
        echo "Check log file for more info: $log_file"

        # Remove the partial/intermediate output so a rerun starts from a clean slate.
        if [ -n "$cleanup_path" ] && [ -d "$cleanup_path" ]; then
            echo "Cleaning up incomplete directory: $cleanup_path"
            rm -rf "$cleanup_path"
        fi

        # Preserve the log under a FAILED_<timestamp>_ prefix for debugging, but NEVER
        # leave it as the success marker. We rename the .partial (the live log); fall back
        # to an already-finalised log only for safety.
        local src="$partial"; [ -f "$src" ] || src="$log_file"
        if [ -n "$src" ] && [ -f "$src" ]; then
            local failed_log
            failed_log="$(dirname "$log_file")/FAILED_$(date '+%Y%m%d-%H%M%S')_$(basename "$log_file")"
            mv "$src" "$failed_log"
            echo "Renamed failed log to: $failed_log"
        fi

        echo
        exit 1
    fi

    # SUCCESS: commit the completion marker. Only now does <name>_log.txt exist.
    [ -f "$partial" ] && mv -f "$partial" "$log_file"
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
    echo "  --dwi-preprocessed FORMAT  DWI is already preprocessed; skip QSIPrep. FORMAT is 'qsiprep' (a qsiprep-"
    echo "                             derivatives tree already at <output_dir>/qsiprep/) or 'hcp' (HCP-YA, staged under"
    echo "                             <bids_dir>/sourcedata/hcp/<ID>/; uses QSIRecon --input-type hcpya)."
    echo "  --fix-inputs               Auto-repair flagged input issues (squeeze singleton 4D, snap voxel-size artifacts). Default: flag only, never mutate."
    echo "  --recon                    Surface recon backend: 'fastsurfer' (default, seg+surf) or 'freesurfer' (recon-all surfaces + FastSurfer --seg_only for CNN subsegs)."
    echo "  --runtime                  Container runtime: 'docker' (default, workstation) or 'apptainer' (rootless, for HPC like CINECA LEONARDO)."
    echo "  --sif-dir                  Directory holding/caching .sif images (apptainer only; default: <output_dir>/.sif). Pulled from Docker Hub on first use."
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
DWI_FORMAT=""           # "" = raw DWI in BIDS dwi/ (run QSIPrep); else a preprocessed format
FIX_INPUTS=false
RECON_BACKEND=fastsurfer
RUNTIME=docker          # container runtime: "docker" (workstation) or "apptainer" (HPC, e.g. LEONARDO)
SIF_DIR=""              # where .sif images live/are pulled (apptainer only); default set after parsing

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
            DWI_FORMAT="$2"
            shift 2
            ;;
        --fix-inputs)
            FIX_INPUTS=true
            shift
            ;;
        --recon)
            RECON_BACKEND="$2"
            shift 2
            ;;
        --runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --sif-dir)
            SIF_DIR="$2"
            shift 2
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

# Validate the recon backend selection.
if [ "$RECON_BACKEND" != "fastsurfer" ] && [ "$RECON_BACKEND" != "freesurfer" ]; then
    echo "ERROR: --recon must be 'fastsurfer' or 'freesurfer' (got '$RECON_BACKEND')."
    usage
fi

# Validate the preprocessed-DWI format selection ("" = raw / run QSIPrep).
if [ -n "$DWI_FORMAT" ] && [ "$DWI_FORMAT" != "qsiprep" ] && [ "$DWI_FORMAT" != "hcp" ]; then
    echo "ERROR: --dwi-preprocessed must be 'qsiprep' or 'hcp' (got '$DWI_FORMAT')."
    usage
fi

# Validate the container runtime and resolve the .sif cache dir (apptainer only).
if [ "$RUNTIME" != "docker" ] && [ "$RUNTIME" != "apptainer" ]; then
    echo "ERROR: --runtime must be 'docker' or 'apptainer' (got '$RUNTIME')."
    usage
fi
# Pick the runtime CLI: Apptainer and SingularityCE share an identical CLI for everything
# we use (exec/run/pull/--nv/--bind/--pwd), and clusters ship one or the other. Prefer
# `apptainer`, fall back to `singularity`, so the same code runs on either.
APPTAINER_BIN="apptainer"
if [ "$RUNTIME" = "apptainer" ]; then
    if command -v apptainer &>/dev/null; then APPTAINER_BIN="apptainer"
    elif command -v singularity &>/dev/null; then APPTAINER_BIN="singularity"
    else
        echo "ERROR: --runtime apptainer requested but neither 'apptainer' nor 'singularity' is on PATH."
        echo "       On a module-based HPC you likely need e.g. 'module load apptainer' first."
        exit 1
    fi
    [ -z "$SIF_DIR" ] && SIF_DIR="$OUTPUT_DIR/.sif"
fi

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

# GPU Configuration Logic. DOCKER_GPU and APPTAINER_NV are set together from the same
# detection so a CPU-only context (GPU disabled, or no driver -- e.g. a SLURM DCGP/CPU
# node) cleanly yields *no* GPU flag for whichever runtime is active. On the GPU side,
# `--gpus` (docker) / `--nv` (apptainer) expose whatever the host/scheduler granted, so
# requesting a single GPU via SLURM `--gres=gpu:1` needs no change here.
if [ "$GPU_OPT" == "none" ]; then
    DOCKER_GPU=""
    APPTAINER_NV=""
    echo "Notice: GPU disabled by user. Running in CPU-only mode."
else
    # Check if nvidia-smi exists and can talk to the driver
    if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        DOCKER_GPU=""
        APPTAINER_NV=""
        echo "WARNING: nvidia-smi not found or driver missing. Falling back to CPU-only mode."
    else
        DOCKER_GPU="--gpus $GPU_OPT"
        APPTAINER_NV="--nv"
        echo "GPU Configuration: runtime=$RUNTIME, docker='$DOCKER_GPU', apptainer='$APPTAINER_NV'"
    fi
fi

# Map a docker image tag to its .sif filename in SIF_DIR, e.g.
#   christianbuda/parrot_mri_reconstruction:latest -> $SIF_DIR/parrot_mri_reconstruction_latest.sif
# (drop the registry/namespace, turn the :tag separator into _). Used by the apptainer path.
sif_path() {
    local tag=$1
    local base=${tag##*/}     # strip registry/namespace
    base=${base//:/_}         # tag separator -> underscore
    echo "$SIF_DIR/${base}.sif"
}

# Ensure required container images are present (pull any that are missing). Image list comes
# from bin/images.sh so build/pull/run never drift. For docker we pull tags into the local
# daemon; for apptainer we pull each docker:// image once into a flattened .sif under SIF_DIR
# (a plain file, reusable across subjects and SLURM array tasks).
ALL_IMAGES=("${EXTERNAL_IMAGES[@]}")
for entry in "${PARROT_IMAGES[@]}"; do
    ALL_IMAGES+=("${entry%%|*}")
done

echo "Checking required container images (runtime: $RUNTIME)..."
if [ "$RUNTIME" = "apptainer" ]; then
    mkdir -p "$SIF_DIR"
    for img in "${ALL_IMAGES[@]}"; do
        sif=$(sif_path "$img")
        if [ -f "$sif" ]; then
            echo "  Found $(basename "$sif")"
        else
            echo "  Missing $(basename "$sif") - pulling docker://$img (this may take a while)..."
            if ! "$APPTAINER_BIN" pull "$sif" "docker://$img"; then
                echo "ERROR: Failed to pull docker://$img into $sif"
                exit 1
            fi
        fi
    done
else
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
fi

# Scratch work directory for BIDS apps that need one (e.g. QSIPrep/nipype, which
# can balloon to tens of GB). Placed inside the output dir so it shares the large
# derivatives filesystem rather than a possibly RAM-backed /tmp, and removed when
# the script exits (success, error, or interrupt). Ephemeral by design: a failed
# QSIPrep run therefore restarts from scratch on re-run (no nipype resume cache).
mkdir -p "$OUTPUT_DIR"   # may be a brand-new output dir; mktemp below needs it to exist
WORK_DIR=$(mktemp -d "$OUTPUT_DIR/.parrot_work.XXXXXX")
WORK_DIR_DOCKER="/derivatives/$(basename "$WORK_DIR")"

# Cleanup on exit: sweep the scratch work dir. Every container now runs as the invoking
# user (docker --user / apptainer are both rootless), so outputs are already user-owned --
# the old root-ownership normalization (a chown from inside a root container) is gone.
CURRENT_SUBJECT=""
cleanup() {
    [ -n "${WORK_DIR:-}" ] && rm -rf "$WORK_DIR"
}
# Graceful stop: previously `trap cleanup EXIT INT TERM` ran cleanup but, since cleanup
# does not exit, the script RESUMED after SIGINT/SIGTERM -- so the only way to stop it was
# SIGKILL, which bypasses check_step and leaves false "completed" logs. Now INT/TERM exit
# (the EXIT trap then runs cleanup once), so SIGTERM stops the pipeline cleanly.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
# CONTAINER RUNTIME ABSTRACTION
# =============================================================================
# container_exec emits the right invocation for the active RUNTIME (docker | apptainer) and
# runs it. Both paths run rootless (docker --user / apptainer-as-you), so the only real
# difference is flag syntax: --gpus/--nv, -v/--bind, -e/--env, -w/--pwd, --entrypoint vs
# `exec <exe>` / `run`. Callers set these globals before each call (optionals auto-reset):
#   CE_GPU   1 to expose the GPU (default 0)
#   CE_HOME  1 to inject a writable managed HOME (default 0; required for the custom Parrot
#            images, which -- unlike the external BIDS apps -- have no world-writable HOME)
#   CE_EXEC  entrypoint executable to run (default: use the image's own entrypoint/runscript)
#   CE_PWD   working directory inside the container (default: image default)
#   CE_ENVS  array of "KEY=VALUE" env vars
#   CE_BINDS array of "src:dst[:ro]" mounts
# Usage: container_exec <image_tag> [args...]   (stdout/stderr are the caller's to redirect)
container_exec() {
    local image=$1; shift
    local -a cmd=( "$@" )
    local rc x
    if [ "$RUNTIME" = "apptainer" ]; then
        local sif; sif=$(sif_path "$image")
        local -a a=( "$APPTAINER_BIN" )
        # `exec <sif> <exe> <args>` when an entrypoint override is given; otherwise `run` the
        # image's own runscript (the converted Docker ENTRYPOINT) with <args>.
        if [ -n "${CE_EXEC:-}" ]; then a+=( exec ); else a+=( run ); fi
        # Apptainer manages HOME specially and REJECTS setting it via --env (it warns
        # "Overriding HOME ... with APPTAINERENV_HOME is not permitted"). The supported way is
        # --home <src>:<dest>, which bind-mounts a writable host scratch at /parrot_home AND
        # sets HOME to it, replacing the default host-$HOME mount (so we never touch the
        # quota'd home on HPC). USER fills the missing /etc/passwd entry (getpass); it is a
        # normal env var, so --env is fine for it.
        a+=( --home "$PARROT_HOME_HOST:/parrot_home" --env USER=parrot )
        [ "${CE_GPU:-0}" = 1 ] && [ -n "$APPTAINER_NV" ] && a+=( $APPTAINER_NV )
        for x in "${CE_ENVS[@]:-}";  do [ -n "$x" ] && a+=( --env "$x" ); done
        for x in "${CE_BINDS[@]:-}"; do [ -n "$x" ] && a+=( --bind "$x" ); done
        [ -n "${CE_PWD:-}" ] && a+=( --pwd "$CE_PWD" )
        a+=( "$sif" )
        [ -n "${CE_EXEC:-}" ] && a+=( "$CE_EXEC" )
        a+=( "${cmd[@]}" )
        "${a[@]}"; rc=$?
    else
        local -a a=( docker run --rm --user "$(id -u):$(id -g)" )
        # External BIDS apps keep their own world-writable HOME on docker (as before); the
        # custom Parrot images get the managed writable HOME when the caller asks (CE_HOME=1).
        if [ "${CE_HOME:-0}" = 1 ]; then
            a+=( -v "$PARROT_HOME_HOST:/parrot_home" -e HOME=/parrot_home -e USER=parrot )
        fi
        [ "${CE_GPU:-0}" = 1 ] && [ -n "$DOCKER_GPU" ] && a+=( $DOCKER_GPU )
        for x in "${CE_ENVS[@]:-}";  do [ -n "$x" ] && a+=( -e "$x" ); done
        for x in "${CE_BINDS[@]:-}"; do [ -n "$x" ] && a+=( -v "$x" ); done
        [ -n "${CE_PWD:-}" ] && a+=( -w "$CE_PWD" )
        [ -n "${CE_EXEC:-}" ] && a+=( --entrypoint "$CE_EXEC" )
        a+=( "$image" )
        a+=( "${cmd[@]}" )
        "${a[@]}"; rc=$?
    fi
    # Reset optionals so they never leak into the next call (arrays are reset by callers).
    CE_GPU=0; CE_HOME=0; CE_EXEC=""; CE_PWD=""; CE_ENVS=(); CE_BINDS=()
    return $rc
}

# Run a step in the MRI reconstruction image, sourcing its environment first. FS_LICENSE
# override: the image bakes FS_LICENSE=/SUBJECTS/license.txt, but we only mount /bids and
# /derivatives -- point FreeSurfer at the license shipped in the BIDS dataset.
run_in_docker_MRI() {
    local step_name=$1
    local log_file=$2
    local cmd=$3

    CE_GPU=1; CE_HOME=1; CE_EXEC=/bin/bash; CE_PWD=/parrot_home
    CE_ENVS=( "FS_LICENSE=/bids/license.txt" )
    CE_BINDS=( "$BIDS_DIR:/bids:ro" "$OUTPUT_DIR:/derivatives" )
    container_exec "$IMG_MRI_RECONSTRUCTION" -c "source /scripts/source_env.sh && $cmd" > "${log_file}.partial" 2>&1

    check_step $? "$step_name" "$log_file"
}

run_in_docker_FWD() {
    local step_name=$1
    local log_file=$2
    local image=$3
    local cmd=$4

    CE_GPU=1; CE_HOME=1; CE_EXEC=/bin/bash
    CE_BINDS=( "$BIDS_DIR:/bids:ro" "$OUTPUT_DIR:/derivatives" )
    container_exec "$image" -c "$cmd" > "${log_file}.partial" 2>&1

    check_step $? "$step_name" "$log_file"
}

# Run a leadfield solver (OpenMEEG/DUNEuro) in the forward-solvers image. These scripts read
# geometry.geom / conductivities.cond / neuronal_strength_dict.json from the current directory
# (the baked WORKDIR /pipeline) and write intermediates (e.g. OpenMEEG's head.hm) there too.
# That's fine as root, but rootless makes /pipeline read-only and -- under Apptainer -- cwd
# defaults to the host PWD. So stage those three tiny config files into a writable per-call
# scratch (on the derivatives parallel FS, since BEM matrices can be large) and cd there
# first. Runtime-agnostic: the cd happens inside the container command for docker and apptainer alike.
run_in_docker_SOLVER() {
    local step_name=$1
    local log_file=$2
    local image=$3
    local cmd=$4

    local pre="s=\$(mktemp -d $WORK_DIR_DOCKER/solver.XXXXXX) && cd \"\$s\" && \
cp /pipeline/geometry.geom /pipeline/conductivities.cond /pipeline/neuronal_strength_dict.json \"\$s\"/ && "
    run_in_docker_FWD "$step_name" "$log_file" "$image" "${pre}${cmd}"
}

# Run the final QC stage in the parrot_qc image. The QC code is baked into the
# image (the qc/ package, like the other Parrot images' scripts); only
# /derivatives is needed (QC reads everything from there, including raw/T1).
# CE_HOME gives a writable HOME for matplotlib/pyvista caches; pyvista renders
# offscreen via Mesa software GL. QC is informational, so this is NON-FATAL: a QC
# failure must never abort a reconstruction whose heavy stages already succeeded.
run_in_docker_QC() {
    local step_name=$1
    local log_file=$2
    local cmd=$3

    CE_HOME=1; CE_EXEC=/bin/bash
    CE_BINDS=( "$OUTPUT_DIR:/derivatives" )
    if ! container_exec "$IMG_QC" -c "$cmd" > "$log_file" 2>&1; then
        echo "WARNING: $step_name failed (see $log_file)" | tee -a "${LOG_FILE:-/dev/stderr}"
    fi
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

    # Per-subject writable HOME for rootless containers (see container_exec). Lives in the
    # ephemeral work dir (swept on exit) on the derivatives filesystem; bound at /parrot_home.
    # Per-subject so concurrent runs (e.g. a SLURM array) never share config/cache/lock files.
    PARROT_HOME_HOST="$WORK_DIR/home_sub-${SUBJECT}"
    mkdir -p "$PARROT_HOME_HOST"

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
    # DWI drives subject-specific structural connectivity + WM anisotropy. Optional:
    # with no usable DWI the pipeline degrades to the template connectome + isotropic
    # FEM. Three sources, selected by --dwi-preprocessed:
    #   "" (raw) : BIDS dwi/  -> QSIPrep -> QSIRecon (--input-type qsiprep)
    #   qsiprep  : a qsiprep-derivatives tree already at <out>/qsiprep/ (skip QSIPrep)
    #   hcp      : HCP-YA native tree at <bids>/sourcedata/hcp/<ID>/ (QSIRecon hcpya)
    HAS_DWI=false
    BVAL_DOCKER=""        # bval source for adaptive recon-spec selection (per format)
    if [ "$DWI_FORMAT" = "hcp" ]; then
        HCP_SRC="$BIDS_DIR/sourcedata/hcp/${SUBJECT}"
        if [ -f "$HCP_SRC/T1w/Diffusion/data.nii.gz" ] && [ -f "$HCP_SRC/T1w/Diffusion/bvals" ] && [ -f "$HCP_SRC/T1w/Diffusion/bvecs" ]; then
            HAS_DWI=true
            BVAL_DOCKER="/bids/sourcedata/hcp/${SUBJECT}/T1w/Diffusion/bvals"
            echo "Found HCP DWI: $HCP_SRC/T1w/Diffusion/data.nii.gz" | tee -a "$LOG_FILE"
        fi
    elif [ "$DWI_FORMAT" = "qsiprep" ]; then
        QP_DWI=$(find "$OUTPUT_DIR/qsiprep/sub-${SUBJECT}/dwi" -name "*space-ACPC_desc-preproc_dwi.nii.gz" 2>/dev/null | head -n 1)
        if [ -n "$QP_DWI" ]; then
            HAS_DWI=true
            BVAL_DOCKER="/derivatives/qsiprep/sub-${SUBJECT}/dwi/$(basename "${QP_DWI%.nii.gz}").bval"
            echo "Found preprocessed QSIPrep DWI: $QP_DWI" | tee -a "$LOG_FILE"
        fi
    else
        DWI_PATH=$(find "$SUB_BIDS_DIR/dwi" -name "sub-${SUBJECT}*_dwi.nii.gz" 2>/dev/null | head -n 1)
        if [ -n "$DWI_PATH" ]; then
            BVAL_PATH="${DWI_PATH%.nii.gz}.bval"
            BVEC_PATH="${DWI_PATH%.nii.gz}.bvec"
            if [ -f "$BVAL_PATH" ] && [ -f "$BVEC_PATH" ]; then
                HAS_DWI=true
                DWI_DOCKER="/bids/sub-${SUBJECT}/dwi/$(basename "$DWI_PATH")"
                BVAL_DOCKER="${DWI_DOCKER%.nii.gz}.bval"
                echo "Found DWI: $DWI_PATH" | tee -a "$LOG_FILE"
            else
                echo "[WARN] DWI found but .bval/.bvec missing alongside it; treating sub-${SUBJECT} as no-DWI." | tee -a "$LOG_FILE"
            fi
        fi
    fi

    if [ -n "$DWI_FORMAT" ] && [ "$HAS_DWI" = false ]; then
        echo "[WARN] --dwi-preprocessed $DWI_FORMAT set but no usable DWI found for sub-${SUBJECT}; falling back to template connectome." | tee -a "$LOG_FILE"
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

    # ---------------------------------------------------------
    # Recon backend selection (per subject -> mixed cohorts work)
    # ---------------------------------------------------------
    # FastSurfer ALWAYS runs and its CNN subsegs (CerebNet/HypVINN) live in
    # fastsurfer/ -> SEG_DIR. Surfaces come either from FastSurfer (full mode) or
    # from FreeSurfer recon-all / a staged HCP recon -> SURF_DIR. An already-present
    # recon (e.g. HCP-staged freesurfer/, detected by surf/lh.white) is reused
    # regardless of --recon, so cohorts mixing HCP and raw subjects just work.
    SEG_DIR="fastsurfer"
    if [ -f "$OUTPUT_DIR/freesurfer/sub-${SUBJECT}/surf/lh.white" ]; then
        SURF_DIR="freesurfer"
    elif [ -f "$OUTPUT_DIR/fastsurfer/sub-${SUBJECT}/surf/lh.white" ]; then
        SURF_DIR="fastsurfer"
    elif [ "$RECON_BACKEND" = "freesurfer" ]; then
        SURF_DIR="freesurfer"
    else
        SURF_DIR="fastsurfer"
    fi
    echo "Recon backend: surfaces from '$SURF_DIR', CNN subsegs from '$SEG_DIR'." | tee -a "$LOG_FILE"

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
        # Full seg+surf when FastSurfer provides the surfaces (SURF_DIR=fastsurfer);
        # --seg_only (just the CNN subsegs CerebNet/HypVINN) when FreeSurfer/HCP
        # provides the surfaces instead.
        if [ "$SURF_DIR" = "fastsurfer" ]; then
            FS_MODE=("--parallel"); fs_desc="seg + surfaces"
        else
            FS_MODE=("--seg_only"); fs_desc="segmentation only (surfaces from $SURF_DIR)"
        fi
        log_step "Running $NAME reconstruction ($fs_desc)..."
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)
        # Use the standardized, MPRAGEised T1 from ingest (T1_DOCKER), not the raw
        # BIDS T1 -- so MP2RAGE subjects get the same conditioned input recon-all uses.
        CE_GPU=1
        CE_BINDS=( "$BIDS_DIR:/data:ro" "$OUTPUT_DIR:/derivatives" )
        container_exec "$IMG_FASTSURFER" \
            --fs_license /data/license.txt \
            --t1 "$T1_DOCKER" \
            --sid "sub-${SUBJECT}" \
            --sd /derivatives/$NAME \
            --3T --threads "$N_THREADS" "${FS_MODE[@]}" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1
        fastsurfer_rc=$?

        # In full mode FastSurfer can exit 0 even when the surf stage dies (e.g. a
        # rejected vox_size), so $? alone isn't enough -- assert the surfaces exist.
        # (--seg_only legitimately produces none, so only check in full mode.)
        if [ "$SURF_DIR" = "fastsurfer" ] && [ "$fastsurfer_rc" -eq 0 ] && [ ! -f "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/surf/lh.white" ]; then
            echo "[ERROR] FastSurfer exited 0 but produced no surfaces (surf/lh.white missing)." | tee -a "$LOG_FILE"
            echo "        Most likely the input voxel size is > 1mm (header artifact); it must be <= 1mm." | tee -a "$LOG_FILE"
            fastsurfer_rc=1
        fi
        check_step "$fastsurfer_rc" "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

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
        CE_BINDS=( "$BIDS_DIR:/bids:ro" "$HIPPUNFOLD_TMP:/output" )
        container_exec "$IMG_HIPPUNFOLD" \
            /bids /output participant \
            --participant_label "$SUBJECT" \
            --modality T1w --cores "$N_THREADS" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1
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
    # FREESURFER RECON-ALL  (only when SURF_DIR=freesurfer)
    # ---------------------------------------------------------
    # Optional surface backend: full recon-all into freesurfer/sub-X. Skipped for
    # the default FastSurfer backend, and skipped-via-placeholder-log for HCP (whose
    # FreeSurfer recon is staged in directly). Uses the standardized, MPRAGEised
    # T1_DOCKER from ingest. Unlike FastSurfer (T1-only), recon-all refines the pial
    # surface with the T2 when one is available (-T2 ... -T2pial); FLAIR refinement
    # stays dropped.
    NAME="freesurfer"
    if [ "$SURF_DIR" = "freesurfer" ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME recon-all (surface reconstruction)..."
            mkdir -p "$OUTPUT_DIR/$NAME"

            # T2 pial refinement when a (standardized) T2 is present.
            fs_t2_args=""
            if [ -n "$T2_PATH" ]; then
                fs_t2_args="-T2 $T2_DOCKER -T2pial"
            fi

            step_start=$(date +%s)
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" \
                "export SUBJECTS_DIR=/derivatives/freesurfer && \
                 recon-all -subject sub-${SUBJECT} -i $T1_DOCKER $fs_t2_args -all -threads $N_THREADS"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    fi

    # ---------------------------------------------------------
    # RECON LUTS  (backend-agnostic)
    # ---------------------------------------------------------
    # The schaefer/atlas stages read the FreeSurfer color LUT + the Schaefer
    # projection LUT from the surfaces dir. FastSurfer/recon-all don't ship these,
    # so install them into SURF_DIR for every backend (incl. HCP, where recon-all
    # was skipped). Idempotent: only runs if the color LUT isn't already there.
    if [ ! -f "$OUTPUT_DIR/$SURF_DIR/sub-${SUBJECT}/FreeSurferColorLUT.txt" ]; then
        log_step "Installing recon LUTs into $SURF_DIR/sub-${SUBJECT}..."
        CE_HOME=1; CE_EXEC=/bin/bash; CE_PWD=/parrot_home
        CE_BINDS=( "$OUTPUT_DIR:/derivatives" )
        container_exec "$IMG_MRI_RECONSTRUCTION" -c \
            "cp \$FREESURFER_HOME/FreeSurferColorLUT.txt /derivatives/$SURF_DIR/sub-${SUBJECT}/FreeSurferColorLUT.txt && \
             cp -r /home/Schaefer2018_LocalGlobal/Parcellations/project_to_individual /derivatives/$SURF_DIR/sub-${SUBJECT}/Schaefer_LUT" \
            >> "$LOG_FILE" 2>&1 \
            || { echo "[ERROR] failed to install recon LUTs into $SURF_DIR/sub-${SUBJECT}" | tee -a "$LOG_FILE"; exit 1; }
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
            "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && \
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

            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi lh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/lh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/lh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_surf2surf --hemi rh --srcsubject fsaverage --trgsubject $SUBJECT --sval-annot /home/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/rh.${ATLAS_NAME}.annot --tval \$SUBJECTS_DIR/$SUBJECT/label/rh.${ATLAS_NAME}.annot"
            run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && mri_aparc2aseg --s $SUBJECT --o \$SUBJECTS_DIR/$SUBJECT/mri/schaefer${n_parcels}_aparc+aseg.mgz --annot $ATLAS_NAME"
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

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions thalamus --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions hippo-amygdala --cross $SUBJECT --threads $N_THREADS"
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "ln -sf /derivatives/$SURF_DIR/sub-${SUBJECT} \$FREESURFER_HOME/subjects/$SUBJECT && segment_subregions brainstem --cross $SUBJECT --threads $N_THREADS"

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
        # Re-run from clean: wipe any partial output so the final `mv m2m_subject ->
        # sub-<ID>` can't nest inside a leftover dir from a killed attempt.
        begin_step "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"
        mkdir -p "$OUTPUT_DIR/$NAME"

        step_start=$(date +%s)

        # charm writes its m2m_subject folder into the current directory, so cwd must be a
        # WRITABLE, bind-mounted path -- the old /home/simnibs_reconstructions was a baked
        # image dir (fine as root, but unwritable when running rootless / under Apptainer's
        # read-only image). Use the per-subject scratch under /derivatives instead. charm /
        # simnibs_python are now on PATH (source_env.sh) since SimNIBS moved to /opt, so no
        # more hardcoded /root/SimNIBS-4.5/bin/... (unreadable to a non-root UID).
        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "cd $WORK_DIR_DOCKER && \
                                                        charm subject $T1_DOCKER ${simnibs_args[*]} --forcerun --fs-dir /derivatives/$SURF_DIR/sub-${SUBJECT} --forcesform && \
                                                        cd / && \
                                                        simnibs_python /scripts/extract_charm_surf.py --charm_dir "$WORK_DIR_DOCKER/m2m_subject/" && \
                                                        cp /scripts/simnibs_conductivities.txt $WORK_DIR_DOCKER/m2m_subject/conductivities.txt && \
                                                        cp /scripts/simnibs_labels.txt $WORK_DIR_DOCKER/m2m_subject/labels.txt && \
                                                        mv $WORK_DIR_DOCKER/m2m_subject /derivatives/$NAME/sub-${SUBJECT}"

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
                                                        micromamba run -n neuro python /scripts/run_cereb_pipeline.py --output_dir /derivatives --subject $SUBJECT --template_dir /home/cerebellum_template/ --threads $N_THREADS --seg_dir $SEG_DIR"

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

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/gather_surfaces.py --output_dir /derivatives --subject $SUBJECT --surf_dir $SURF_DIR"
 
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

        run_in_docker_MRI "$NAME" "$LOG_DIR/${NAME}_log.txt" "micromamba run -n neuro python /scripts/make_atlas.py --T1_path $T1_DOCKER --output_dir /derivatives --subject $SUBJECT --surf_dir $SURF_DIR --seg_dir $SEG_DIR"
 
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
    if [ "$HAS_DWI" = true ] && [ -z "$DWI_FORMAT" ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (DWI preprocessing)..."
            mkdir -p "$OUTPUT_DIR/$NAME"

            step_start=$(date +%s)

            # Output resolution: --output-resolution is mandatory and forces an
            # ISOTROPIC grid. Policy = min(native smallest axis, 1.25): upsample
            # coarse DWI toward the MRtrix-recommended ~1.25 mm for tractography,
            # but never downsample finer-than-1.25 acquisitions. Read the native
            # voxel size with nibabel from our MRI image (it has the env).
            # Set CE_* INSIDE the command substitution: a subshell, so they don't leak into
            # the next (non-substituted) container_exec call -- container_exec's reset runs in
            # the subshell and wouldn't propagate to the parent.
            OUTPUT_RES=$(
                CE_HOME=1; CE_EXEC=micromamba; CE_BINDS=( "$BIDS_DIR:/bids:ro" )
                container_exec "$IMG_MRI_RECONSTRUCTION" \
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
            # bind specs ("src:dst:ro"), runtime-agnostic (container_exec turns these into
            # -v or --bind as needed).
            QSIPREP_BIDS_MOUNTS=(
                "$BIDS_DIR/dataset_description.json:/bids/dataset_description.json:ro"
                "$BIDS_DIR/license.txt:/bids/license.txt:ro"
                "$BIDS_DIR/sub-${SUBJECT}:/bids/sub-${SUBJECT}:ro"
            )
            # participants.tsv is optional; bind it only if present -- a missing
            # bind source makes Docker silently create an empty dir at /bids.
            if [ -f "$BIDS_DIR/participants.tsv" ]; then
                QSIPREP_BIDS_MOUNTS+=( "$BIDS_DIR/participants.tsv:/bids/participants.tsv:ro" )
            fi

            # Runs rootless (container_exec --user / apptainer); the image's HOME
            # (/home/qsiprep) is world-writable so on docker we keep it as-is (no CE_HOME).
            # TemplateFlow is cached persistently across runs.
            CE_GPU=1
            CE_ENVS=( "TEMPLATEFLOW_HOME=/templateflow" )
            CE_BINDS=( "$TEMPLATEFLOW_DIR:/templateflow" "${QSIPREP_BIDS_MOUNTS[@]}" "$OUTPUT_DIR:/derivatives" )
            container_exec "$IMG_QSIPREP" \
                /bids "/derivatives/$NAME" participant \
                --participant-label "$SUBJECT" \
                --fs-license-file /bids/license.txt \
                --output-resolution "$OUTPUT_RES" \
                --nprocs "$N_THREADS" \
                --omp-nthreads "$N_THREADS" \
                --skip-bids-validation \
                -w "$WORK_DIR_DOCKER" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1

            check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    elif [ "$HAS_DWI" = true ] && [ -n "$DWI_FORMAT" ]; then
        echo "$NAME skipped for sub-${SUBJECT} (--dwi-preprocessed $DWI_FORMAT): DWI already preprocessed." | tee -a "$LOG_FILE"
    else
        echo "$NAME skipped for sub-${SUBJECT} (no usable DWI)." | tee -a "$LOG_FILE"
    fi

    # ---------------------------------------------------------
    # QSIRECON (TRACTOGRAPHY)
    # ---------------------------------------------------------
    # Optional. Selects an MRtrix recon spec adaptively from the bval shell scheme
    # and exports the tractogram + SIFT2 weights for tck2connectome (run in the
    # connectivity stage). A too-sparse scheme -> skip (template fallback). Input
    # source depends on --dwi-preprocessed: qsiprep-derivatives (raw/qsiprep) via
    # --input-type qsiprep, or HCP-YA native tree via --input-type hcpya.
    NAME="qsirecon"
    if [ "$HAS_DWI" = true ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (tractography)..."
            step_start=$(date +%s)

            # Per-source recon input: dir, --input-type, an optional extra mount, and
            # whether the input is actually present.
            RECON_INPUT_MOUNT=()
            recon_ready=true
            if [ "$DWI_FORMAT" = "hcp" ]; then
                # HCP-YA native tree; ingress2qsirecon converts it internally. Bind only
                # this subject so QSIRecon doesn't index the whole HCP cohort.
                RECON_INPUT_MOUNT=("$BIDS_DIR/sourcedata/hcp/${SUBJECT}:/hcp_in/${SUBJECT}:ro")
                RECON_INPUT_DIR="/hcp_in"
                RECON_INPUT_TYPE="hcpya"
            else
                # raw or qsiprep: consume the qsiprep-derivatives tree.
                RECON_INPUT_DIR="/derivatives/qsiprep"
                RECON_INPUT_TYPE="qsiprep"
                [ -d "$OUTPUT_DIR/qsiprep/sub-${SUBJECT}" ] || recon_ready=false
            fi

            if [ "$recon_ready" != true ]; then
                echo "[WARN] $NAME: no recon input for sub-${SUBJECT}; skipping (template fallback)." | tee -a "$LOG_FILE"
            else
                # Adaptive recon-spec selection from the acquisition shell scheme:
                # >=2 non-zero shells -> MSMT; 1 shell with >=28 dirs -> SS3T; else skip.
                # BVAL_DOCKER (set during detection) may live under /bids or /derivatives.
                # CE_* set inside the substitution (subshell) so they don't leak -- see the
                # qsiprep output-resolution probe above for the rationale.
                RECON_CHOICE=$(
                    CE_HOME=1; CE_EXEC=micromamba; CE_BINDS=( "$BIDS_DIR:/bids:ro" "$OUTPUT_DIR:/derivatives" )
                    container_exec "$IMG_MRI_RECONSTRUCTION" \
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
                    echo "Selected recon spec: $SPEC (shell choice '$RECON_CHOICE', input-type $RECON_INPUT_TYPE)." | tee -a "$LOG_FILE"
                    mkdir -p "$WORK_DIR/qsirecon_out"

                    # QSIRecon reuses our FreeSurfer (ACT-hsvs); persistent TemplateFlow
                    # cache; --user avoids root-owned outputs. Output lands under the
                    # ephemeral work dir, then we relocate the results into place. We bind
                    # only license.txt from $BIDS_DIR (plus the HCP subject tree for hcpya)
                    # to avoid needlessly exposing the whole dataset/derivatives tree.
                    CE_GPU=1
                    CE_ENVS=( "TEMPLATEFLOW_HOME=/templateflow" )
                    CE_BINDS=(
                        "$TEMPLATEFLOW_DIR:/templateflow"
                        "$PARROT_SCRIPT_DIR/template_data/qsirecon_specs:/specs:ro"
                        "$BIDS_DIR/license.txt:/bids/license.txt:ro"
                        "$OUTPUT_DIR:/derivatives"
                        "${RECON_INPUT_MOUNT[@]}"
                    )
                    container_exec "$IMG_QSIRECON" \
                        "$RECON_INPUT_DIR" "$WORK_DIR_DOCKER/qsirecon_out" participant \
                        --participant-label "$SUBJECT" \
                        --recon-spec "/specs/$SPEC" \
                        --input-type "$RECON_INPUT_TYPE" \
                        --fs-subjects-dir /derivatives/$SURF_DIR \
                        --fs-license-file /bids/license.txt \
                        --nprocs "$N_THREADS" \
                        --omp-nthreads "$N_THREADS" \
                        -w "$WORK_DIR_DOCKER" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1
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

    # Tractography present -> the subject connectome is built later, AFTER dwi2t1
    # registers the tracts into the atlas's T1/mesh space (see the connectome
    # matrices block below the DWI-tensor stages). No usable tractography here ->
    # fall back to the group-average template connectome.
    if [ "$HAVE_TRACKS" = false ]; then
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
    # DWI TENSOR  (anisotropic-conductivity front end, WIP)
    # ---------------------------------------------------------
    # Fit a diffusion tensor from the QSIPrep-preprocessed DWI -- step [A] of the
    # WM-anisotropy feature. Runs in the QSIRecon image (MRtrix), same pattern as
    # the connectome step. Output stays in QSIPrep's ACPC space; resampling +
    # eigenvector reorientation onto the FEM mesh is a separate, not-yet-built
    # step. Needs only the qsiprep DWI (not the tractogram). The tensor will be
    # consumed by the anisotropic FEM leadfield, so a failure is fatal like every
    # other step (check_step exits + cleans the partial output dir).
    # HCP fits directly from the staged HCP DWI (already in T1 space -> labelled
    # space-T1, no dwi2t1 needed); raw/qsiprep fit the QSIPrep ACPC DWI (-> space-ACPC,
    # carried to T1 by dwi2t1 below).
    NAME="dwitensor"
    dwitensor_ready=false
    DWITENSOR_FMT=""
    DWITENSOR_MOUNT=()
    if [ "$HAS_DWI" = true ]; then
        if [ "$DWI_FORMAT" = "hcp" ]; then
            dwitensor_ready=true
            DWITENSOR_FMT="hcp"
            DWITENSOR_MOUNT=("$BIDS_DIR/sourcedata/hcp/${SUBJECT}:/hcp_in/${SUBJECT}:ro")
        elif compgen -G "$OUTPUT_DIR/qsiprep/sub-${SUBJECT}/dwi/"*space-ACPC_desc-preproc_dwi.nii.gz > /dev/null 2>&1; then
            dwitensor_ready=true
        fi
    fi
    if [ "$dwitensor_ready" = true ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (DTI fit for anisotropic conductivity)..."
            step_start=$(date +%s)

            CE_EXEC=bash
            CE_BINDS=(
                "$OUTPUT_DIR:/derivatives"
                "$PARROT_SCRIPT_DIR/bin/make_dwitensor.sh:/make_dwitensor.sh:ro"
                "${DWITENSOR_MOUNT[@]}"
            )
            container_exec "$IMG_QSIRECON" \
                /make_dwitensor.sh "$SUBJECT" "$DWITENSOR_FMT" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1
            check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    fi

    # ---------------------------------------------------------
    # DWI -> T1 SPACE  (registration + tensor refit + tracts), WIP
    # ---------------------------------------------------------
    # Step [B]: register the QSIPrep DWI (ACPC space) to Parrot T1/mesh space and
    # carry the derivatives over -- MRtrix-native in the QSIRecon image. Tensor is
    # reoriented the institutional way (mrtransform rotates the DW gradients, then
    # re-fit in T1), and the tractogram is transformed so it shares the atlas's
    # space (which lets the connectome step below run correctly). Fatal on failure
    # like every other step.
    # Skipped for HCP: its DWI is already in T1 space (dwitensor wrote space-T1
    # directly, and the qsirecon-hcpya tractogram is already T1w/mesh space), so
    # there is no ACPC->anat transform to apply.
    NAME="dwi2t1"
    if [ "$HAS_DWI" = true ] && [ "$DWI_FORMAT" != "hcp" ] && \
       compgen -G "$OUTPUT_DIR/qsiprep/sub-${SUBJECT}/dwi/"*space-ACPC_desc-preproc_dwi.nii.gz > /dev/null 2>&1; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (register DWI derivatives to T1 space)..."
            step_start=$(date +%s)

            # Heavy scratch (resampled 4D DWI, decompressed tractogram, warp) goes
            # to the swept WORK_DIR, not the derivatives tree. Products land under
            # dwitensor/ (tensor + transform) and qsirecon/ (T1 tractogram); this
            # stage has no output folder of its own.
            CE_EXEC=bash
            CE_BINDS=(
                "$OUTPUT_DIR:/derivatives"
                "$PARROT_SCRIPT_DIR/bin/dwi_to_t1.sh:/dwi_to_t1.sh:ro"
            )
            container_exec "$IMG_QSIRECON" \
                /dwi_to_t1.sh "$SUBJECT" "$WORK_DIR_DOCKER" > "$LOG_DIR/${NAME}_log.txt.partial" 2>&1
            check_step $? "$NAME" "$LOG_DIR/${NAME}_log.txt"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    fi

    # ---------------------------------------------------------
    # CONNECTOME MATRICES  (tck2connectome, in T1 space)
    # ---------------------------------------------------------
    # Runs after dwi2t1 so the tractogram is in the atlas's T1/mesh space. Same
    # MRtrix3 that generated the tracks (QSIRecon image). Builds the subject
    # connectome from the T1 tractogram + native T1 atlas.
    NAME="connectivity"
    if [ "$HAVE_TRACKS" = true ]; then
        if [ ! -f "$LOG_DIR/${NAME}-matrices_log.txt" ]; then
            log_step "Running $NAME matrices (tck2connectome)..."
            step_start=$(date +%s)

            CE_EXEC=bash
            CE_BINDS=(
                "$OUTPUT_DIR:/derivatives"
                "$PARROT_SCRIPT_DIR/bin/make_connectomes.sh:/make_connectomes.sh:ro"
            )
            container_exec "$IMG_QSIRECON" \
                /make_connectomes.sh "$SUBJECT" > "$LOG_DIR/${NAME}-matrices_log.txt.partial" 2>&1
            check_step $? "$NAME matrices" "$LOG_DIR/${NAME}-matrices_log.txt"

            step_end=$(date +%s)
            echo "$NAME matrices completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME matrices log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
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

        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python /scripts/place_electrodes.py --subject $SUBJECT --output_dir /derivatives"
 
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
            # Re-run from clean: wipe any partial dipoles for THIS spacing (place_dipoles
            # writes per-surface dirs incrementally and won't redo completed surfaces).
            begin_step "$LOG_DIR/${NAME}-${spacing}mm_log.txt" "$OUTPUT_DIR/$NAME/sub-${SUBJECT}/spacing${spacing}mm"
            echo "Placing dipoles at $spacing mm spacing..."

            step_start=$(date +%s)
            run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}-${spacing}mm_log.txt" "$DOCKER_IMAGE" "python /scripts/place_dipoles.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing${DIPOLE_SEED:+ --seed $DIPOLE_SEED}"
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

        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python /scripts/nifti_to_inr.py --nifti_path /derivatives/tissuelabels/sub-${SUBJECT}/electrical/$VOLUME_TO_MESH.nii.gz --inr_path /derivatives/tetmesh/sub-${SUBJECT}/label_field.inr"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "mesher $N_THREADS /derivatives/tetmesh/sub-${SUBJECT}/label_field.inr /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh $ANGLE $DIST $DEF_SURF $DEF_VOL $RATIO $SMOOTH $OPT_TIME ${TISSUE_ARGS[*]}"
        run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python /scripts/mesh_postprocessing.py --reference_nifti /derivatives/tissuelabels/sub-${SUBJECT}/electrical/$VOLUME_TO_MESH.nii.gz --mesh /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --output /derivatives/tetmesh/sub-${SUBJECT}/transformed_tetrahedral_mesh.mesh --export_vtu"
 
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
    # ANISOTROPY  (WM conductivity tensors for the CGAL FEM)
    # ---------------------------------------------------------
    # Optional. Turns the subject diffusion tensor (cerebral, from the dwitensor
    # stage) -- plus the warped cerebellar template DTI when present -- into one
    # anisotropic conductivity tensor per White-Matter tetrahedron of the CGAL
    # mesh. Gated on a subject DTI in mesh/T1 space; with no DWI it is skipped and
    # the CGAL leadfield below stays isotropic (unchanged). Runs in the solvers
    # image (needs the mesh, hence after tetmesh).
    NAME="anisotropy"
    CEREBRAL_DTI="$OUTPUT_DIR/dwitensor/sub-${SUBJECT}/sub-${SUBJECT}_space-T1_model-dti_tensor.nii.gz"
    if [ -f "$CEREBRAL_DTI" ]; then
        if [ ! -f "$LOG_DIR/${NAME}_log.txt" ]; then
            log_step "Running $NAME (DTI -> WM conductivity tensors)..."
            step_start=$(date +%s)

            # Cerebellar WM uses the warped cerebellar template DTI when available,
            # routed by the FastSurfer cerebellum segmentation; otherwise every WM
            # tet is sampled from the subject (cerebral) DTI.
            CEREB_ARGS=""
            if [ -f "$OUTPUT_DIR/cerebellum/sub-${SUBJECT}/nonlinear_DTI.nii.gz" ] && \
               [ -f "$OUTPUT_DIR/$SEG_DIR/sub-${SUBJECT}/mri/cerebellum.CerebNet.nii.gz" ]; then
                CEREB_ARGS="--cerebellar_dti /derivatives/cerebellum/sub-${SUBJECT}/nonlinear_DTI.nii.gz --cerebellum_mask /derivatives/$SEG_DIR/sub-${SUBJECT}/mri/cerebellum.CerebNet.nii.gz"
            fi

            run_in_docker_FWD "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python3 /scripts/dti_to_conductivity_tensors.py --subject $SUBJECT --output_dir /derivatives --mesh_path /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --tissue_names /derivatives/tetmesh/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/tetmesh/sub-${SUBJECT}/conductivities.txt --cerebral_dti /derivatives/dwitensor/sub-${SUBJECT}/sub-${SUBJECT}_space-T1_model-dti_tensor.nii.gz $CEREB_ARGS"

            step_end=$(date +%s)
            echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
        else
            echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
        fi
    else
        echo "$NAME skipped for sub-${SUBJECT} (no subject DTI; CGAL leadfield stays isotropic)." | tee -a "$LOG_FILE"
    fi

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
        run_in_docker_SOLVER "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python3 /scripts/make_leadfield_openmeeg.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing"

        spacing=$(printf "%.1f" "$SPACING_DUNEURO_SIMNIBS")
        echo "Solving forward problem with DUNEuro using SimNIBS charm mesh, at $spacing mm dipole spacing"
        run_in_docker_SOLVER "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python3 /scripts/make_leadfield_duneuro.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing --threads $N_THREADS --mesh_path /derivatives/simnibscharm/sub-${SUBJECT}/subject.msh --tissue_names /derivatives/simnibscharm/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/simnibscharm/sub-${SUBJECT}/conductivities.txt --label simnibs --valid_tissues \"Gray-Matter\""

        spacing=$(printf "%.1f" "$SPACING_DUNEURO_CGAL")
        echo "Solving forward problem with DUNEuro using CGAL mesh, at $spacing mm dipole spacing"
        run_in_docker_SOLVER "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python3 /scripts/make_leadfield_duneuro.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing --threads $N_THREADS --mesh_path /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --tissue_names /derivatives/tetmesh/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/tetmesh/sub-${SUBJECT}/conductivities.txt --label CGAL --valid_tissues $CGAL_VALID_TISSUES"

        # Anisotropic CGAL leadfield (extra, alongside the isotropic one above) --
        # only when the anisotropy stage produced WM conductivity tensors. White
        # Matter must stay OUT of CGAL_VALID_TISSUES here, or the Venant source
        # model's monopole patch collapses for WM dipoles.
        if [ -f "$OUTPUT_DIR/anisotropy/sub-${SUBJECT}/conductivity_tensors.npy" ]; then
            echo "Solving forward problem with DUNEuro using CGAL mesh (ANISOTROPIC WM), at $spacing mm dipole spacing"
            run_in_docker_SOLVER "$NAME" "$LOG_DIR/${NAME}_log.txt" "$DOCKER_IMAGE" "python3 /scripts/make_leadfield_duneuro.py --subject $SUBJECT --output_dir /derivatives --dipole_spacing $spacing --threads $N_THREADS --mesh_path /derivatives/tetmesh/sub-${SUBJECT}/tetrahedral_mesh.mesh --tissue_names /derivatives/tetmesh/sub-${SUBJECT}/labels.txt --conductivities_path /derivatives/tetmesh/sub-${SUBJECT}/conductivities.txt --label CGAL_anisotropic --valid_tissues $CGAL_VALID_TISSUES --dti_tensors_path /derivatives/anisotropy/sub-${SUBJECT}/conductivity_tensors.npy"
        fi

        step_end=$(date +%s)
        echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"
    else
        echo "$NAME log file detected for subject $SUBJECT. Skipping step..." | tee -a "$LOG_FILE"
    fi    


    # ---------------------------------------------------------------------
    # FINAL QC: validate every stage's outputs + render an HTML report.
    # ALWAYS runs (no idempotency guard) so the report reflects the latest
    # outputs, and is NON-FATAL -- it is informational and must never block the
    # pipeline. It is quick (~2-3 min/subject) relative to the heavy stages.
    # ---------------------------------------------------------------------
    NAME="qc"
    log_step "Running final $NAME for subject $SUBJECT..."
    step_start=$(date +%s)
    run_in_docker_QC "$NAME" "$LOG_DIR/${NAME}_log.txt" \
        "python /qc/run_qc.py --subject $SUBJECT --output_dir /derivatives --threads $N_THREADS"
    step_end=$(date +%s)
    echo "$NAME completed in $(( (step_end - step_start) / 60 )) minutes." | tee -a "$LOG_FILE"

    # All stages done for this subject. Outputs are already user-owned (rootless), so there
    # is nothing to re-own; just clear CURRENT_SUBJECT for symmetry with the cleanup trap.
    CURRENT_SUBJECT=""
done

# Group-level QC: aggregate every subject's report into qc/index.html. Always
# refreshed (no idempotency guard) and non-fatal: a failure here must not fail a
# run whose per-subject reports already succeeded.
echo "Writing group QC index..."
mkdir -p "$OUTPUT_DIR/logs"
run_in_docker_QC "qc-group" "$OUTPUT_DIR/logs/qc-group_log.txt" \
    "python /qc/run_qc.py --group --output_dir /derivatives"

echo ""
echo "====================================================================="
echo "ALL SUBJECTS PROCESSED SUCCESSFULLY!"
echo "====================================================================="