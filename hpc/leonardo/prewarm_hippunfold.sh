#!/bin/bash
###############################################################################
# Populate a HippUnfold model/template cache IN PLACE at the path the orchestrator
# uses (HIPPUNFOLD_CACHE_DIR, default <output_dir>/.hippunfold_cache; see the
# hippunfold step in bin/run_reconstruction.sh). No rsync: run it wherever the
# cache should live (the LOGIN node writes straight onto the work filesystem).
#
# WHY: HippUnfold downloads its atlases/templates from OSF (files.ca-1.osf.io)
# at runtime. LEONARDO's COMPUTE NODES cannot reach OSF ("Network is unreachable"
# for 35.241.38.243); LOGIN nodes do have egress -- warm the cache there.
#
# Version-matched: the OSF/Zenodo URLs are read straight from the HippUnfold image
# config (resource_urls), so this stays correct if the image is updated -- just keep
# $HIPPUNFOLD_IMAGE / the .sif tag the SAME as what runs on the cluster.
#
# RUNTIME=apptainer uses the .sif in SIF_DIR (login node, no docker); RUNTIME=docker
# uses the Hub/local image (workstation). Defaults match a T1w run (atlas multihist7;
# templates upenn + CITI168; nnU-Net model T1w). Needs wget + unzip on PATH either way.
#
#   # on a LOGIN node (has egress + singularity + the .sif cache):
#   RUNTIME=apptainer SIF_DIR=$WORKDIR/parrot/parrot_sif \
#     bash hpc/leonardo/prewarm_hippunfold.sh $WORKDIR/parrot/bids/derivatives/.hippunfold_cache
#   # on the workstation:
#   bash hpc/leonardo/prewarm_hippunfold.sh ./hippunfold_cache
###############################################################################
set -uo pipefail

# Optional personal config (gitignored) -- picks up SIF_DIR/OUTPUT_DIR if you set them there.
for _c in "${PARROT_CONFIG:-}" "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)/config.local.sh" \
          "${SLURM_SUBMIT_DIR:-}/hpc/leonardo/config.local.sh" "$HOME/parrot-neuro/hpc/leonardo/config.local.sh"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { . "$_c"; break; }
done

CACHE="${1:-${OUTPUT_DIR:+$OUTPUT_DIR/.hippunfold_cache}}"; CACHE="${CACHE:-$PWD/hippunfold_cache}"  # populate <output_dir>/.hippunfold_cache by default when configured
RUNTIME="${RUNTIME:-docker}"            # docker (workstation) | apptainer (login node, uses SIF_DIR)
SIF_DIR="${SIF_DIR:-${SIF:-}}"          # .sif cache dir (apptainer only); falls back to $SIF from config
IMG="${HIPPUNFOLD_IMAGE:-khanlab/hippunfold:latest}"   # docker image ref (docker runtime)
ATLASES="${ATLASES:-multihist7}"        # space-separated
TEMPLATES="${TEMPLATES:-upenn CITI168}" # upenn = nnU-Net T1w train space; CITI168 = default output template
MODELS="${MODELS:-T1w}"                 # nnU-Net model(s); MODELS="" to skip (Zenodo is reachable on-node)

command -v wget  >/dev/null || { echo "ERROR: wget not found."; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: unzip not found."; exit 1; }

# Run a bash -c command inside the HippUnfold container under the selected runtime.
run_in_hippunfold() {   # $1 = bash -c command string; stdout is the caller's
  case "$RUNTIME" in
    docker)
      command -v docker >/dev/null || { echo "ERROR: RUNTIME=docker but docker not found." >&2; return 1; }
      docker run --rm --entrypoint bash "$IMG" -c "$1" ;;
    apptainer)
      local app sif
      app="$(command -v apptainer || command -v singularity || true)"
      [ -n "$app" ] || { echo "ERROR: RUNTIME=apptainer but no apptainer/singularity on PATH." >&2; return 1; }
      sif="$SIF_DIR/hippunfold_latest.sif"
      [ -f "$sif" ] || { echo "ERROR: $sif not found (set SIF_DIR to your .sif cache)." >&2; return 1; }
      "$app" exec "$sif" bash -c "$1" ;;
    *) echo "ERROR: RUNTIME must be 'docker' or 'apptainer' (got '$RUNTIME')." >&2; return 1 ;;
  esac
}

# Dump the version-matched URL map from the image: lines of "<kind>\t<name>\t<url>".
echo "== reading resource_urls from the HippUnfold image ($RUNTIME) =="
MAP="$(run_in_hippunfold '
D=$(python -c "import hippunfold,os;print(os.path.dirname(hippunfold.__file__))")
python - "$D" <<PY
import sys, yaml, os
cfg=yaml.safe_load(open(os.path.join(sys.argv[1],"config","snakebids.yml")))
ru=cfg.get("resource_urls",{})
# emit under stable labels; models live under the "nnunet_model" config key
for label, key in (("atlas","atlas"),("template","template"),("model","nnunet_model")):
    for k,v in (ru.get(key,{}) or {}).items():
        u=v if str(v).startswith("http") else "https://"+str(v)
        print(f"{label}\t{k}\t{u}")
PY')"
[ -n "$MAP" ] || { echo "ERROR: could not read resource_urls from the image."; exit 1; }

url_for(){ awk -F'\t' -v k="$1" -v n="$2" '$1==k && $2==n {print $3}' <<<"$MAP"; }

mkdir -p "$CACHE"
missing=()

# atlas/<name> and template/<name>: wget the zip, unzip into the cache dir (mirrors
# the download_extract_{atlas,template} rules: `wget URL -O temp.zip && unzip -d DIR`).
fetch_zip() {   # $1=kind (atlas|template)  $2=name
  local kind="$1" name="$2" d="$CACHE/$1/$2" u tmp
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then echo "  have    $kind/$name"; return; fi
  u="$(url_for "$kind" "$name")"
  [ -n "$u" ] || { echo "  NO URL  $kind/$name (not in resource_urls)"; missing+=( "$kind/$name" ); return; }
  echo "  fetch   $kind/$name  <- $u"
  tmp="$(mktemp)"
  if wget -q --show-progress -O "$tmp" "$u" && mkdir -p "$d" && unzip -oq "$tmp" -d "$d"; then :; else
    echo "  FAILED  $kind/$name"; missing+=( "$kind/$name" ); rm -rf "$d"
  fi
  rm -f "$tmp"
}

for a in $ATLASES;   do fetch_zip atlas    "$a"; done
for t in $TEMPLATES; do fetch_zip template "$t"; done

# model/<file>.tar: the download rule just fetches the tar (nnunet.smk extracts it
# at runtime, locally, no network). Save under model/ with the URL's basename.
for m in $MODELS; do
  u="$(url_for model "$m")"
  [ -n "$u" ] || { echo "  NO URL  model/$m"; missing+=( "model/$m" ); continue; }
  base="${u##*/}"; out="$CACHE/model/$base"
  if [ -f "$out" ]; then echo "  have    model/$base"; continue; fi
  echo "  fetch   model/$base  <- $u"
  mkdir -p "$CACHE/model"
  if wget -q --show-progress -O "$out.part" "$u" && mv -f "$out.part" "$out"; then :; else
    echo "  FAILED  model/$base"; missing+=( "model/$m" ); rm -f "$out.part"
  fi
done

echo
echo "cache at: $CACHE"
du -sh "$CACHE" 2>/dev/null | awk '{print "  size: "$1}'
if [ "${#missing[@]}" -gt 0 ]; then echo "MISSING (${#missing[@]}): ${missing[*]}"; exit 1; fi
echo "OK: atlas/template(/model) present."
