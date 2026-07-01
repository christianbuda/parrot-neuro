#!/bin/bash
###############################################################################
# Pre-populate a HippUnfold model/template cache LOCALLY, to rsync up to the
# cluster's persistent HIPPUNFOLD_CACHE_DIR (see the hippunfold step in
# bin/run_reconstruction.sh, default <output_dir>/.hippunfold_cache).
#
# WHY: HippUnfold downloads its atlases/templates from OSF (files.ca-1.osf.io)
# at runtime. LEONARDO's COMPUTE NODES cannot reach OSF ("Network is unreachable"
# for 35.241.38.243) -- Zenodo works, OSF does not. So the download can never
# succeed on the node; we fetch it here (OSF reachable) and ship the cache.
#
# Version-matched: the OSF/Zenodo URLs are read straight from the HippUnfold
# image config (resource_urls), so this stays correct if the image is updated --
# just make sure $HIPPUNFOLD_IMAGE is the SAME tag as the .sif on the cluster.
#
# Run on the WORKSTATION (needs docker + internet). Defaults match a T1w run
# (atlas multihist7; templates upenn + CITI168; nnU-Net model T1w).
#
#   bash hpc/leonardo/prewarm_hippunfold.sh                 # -> ./hippunfold_cache
#   bash hpc/leonardo/prewarm_hippunfold.sh /data/hu_cache  # custom dir
#   MODELS="" bash hpc/leonardo/prewarm_hippunfold.sh       # skip the 2.3G model (Zenodo works on-node)
#   DEST=user@login.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/bids/derivatives/.hippunfold_cache/ \
#       bash hpc/leonardo/prewarm_hippunfold.sh             # build AND rsync up
###############################################################################
set -uo pipefail

IMG="${HIPPUNFOLD_IMAGE:-khanlab/hippunfold:latest}"
CACHE="${1:-$PWD/hippunfold_cache}"
ATLASES="${ATLASES:-multihist7}"        # space-separated
TEMPLATES="${TEMPLATES:-upenn CITI168}" # upenn = nnU-Net T1w train space; CITI168 = default output template
MODELS="${MODELS:-T1w}"                 # nnU-Net model(s); MODELS="" to skip (Zenodo is reachable on-node)
DEST="${DEST:-}"                        # optional rsync target; empty = just print the command

command -v docker >/dev/null || { echo "ERROR: docker not found (this runs on your workstation)."; exit 1; }
command -v wget  >/dev/null || { echo "ERROR: wget not found."; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: unzip not found."; exit 1; }

# Dump the version-matched URL map from the image: lines of "<kind>\t<name>\t<url>".
echo "== reading resource_urls from $IMG =="
MAP="$(docker run --rm --entrypoint bash "$IMG" -c '
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
[ -n "$MAP" ] || { echo "ERROR: could not read resource_urls from $IMG."; exit 1; }

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
if [ "${#missing[@]}" -gt 0 ]; then echo "MISSING (${#missing[@]}): ${missing[*]}"; fi

# --- ship it to the cluster --------------------------------------------------
# The destination is the cluster's HIPPUNFOLD_CACHE_DIR = <output_dir>/.hippunfold_cache
# (or wherever you set HIPPUNFOLD_CACHE_HOST). Trailing slash on both sides = merge contents.
if [ -n "$DEST" ]; then
  echo "== rsync -> $DEST =="
  rsync -avP "$CACHE"/ "$DEST"
else
  echo "To ship it up (dest = the cluster's <output_dir>/.hippunfold_cache):"
  echo "  rsync -avP $CACHE/ <USER>@login.leonardo.cineca.it:/leonardo_work/<ACCT>/parrot/bids/derivatives/.hippunfold_cache/"
fi

[ "${#missing[@]}" -eq 0 ] || exit 1
