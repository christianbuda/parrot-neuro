#!/bin/bash
# source freesurfer and FSL

export FREESURFER_HOME=/usr/local/freesurfer/8.1.0
source /usr/local/freesurfer/8.1.0/SetUpFreeSurfer.sh > /dev/null

export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH

# SimNIBS lives under /opt (so a non-root UID can read it). Its PATH entry would otherwise
# only exist in a HOME .bashrc the pipeline never sources, so add it here explicitly. This
# lets the orchestrator call `charm`/`simnibs_python` by name instead of hardcoding
# /root/SimNIBS-4.5/bin/... (which is unreadable when running rootless).
export PATH=/opt/SimNIBS-4.5/bin:$PATH
