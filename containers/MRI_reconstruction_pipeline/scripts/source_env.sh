#!/bin/bash
# source freesurfer and FSL

export FREESURFER_HOME=/usr/local/freesurfer/8.1.0
source /usr/local/freesurfer/8.1.0/SetUpFreeSurfer.sh > /dev/null

export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
