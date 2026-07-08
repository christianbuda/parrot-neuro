# Start from Ubuntu
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

SHELL ["/bin/bash", "--login", "-c"]

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# install some dependencies explicitly (Good for caching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tcsh \
    bc \
    tar \
    libgomp1 \
    perl \
    file \
    libgl1 \
    libxmu6 \
    libxt6 \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install freesurfer
RUN wget -qO /tmp/fs.deb https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.1.0/freesurfer_ubuntu22-8.1.0_amd64.deb \
    && apt-get update \
    && apt-get install -y /tmp/fs.deb \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage_sym \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/bert \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/V1_average \
    && rm -rf /usr/local/freesurfer/8.1.0/trctrain \
    && rm -rf /usr/local/freesurfer/8.1.0/diffusion \
    && rm -rf /usr/local/freesurfer/8.1.0/average/mult-comp-cor \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/cvs_avg35 \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/cvs_avg35_inMNI152 \
    && rm -rf /usr/local/freesurfer/8.1.0/mni-1.4 \
    && rm -rf /usr/local/freesurfer/8.1.0/docs \
    && rm -rf /usr/local/freesurfer/8.1.0/lib/qt \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage3 \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage4 \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage5 \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage6 \
    && rm -rf /usr/local/freesurfer/8.1.0/subjects/fsaverage_sym \
    && echo "export FREESURFER_HOME=/usr/local/freesurfer/8.1.0" >> ~/.bashrc \
    && echo "source /usr/local/freesurfer/8.1.0/SetUpFreeSurfer.sh" >> ~/.bashrc \
    && rm -rf /var/lib/apt/lists/* \
    && rm /tmp/fs.deb

ENV FREESURFER_HOME=/usr/local/freesurfer/8.1.0

RUN $FREESURFER_HOME/python/bin/python3 -m pip install --upgrade pip \
    && bash -c "yes y | PIP_DEFAULT_TIMEOUT=1000 $FREESURFER_HOME/bin/fs_install_cuda"

#RUN bash -c "yes y | $FREESURFER_HOME/bin/fs_install_cuda"

# install fsl
RUN wget -qO /tmp/getfsl.sh https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh
RUN bash /tmp/getfsl.sh /usr/local/fsl \
    && rm /tmp/getfsl.sh \
    && rm -rf /usr/local/fsl/pkgs \
    && rm -rf /usr/local/fsl/src \
    && rm -rf /usr/local/fsl/doc \
    && rm -rf /usr/local/fsl/extras \
    && rm -rf /usr/local/fsl/data/atlases \
    && rm -rf /usr/local/fsl/data/possum \
    && rm -f /usr/local/fsl/bin/eddy_cuda* \
    && rm -f /usr/local/fsl/bin/ptx2_gpu* \
    && rm -f /usr/local/fsl/bin/bedpostx_gpu* \
    && echo "export FSLDIR=/usr/local/fsl" >> ~/.bashrc \
    && echo "source $FSLDIR/etc/fslconf/fsl.sh" >> ~/.bashrc \
    && echo "export PATH=$FSLDIR/bin:$PATH" >> ~/.bashrc

# install simnibs into /opt (NOT the default $HOME=/root) so it is world-readable and
# therefore usable when the container runs as an arbitrary non-root UID (Apptainer/HPC,
# or `docker run --user`). The installer targets $HOME, so set HOME=/opt for this layer
# only; runtime HOME is configured near the end. test -x fails the build fast if the
# installer ever changes its default layout. The installer leaves most of the tree
# owner-only (it ran as root), which blocks a non-root UID from traversing into the
# bundled python -- chmod a+rX makes dirs traversable and keeps executables runnable for all.
RUN wget -qO /tmp/simnibs.tar.gz https://github.com/simnibs/simnibs/releases/download/v4.5.0/simnibs_installer_linux.tar.gz \
    && tar -xzf /tmp/simnibs.tar.gz -C /tmp \
    && HOME=/opt /tmp/simnibs_installer/install -s \
    && test -x /opt/SimNIBS-4.5/bin/charm \
    && rm -rf /tmp/simnibs_installer \
    && rm -rf /opt/SimNIBS-4.5/documentation \
    && rm -rf /opt/SimNIBS-4.5/examples \
    && chmod -R a+rX /opt/SimNIBS-4.5 \
    && rm /tmp/simnibs.tar.gz

# install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C /tmp -xvj bin/micromamba \
    && mv /tmp/bin/micromamba /usr/local/bin/micromamba \
    && rm -rf /tmp/bin
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN micromamba create -y -n neuro -c conda-forge \
    --override-channels \
    python=3.12 \
    "mne>=1.7.0" \
    pymeshfix \
    antspyx \
    trimesh \
    networkx \
    nibabel \
    scipy \
    numpy \
    rtree \
    pyembree \
    && micromamba clean --all --yes \
    && rm -rf /opt/conda/pkgs/*
ENV PATH="/opt/conda/envs/neuro/bin:$PATH"

# embreex gives trimesh an embree-accelerated ray backend, making the AUTHORITATIVE
# trimesh.proximity.signed_distance ~4-14x faster (the inside/outside ray-cast is its
# bottleneck). Used by gather_surfaces' BEM nesting repair. Modern trimesh prefers
# embreex over the conda-forge pyembree above (which it silently ignores).
RUN pip install --no-cache-dir embreex

# copy essential files
RUN wget -qO /tmp/cerebellum.tar.gz "https://github.com/christianbuda/parrot-neuro/releases/download/cerebellum_template/cerebellum_template.tar.gz" \
    && tar -xzf /tmp/cerebellum.tar.gz -C /home \
    && rm /tmp/cerebellum.tar.gz
RUN wget -qO /tmp/bigbrain.tar.gz "https://github.com/christianbuda/parrot-neuro/releases/download/bigbrain_scans/bigbrain_scans.tar.gz" \
    && tar -xzf /tmp/bigbrain.tar.gz -C /home \
    && rm /tmp/bigbrain.tar.gz
RUN wget -qO /home/bigbrain_scans/full16_100um_2009b_sym.nii.gz "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Volumes/MNI-ICBM152_Space/nii/full16_100um_2009b_sym.nii.gz"
COPY Schaefer2018_LocalGlobal /home/Schaefer2018_LocalGlobal
COPY scripts /scripts

# set fs license and make it so that freesurfer and fsl are sourced in non interactive shells
ENV FS_LICENSE=/SUBJECTS/license.txt

# Rootless defaults. When the container runs as an arbitrary UID there is no /etc/passwd
# entry, so HOME defaults to "/" (unwritable) and getpass.getuser() / whoami fail. A
# writable HOME and a USER name fix both. The orchestrator overrides HOME with a per-subject
# scratch at runtime; these are the standalone fallbacks (e.g. `docker run --user`).
ENV HOME=/tmp
ENV USER=parrot

# Set default command
ENTRYPOINT [ "/bin/bash", "--login", "-c", "/scripts/run_pipeline.sh \"$@\"", "_" ]
