# Use the stable DUNE 2.10 core image
FROM registry.dune-project.org/docker/ci/dune:2.10

# 1. Force HOME back to /duneci so the pre-configured dune.opts works perfectly
ENV HOME=/duneci
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=""

# Switch to root to install packages
USER root

# --- 2. Install System and Python Dependencies ---
RUN apt-get update && apt-get install -y \
    python3-pip \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMEEG and other useful python libraries
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    "numpy<2.0" \
    scipy \
    h5py \
    trimesh \
    pymeshlab \
    meshio \
    nibabel \
    rtree \
    embreex

# --- 3. Setup DUNEuro Workspace ---
WORKDIR /duneci/modules

# Pin dune-pdelab to the tested commit on releases/2.10 for reproducibility
# (a plain branch clone tracks the moving branch tip; a full clone + checkout of
# the exact SHA reproduces the validated image regardless of later branch moves).
# dune-pdelab: 17a2c0509b5e0912e1bec6783eeb15d8a342bb5b
RUN git clone https://gitlab.dune-project.org/pdelab/dune-pdelab.git && \
    cd dune-pdelab && git checkout 17a2c0509b5e0912e1bec6783eeb15d8a342bb5b

# Pin duneuro and duneuro-py to specific tested commits for reproducibility.
# duneuro:    8f344b4da9c128ddf3e47af5ec136d05a3aeb162
# duneuro-py: bb9f5e609055001857b230d1c59c55e86799fabf
RUN git clone https://gitlab.dune-project.org/duneuro/duneuro.git && \
    cd duneuro && git checkout 8f344b4da9c128ddf3e47af5ec136d05a3aeb162 && cd .. && \
    git clone https://gitlab.dune-project.org/duneuro/duneuro-py.git && \
    cd duneuro-py && git checkout bb9f5e609055001857b230d1c59c55e86799fabf && \
    git submodule update --init --recursive

# --- 4. Compile DUNEuro ---
RUN ./dune-common/bin/dunecontrol --opts=/duneci/dune.opts --builddir=$(pwd)/build-release all

# --- 6. Install OpenMEEG CLI Binaries ---
# pip only installed the Python API. We use GitHub's API to fetch the latest C++ binaries.
RUN apt-get update && apt-get install -y curl jq && \
    curl -L https://files.inria.fr/OpenMEEG/download/OpenMEEG-2.4.9999-Linux.tar.gz -o openmeeg_bin.tar.gz && \
    mkdir -p /opt/openmeeg && \
    tar -xzf openmeeg_bin.tar.gz -C /opt/openmeeg --strip-components=1 && \
    cp -a /opt/openmeeg/bin/* /usr/local/bin/ && \
    cp -a /opt/openmeeg/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm openmeeg_bin.tar.gz

# --- 5. Final Setup ---
ENV PYTHONPATH="/duneci/modules/build-release/duneuro-py/src:${PYTHONPATH}"

RUN mkdir /scripts
COPY make_leadfield_openmeeg.py make_leadfield_duneuro.py \
     dti_to_conductivity_tensors.py mesh_io.py \
     make_leadfield_artifacts.py make_leadfield_hartmut_muscle.py /scripts/

WORKDIR /pipeline

COPY geometry.geom conductivities.cond neuronal_strength_dict.json .

ENTRYPOINT ["python3"]
