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
    rtree

# --- 3. Setup DUNEuro Workspace ---
WORKDIR /duneci/modules

# Pull the stable 2.10 release of dune-pdelab
RUN git clone --branch releases/2.10 https://gitlab.dune-project.org/pdelab/dune-pdelab.git 

# Pull the latest duneuro and duneuro-py 
RUN git clone https://gitlab.dune-project.org/duneuro/duneuro.git && \
    git clone --recursive https://gitlab.dune-project.org/duneuro/duneuro-py.git

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
COPY make_leadfield_openmeeg.py make_leadfield_duneuro.py /scripts/

WORKDIR /pipeline

COPY geometry.geom conductivities.cond neuronal_strength_dict.json .

ENTRYPOINT ["python3"]
