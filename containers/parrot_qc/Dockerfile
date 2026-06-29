# Parrot QC image: lightweight viz/render deps + the baked QC package.
#
# Follows the same blueprint as the other Parrot images: the QC code (the qc/
# package) is COPYed into the image (see bottom), so editing it requires a
# rebuild (./bin/build.sh qc), just like the other containers' scripts.
#
# 3D snapshots use pyvista offscreen via Xvfb + Mesa software GL (no GPU needed).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYVISTA_OFF_SCREEN=true \
    MPLBACKEND=Agg

# Xvfb (for pyvista.start_xvfb) + Mesa software GL/GLX runtime libs so VTK can
# render offscreen without a GPU or a real display.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libosmesa6 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libxt6 \
        libxi6 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Bake the QC package (run as: python /qc/run_qc.py ...).
COPY qc /qc
WORKDIR /qc
