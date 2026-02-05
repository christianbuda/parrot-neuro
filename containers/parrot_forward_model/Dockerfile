# --- Builder Stage ---
FROM debian:bookworm-slim AS builder

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libgmp-dev \
    libmpfr-dev \
    zlib1g-dev \
    libboost-dev \
    libinsighttoolkit5-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Manually Install CGAL 6.0.1
WORKDIR /tmp
RUN wget -q https://github.com/CGAL/cgal/releases/download/v6.1.1/CGAL-6.1.1.tar.xz \
    && tar -xf CGAL-6.1.1.tar.xz \
    && cd CGAL-6.1.1 \
    && cmake . -DCMAKE_BUILD_TYPE=Release \
    && make install

# 3. Build your application
WORKDIR /app
COPY mesher.cpp CMakeLists.txt ./

RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# --- Runtime Stage ---
FROM python:3.12-slim

WORKDIR /scripts

# Install runtime libraries
RUN apt-get update && apt-get install -y \
    git \
    libgmp10 \
    libmpfr6 \
    zlib1g \
    libinsighttoolkit5-dev \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/mesher /usr/local/bin/mesher

# Now copy python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy python scripts
COPY place_electrodes.py place_dipoles.py nifti_to_inr.py mesh_to_vtu.py ./
