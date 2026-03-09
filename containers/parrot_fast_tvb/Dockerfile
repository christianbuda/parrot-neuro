FROM debian:stable AS builder

RUN apt-get update && \
    apt-get -y install \
    gcc \
    gsl-bin \
    libgsl-dev 

WORKDIR /build
COPY tvb_multicore.c .

# Compile the C code directly inside the builder stage
RUN gcc -Wall -pedantic -Wextra -std=c99 -msse2 -O3 -ftree-vectorize -ffast-math -funroll-loops -fomit-frame-pointer -m64 tvb_multicore.c -o tvb -lgsl -lgslcblas -lm -lpthread

# ==========================================
# STAGE 2: Runner
# ==========================================
FROM debian:stable-slim

RUN apt-get update && \
    apt-get -y install \
    gsl-bin \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/* # Cleans up apt cache to save space

# Copy the compiled binary from the builder stage
COPY --from=builder /build/tvb /tvb

# Copy and set up the start script
COPY start_simulation.sh /start_simulation.sh
RUN chmod 775 /start_simulation.sh

# Set the script as the entrypoint so arguments are passed to it automatically
ENTRYPOINT ["/start_simulation.sh"]