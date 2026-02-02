FROM python:3.12-slim

WORKDIR /scripts

# 1. Install git (Required to fetch the github packages)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. Now copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy scripts
COPY place_electrodes.py place_dipoles.py ./
