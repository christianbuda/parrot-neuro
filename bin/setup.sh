#!/bin/bash

# docker create -it --gpus all --name recon_pipe -v /srv/nfs-data/sisko/christian/SUBJECTS:/SUBJECTS --entrypoint /bin/bash christianbuda/recon_container:latest


# List of images to download
IMAGES=(
    "deepmi/fastsurfer:latest"
    "khanlab/hippunfold:latest"
    "christianbuda/parrot_MRI_reconstruction:latest"
    "christianbuda/parrot_forward_model:latest"
)

# Variable to accumulate the final clean report
REPORT=""

# Clear screen initially
clear

# Loop through each image
for img in "${IMAGES[@]}"; do
    
    # Print the current status (Clean report + Current action)
    echo "$REPORT"
    echo "---------------------------------------------------"
    echo "Checking for: $img..."

    # Check if image exists locally
    if [[ "$(docker images -q "$img" 2> /dev/null)" == "" ]]; then
        echo "Not found locally. Downloading (this may take time)..."
        echo ""
        
        # Run docker pull, allowing the progress bars to show
        docker pull "$img"
        
        # Check if the download was successful
        if [ $? -eq 0 ]; then
            # CLEAR THE SCREEN to wipe the messy progress bars
            clear
            # Add this success to our report
            REPORT="${REPORT}Success! Downloaded: $img"$'\n'
        else
            echo "ERROR: Failed to download $img"
            exit 1
        fi
    else
        # If found locally, we still clear and add to report for consistency
        clear
        REPORT="${REPORT}Success! Found locally: $img"$'\n'
    fi
done

# Print the final clean report one last time
echo "$REPORT"
