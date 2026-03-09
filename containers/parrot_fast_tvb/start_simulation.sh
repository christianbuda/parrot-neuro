#!/bin/bash
# Exit immediately if any command fails
set -e

INPUT_DIR="/input"
OUTPUT_DIR="/output"

# Check for input directory and exit with an error code if missing
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory ($INPUT_DIR) does not exist. Terminating." >&2
    exit 1
fi

# Check for output directory and exit with an error code if missing
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory ($OUTPUT_DIR) does not exist. Terminating." >&2
    exit 1
fi

# Replace the shell process with the tvb binary and pass all arguments
exec /tvb "$@"