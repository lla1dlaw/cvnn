#!/bin/bash

# ---
# Script to recursively remove a hardcoded list of directories.
# WARNING: The 'rm -rf' command is permanent and cannot be undone.
# Double-check the directory paths in the DIRS_TO_DELETE array
# before running this script.
# ---

# --- Configuration ---
# Add the full paths of all the directories you want to delete into this list.
# Use quotes to handle paths with spaces.
DIRS_TO_DELETE=(
  "./real_metrics"
  "./real_plots"
  "./real_models"
  "./complex_metrics"
  "./complex_plots"
  "./complex_models"
)

log_file="./training_log.txt"

# --- Execution ---

# Loop through each directory specified in the array
for dir in "${DIRS_TO_DELETE[@]}"; do
  # Check if the path exists AND is a directory
  if [ -d "$dir" ]; then
    # The 'rm -rf' command recursively (-r) and forcefully (-f)
    # removes the directory and all its contents.
    rm -rf "$dir"
    echo "Deleted $dir"
  fi
done

if [ -f "$log_file" ]; then
  rm "$log_file"
  echo "Deleted $log_file"
fi
