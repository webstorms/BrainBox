#!/bin/bash

# Abort on any failed command
set -e

# Install dependencies
printf "Loading dependencies..."
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda env update -f environment.yml --prune > /dev/null 2>&1
conda activate brainbox

# Run tests
printf "Running tests..."
pytest .

# Run formatting and linting
printf "Running formatting..."
black --check .