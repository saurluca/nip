#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if uv is installed
if ! command_exists uv; then
    echo "Error: uv is not installed. Please install it first."
    echo "Visit: https://astral.sh/uv/install"
    exit 1
fi

# Check if required packages are installed, if not install them
if ! command_exists jupyter; then
    echo "Installing required packages..."
    uv pip install nbconvert pyppeteer jupyter
fi

# Check if lab3.ipynb exists
if [ ! -f "lab3.ipynb" ]; then
    echo "Error: lab3.ipynb not found in the current directory"
    exit 1
fi

# Convert the notebook to PDF
jupyter nbconvert --to pdf lab3.ipynb

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion successful! Output file: lab3.pdf"
else
    echo "Conversion failed. Please check the error messages above."
fi
