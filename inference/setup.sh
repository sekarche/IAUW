#!/bin/bash

# ==============================
# Underwater Image Enhancement Setup Script for Raspberry Pi
# ==============================

echo "===== Underwater Image Enhancement Setup ====="
echo "This script will install all required packages and set up your Raspberry Pi environment."

# ---------- System Update ----------
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# ---------- System Dependencies ----------
echo "Installing essential system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopenjp2-7 \
    libtiff5 \
    libjpeg-dev \
    libopenblas-dev \
    libatlas-base-dev

# ---------- Virtual Environment ----------
echo "Creating and activating Python virtual environment (underwater_env)..."
python3 -m venv underwater_env
source underwater_env/bin/activate

# ---------- Python Dependencies ----------
echo "Upgrading pip and installing required Python packages..."
pip install --upgrade pip

# Try to install from requirements.txt if present; else install directly
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    pip install \
        numpy==1.24.3 \
        opencv-python-headless==4.8.0.74 \
        pillow==10.0.0 \
        torch==1.13.1 \
        torchvision==0.14.1 \
        onnxruntime==1.15.1
fi

# ---------- Project Directory Structure ----------
echo "Setting up project directories..."
mkdir -p models
mkdir -p input_images
mkdir -p output_images
mkdir -p weights

echo "===== Setup Complete! ====="
echo "Activate your environment with: source underwater_env/bin/activate"
echo "Place your ONNX models in ./models and your test images in ./input_images."
