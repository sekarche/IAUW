#!/bin/bash

echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "📸 Installing camera & OpenCV dependencies..."
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-opencv \
  libjpeg-dev \
  libtiff-dev \
  libopenjp2-7 \
  libopenblas-dev \
  libatlas-base-dev \
  python3-libcamera \
  libcamera-apps \
  libcap-dev

echo "🧱 Creating virtual environment..."
python3 -m venv enhancer-env
source enhancer-env/bin/activate

echo "🐍 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing Python packages..."
pip install \
  numpy \
  opencv-python-headless \
  pillow \
  torch \
  torchvision \
  onnx \
  onnxruntime \
  psutil \
  picamera2 \
  scikit-learn \
  matplotlib

echo "✅ All dependencies installed. To activate environment later:"
echo "source enhancer-env/bin/activate"
