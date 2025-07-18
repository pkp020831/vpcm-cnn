#!/usr/bin/env bash

set -e  # Exit on any error

# Define the temporary directory for proxsuite
PROXSUITE_DIR="$HOME/proxsuite"

# Remove existing proxsuite directory if it exists to ensure a clean clone
echo "Removing existing ${PROXSUITE_DIR} directory..."
rm -rf "${PROXSUITE_DIR}"

# Clone proxsuite
echo "Cloning proxsuite repository..."
git clone --recursive https://github.com/Simple-Robotics/proxsuite.git "${PROXSUITE_DIR}"
cd "${PROXSUITE_DIR}" || exit 1
git checkout tags/v0.6.4
git submodule update --init --recursive

echo "Patching ProxSuite CMake files..."
# Patch CMakeLists.txt to require CMake >= 3.22
sed -i 's/cmake_minimum_required(VERSION 3.10)/cmake_minimum_required(VERSION 3.22)/g' CMakeLists.txt
# Comment out the CMake version check in cmake-module/base.cmake
sed -i '/if(CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 3\.22)/i # Patched by build.sh' cmake-module/base.cmake
sed -i '/if(CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 3\.22)/,/endif\(\)/s/^/# /' cmake-module/base.cmake
# Patch pybind11/CMakeLists.txt to require CMake >= 3.15
sed -i 's/cmake_minimum_required(VERSION 3.4)/cmake_minimum_required(VERSION 3.15)/g' bindings/python/external/pybind11/CMakeLists.txt

# Install dependencies including a recent CMake version and simde from conda-forge
echo "Installing conda dependencies (cmake>=3.22, eigen, simde) from conda-forge..."
conda install -c conda-forge cmake eigen simde -y

# Clean and create build directory
echo "Cleaning and creating build directory..."
rm -rf build
mkdir build && cd build || exit 1

# Configure and build
echo "Configuring CMake..."
cmake .. -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=ON -DBUILD_TESTING=OFF -DBUILD_WITH_OPENMP_SUPPORT=ON

echo "Building and installing proxsuite..."
make -j1
make install

echo "Proxsuite build and installation complete."
