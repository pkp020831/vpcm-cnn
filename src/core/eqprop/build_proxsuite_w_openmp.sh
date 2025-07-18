#!/usr/bin/env bash

set -e  # Exit on any error

# Function to detect virtual environment type
detect_venv() {
    if [[ -n "$CONDA_DEFAULT_ENV" || -n "$CONDA_PREFIX" ]]; then
        echo "conda"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        # Check if it's a UV environment (no pip module)
        if ! python -m pip --version >/dev/null 2>&1; then
            echo "uv"
        else
            echo "venv"
        fi
    else
        echo "global"
    fi
}

# Function to get installation prefix
get_install_prefix() {
    local env_type=$1
    case $env_type in
        "conda")
            echo "$CONDA_PREFIX"
            ;;
        "venv"|"uv")
            echo "$VIRTUAL_ENV"
            ;;
        "global")
            echo "$HOME/.local"
            ;;
    esac
}

# Function to install system dependencies
install_system_deps() {
    local env_type=$1
    echo "Installing system dependencies..."

    # Check if running on Ubuntu/Debian
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y cmake libeigen3-dev build-essential git

        # For UV environments, install LLVM OpenMP to match conda behavior
        if [[ "$env_type" == "uv" ]]; then
            echo "Installing LLVM OpenMP for UV environment..."
            sudo apt-get install -y libomp-dev libomp5
            echo "✅ LLVM OpenMP installed for UV environment"
        fi

        # Install SIMDE if available
        if apt-cache search libsimde-dev | grep -q libsimde-dev; then
            sudo apt-get install -y libsimde-dev
            echo "✅ SIMDE installed from package manager"
        else
            echo "⚠️  SIMDE not available in package manager, will build without vectorization support"
        fi
    # Check if running on macOS
    elif command -v brew >/dev/null 2>&1; then
        brew install cmake eigen git

        # For UV environments, install LLVM OpenMP
        if [[ "$env_type" == "uv" ]]; then
            echo "Installing LLVM OpenMP for UV environment..."
            brew install libomp
            echo "✅ LLVM OpenMP installed for UV environment"
        fi

        # Install SIMDE via Homebrew
        if brew list simde >/dev/null 2>&1 || brew install simde; then
            echo "✅ SIMDE installed via Homebrew"
        else
            echo "⚠️  SIMDE installation failed, will build without vectorization support"
        fi
    # Check if running on Fedora/RHEL
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y cmake eigen3-devel gcc-c++ make git

        # For UV environments, install LLVM OpenMP
        if [[ "$env_type" == "uv" ]]; then
            echo "Installing LLVM OpenMP for UV environment..."
            sudo dnf install -y libomp libomp-devel
            echo "✅ LLVM OpenMP installed for UV environment"
        fi

        echo "⚠️  SIMDE not readily available on Fedora/RHEL, will build without vectorization support"
    else
        echo "Warning: Could not detect package manager. Please install cmake and eigen3 manually."
        if [[ "$env_type" == "uv" ]]; then
            echo "⚠️  Warning: Also manually install LLVM OpenMP for UV environment compatibility"
        fi
    fi
}

# Function to install Python dependencies
install_python_deps() {
    local env_type=$1

    case $env_type in
        "conda")
            echo "Installing Python dependencies via conda..."
            conda install -c conda-forge cmake eigen simde -y
            ;;
        "venv"|"global")
            echo "Installing Python dependencies via pip..."
            python -m pip install --upgrade pip
            python -m pip install cmake numpy
            ;;
        "uv")
            echo "Installing Python dependencies via uv..."
            if command -v uv >/dev/null 2>&1; then
                uv add cmake numpy
            else
                echo "Warning: uv command not found. Trying with direct python installation..."
                # For UV environments, we can still install packages directly
                python -c "import sys; sys.path.insert(0, '$VIRTUAL_ENV/lib/python*/site-packages')"
                # Install to the UV environment using python directly
                curl -sSL https://bootstrap.pypa.io/get-pip.py | python
                python -m pip install cmake numpy
            fi
            ;;
    esac
}

# Main execution
main() {
    local env_type
    local install_prefix

    echo "=== ProxSuite Build Script ==="

    # Detect environment
    env_type=$(detect_venv)
    echo "Detected environment: $env_type"

    # Warn if using global environment
    if [[ "$env_type" == "global" ]]; then
        echo "⚠️  WARNING: No virtual environment detected!"
        echo "   Installing to global Python environment may cause conflicts."
        echo "   Consider activating a virtual environment (venv or conda) before running this script."
        echo ""
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi

    # Get installation prefix
    install_prefix=$(get_install_prefix "$env_type")
    echo "Installation prefix: $install_prefix"

    # Create installation directory if it doesn't exist
    mkdir -p "$install_prefix"

    # Install dependencies
    # install_system_deps "$env_type"
    install_python_deps "$env_type"

    # Clone and build ProxSuite
    echo "Cloning ProxSuite repository..."
    cd "$HOME" || exit 1

    # Remove existing directory if it exists
    if [[ -d "proxsuite" ]]; then
        echo "Removing existing proxsuite directory..."
        rm -rf proxsuite
    fi

    git clone --recursive https://github.com/Simple-Robotics/proxsuite.git
    cd proxsuite || exit 1

    echo "Building ProxSuite..."
    mkdir -p build && cd build || exit 1

    # Check if SIMDE is available
    local cmake_flags=()
    cmake_flags+=(-DCMAKE_INSTALL_PREFIX="$install_prefix")
    cmake_flags+=(-DCMAKE_BUILD_TYPE=Release)
    cmake_flags+=(-DBUILD_PYTHON_INTERFACE=ON)
    cmake_flags+=(-DBUILD_TESTING=OFF)
    cmake_flags+=(-DBUILD_WITH_OPENMP_SUPPORT=ON)

    # For UV environments, force use of LLVM OpenMP to match conda behavior
    if [[ "$env_type" == "uv" ]]; then
        echo "Configuring LLVM OpenMP for UV environment..."
        # Try to find LLVM OpenMP library paths
        if [[ -f "/usr/lib/x86_64-linux-gnu/libomp.so" ]]; then
            cmake_flags+=(-DOpenMP_CXX_FLAGS="-fopenmp=libomp")
            cmake_flags+=(-DOpenMP_CXX_LIB_NAMES="omp")
            cmake_flags+=(-DOpenMP_omp_LIBRARY="/usr/lib/x86_64-linux-gnu/libomp.so")
        elif [[ -f "/usr/local/lib/libomp.so" ]]; then
            cmake_flags+=(-DOpenMP_CXX_FLAGS="-fopenmp=libomp")
            cmake_flags+=(-DOpenMP_CXX_LIB_NAMES="omp")
            cmake_flags+=(-DOpenMP_omp_LIBRARY="/usr/local/lib/libomp.so")
        else
            echo "⚠️  LLVM OpenMP not found, falling back to system OpenMP"
        fi
    fi

    # Add compiler flags for better memory management and fewer warnings
    cmake_flags+=(-DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-unused-parameter -Wno-cast-qual -fno-strict-aliasing -DPY_SSIZE_T_CLEAN")
    cmake_flags+=(-DCMAKE_C_FLAGS="-Wno-deprecated-declarations -Wno-unused-parameter -Wno-cast-qual -fno-strict-aliasing")
    # Python binding optimization for memory management
    cmake_flags+=(-DBUILD_WITH_COLLISION_SUPPORT=OFF)
    cmake_flags+=(-DPYTHON_EXECUTABLE="$(which python)")

    # Check for SIMDE availability
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists simde; then
        echo "✅ SIMDE found via pkg-config, enabling vectorization"
        cmake_flags+=(-DBUILD_WITH_VECTORIZATION_SUPPORT=ON)
    elif [[ -d "/usr/include/simde" || -d "/usr/local/include/simde" || -d "$VIRTUAL_ENV/include/simde" ]]; then
        echo "✅ SIMDE headers found, enabling vectorization"
        cmake_flags+=(-DBUILD_WITH_VECTORIZATION_SUPPORT=ON)
    else
        echo "⚠️  SIMDE not found, disabling vectorization support"
        cmake_flags+=(-DBUILD_WITH_VECTORIZATION_SUPPORT=OFF)
    fi

    cmake .. "${cmake_flags[@]}"

    make install -j"$(nproc)"
    # clean up build files
    make clean
    echo "✅ ProxSuite installation completed successfully!"
    echo "   Installed to: $install_prefix"

    # Add to PATH if necessary
    if [[ "$env_type" == "global" ]]; then
        echo ""
        echo "Note: You may need to add $install_prefix/bin to your PATH:"
        echo "export PATH=\"$install_prefix/bin:\$PATH\""
    fi
}

# Run main function
main "$@"
