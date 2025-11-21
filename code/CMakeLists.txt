# --------------------------------------------------------------------------------------
# 1. COMPILER FIXES (To resolve CUDA/GCC incompatibility)
# --------------------------------------------------------------------------------------

# Explicitly set the host C/C++ compiler to the supported version (GCC 11).
set(CMAKE_C_COMPILER "gcc-11")
set(CMAKE_CXX_COMPILER "g++-11")

# Explicitly tells nvcc which host compiler binary to use (via -ccbin).
set(CMAKE_CUDA_HOST_COMPILER "g++-11")

cmake_minimum_required(VERSION 3.12)
project(cuda_reconstruction LANGUAGES CXX CUDA)

# --------------------------------------------------------------------------------------
# 2. VIRTUAL ENVIRONMENT PATH RESOLUTION (To force VENV Python)
# --------------------------------------------------------------------------------------
# Define the relative path to the VENV's interpreter and resolve to absolute path.
set(VENV_PYTHON_REL "${CMAKE_CURRENT_SOURCE_DIR}/../code/venv/bin/python")
get_filename_component(VENV_PYTHON_ABS "${VENV_PYTHON_REL}" ABSOLUTE)

# Force CMake to use this specific interpreter for ALL Python checks.
set(Python_EXECUTABLE "${VENV_PYTHON_ABS}")
set(Python3_EXECUTABLE "${VENV_PYTHON_ABS}")

# --------------------------------------------------------------------------------------
# 3. DEPENDENCY FINDING for Python (numpy & pybind11 & opencv)
# --------------------------------------------------------------------------------------
# Find Python and NumPy (sets Python3::NumPy properties)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# This block calculates the precise path to the config file shown in your image:
# /venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/

# Get the VENV root directory by stripping '/bin/python' from the executable path.
# VENV_ROOT=/afs/andrew.cmu.edu/usr<user number>/<andrewID>/<rest of local path>/3D-Image-Reconstruction-15618-Project/code/venv/
get_filename_component(VENV_BIN_DIR ${Python3_EXECUTABLE} DIRECTORY) # Result: .../venv/bin
get_filename_component(VENV_ROOT ${VENV_BIN_DIR} DIRECTORY)           # Result: .../venv

# Construct the absolute path to the pybind11 config folder.
set(pybind11_DIR "${VENV_ROOT}/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Now, find_package will use the path set in pybind11_DIR.
find_package(pybind11 CONFIG REQUIRED)

# --------------------------------------------------------------------------------------
# 4. BUILD TARGETS
# --------------------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES "75") 

# Define the shared library module.
pybind11_add_module(cuda_ops_module 
    src/bindings.cpp
    src/cuda_ops.cu
)

# Link necessary libraries.
target_link_libraries(cuda_ops_module 
    PRIVATE 
    Python3::NumPy 
    pybind11::module
    cudart_static # Link against the CUDA Runtime library
    # ${OpenCV_LIBS}
)

# Specify the dir of where to find the pybind11::module.
target_link_directories(cuda_ops_module
    PRIVATE
    ${pybind11_LIBRARY_DIRS}
    # ${OpenCV_INCLUDE_DIRS}
)