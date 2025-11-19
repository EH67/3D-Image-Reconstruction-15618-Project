# 3D Image Reconstruction

## Relavant Links
- Project Proposal: https://github.com/EH67/3D-Image-Reconstruction-15618-Project/blob/main/Project_Proposal.md

## Setup Instructions (for running on GHC Machines)
### Clone the repository
```shell
git clone git@github.com:EH67/3D-Image-Reconstruction-15618-Project.git
```
### Setting up Python virtual environment
For our project, we'll be writing the basic code in Python (I/O, visualization, etc) and calling CUDA kernels that we have implemented directly. This requires a specific package "pybind11", which cannot be installed directly on the GHC machine. Thus, we will be setting up a virtual environment and downloading it there.

Run the next few commands in order:
1. Travel to the directory containing all the code.
    - `cd code`

2. Create a python virtualenv in /code/venv/
    - `python3 -m venv venv`
3. Install necessary packages in the venv.
    - `./venv/bin/pip3 install pybind11 numpy`

### Create build dir & run CMake
This step only needs to be done once (unless the CMake is edited, then delete the build dir and repeat all the steps again).

1. Create build directory and travel to it
    - `mkdir build && cd build`
2. Run cmake using the CMakeLists. in /code
    - `cmake ..`


### Building the Project & Running Code
1. Ensure you are in the build/ directory
2. Compile the CUDA kernel functions and CPP wrapper
    - `make`
3. Run the python main function.
    - `../venv/bin/python ../main.py`