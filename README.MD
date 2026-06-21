# junction_detector

**MMI 713 - Applied Parallel Programming on GPU**
**Term Project**
**Student:** Merve Nur Zembil
**Student Number:** 2376242

## Overview

`junction_detector` is a CPU/CUDA project for detecting complex junction candidates in HERE HD map data. The input is a set of HERE HDLM JSON tile files and configuration files. The output is a JSON file that can be used by the MATLAB visualization script of my project group to display detected junctions, primary/boundary roads, secondary/interior roads, and non-junction background roads.

The project follows this general pipeline:

```text
HERE HDLM JSON files + config
        |
        v
export_graph_input.py
        |
        v
intermediate/cpp_input_configXX.json
        |
        +--------------------------+
        |                          |
        v                          v
junction_detector_cpu.cpp          junction_detector_gpu.cu
     (CPU baseline)                  (CUDA implementation)
        |                          |
        v                          v
exports/cpu_configXX.json      exports/gpu_configXX.json
        |                          |
        +--------------------------+
        |
        v
run_road_network.m

```

The Python script is used only for data preparation. It reads the copied HERE/config files, classifies links, builds a compact CSR graph, and writes an intermediate JSON file. The C++ and CUDA programs then read this intermediate graph and run the junction detection algorithm.

Similarly, matlab script is only used for visualization purposes. It displays the detected junctions in a given map.

## Prerequisites

The project was tested on Ubuntu with GCC 11.4 and CUDA 13.2.

Install the basic build tools, CMake, and the JSON library used by the C++ code:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake nlohmann-json3-dev
```

The CUDA implementation also requires the NVIDIA CUDA Toolkit. Check that `nvcc` is available:

```bash
nvcc --version
```

The preprocessing script uses Python to prepare the compact graph input for the C++ and CUDA detectors. A virtual environment is recommended so that the Python package versions are fixed.

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The required Python packages are listed in requirements.txt. Beware that numpy version <2 is required, otherwise the `export_graph_input.py` python script does not work. 

Before running the exporter, activate the environment:

```bash
source .venv/bin/activate
```

## Build and Run

Run all commands from the root of the project folder:

```bash
cd junction_detector
```

Generate the intermediate graph input with a given config file's number:

```bash
python3 export_graph_input.py 12
```

Build CPU and GPU executables:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j
cd ..
```

Run the CPU baseline with a given config file's number:

```bash
./build/junction_detector_cpu 12
```

Run the GPU baseline with a given config file's number:

```bash
./build/junction_detector_gpu 12
```

The CPU output is:

```text
exports/cpu_config12.json
```

The CUDA output is:

```text
exports/gpu_config12.json
```

## MATLAB Visualization Output

The MATLAB visualization script reads final CPU/GPU exported JSON files. It runs with the input of the path of the JSON export to be visualized.

## Helper Script

To generate the intermediate graph input of all given configs under configs/, build and run CPU and GPU codes, and then visualize these outputs using matlab, simply run the following script after sourcing .venv:

```bash
./build_and_run.sh
```

For this to be able to use matlab, add it to your path:

```bash
ls /usr/local/MATLAB            # this outputs R2025b for me so the path is: 
export PATH=/usr/local/MATLAB/R2025b/bin:$PATH
```