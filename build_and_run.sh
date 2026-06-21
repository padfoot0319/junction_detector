#!/usr/bin/env bash

# Build, export graph inputs, run CPU/GPU detectors, and create visualizations.
# Run this from the root of the project folder: ./build_and_run.sh

set -u

cd "$(dirname "$0")"

rm -rf build
mkdir -p build exports visualizations

cmake -S . -B build > /dev/null
cmake --build build -j > /dev/null

FAILED=()
COUNT=0
VIS_COUNT=0

mapfile -t CONFIG_FILES < <(find configs -maxdepth 1 -type f -name 'config*.json' | sort -V)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "No config files found."
    exit 1
fi

is_synthetic() {
    grep -q '"Synthetic"[[:space:]]*:[[:space:]]*true' "$1"
}

# Export graph inputs only for real configs.
# Synthetic configs already have their intermediate/graph_input_configN.json files.
for CFG in "${CONFIG_FILES[@]}"; do
    NAME="$(basename "${CFG}" .json)"
    NUM="${NAME#config}"

    if is_synthetic "${CFG}"; then
        continue
    fi

    if ! python export_graph_input.py "${NUM}" > /dev/null 2>&1; then
        FAILED+=("export:${NAME}")
    fi
done

echo "Run     Config           Roads Secondary   Primary     Edges   Junctions   Algorithm_s"

for CFG in "${CONFIG_FILES[@]}"; do
    NAME="$(basename "${CFG}" .json)"
    NUM="${NAME#config}"

    if ! ./build/junction_detector_cpu "${NUM}"; then
        FAILED+=("cpu:${NAME}")
    fi

    COUNT=$((COUNT + 1))
done

for CFG in "${CONFIG_FILES[@]}"; do
    NAME="$(basename "${CFG}" .json)"
    NUM="${NAME#config}"

    if ! ./build/junction_detector_gpu "${NUM}"; then
        FAILED+=("gpu:${NAME}")
    fi
done

# Visualize only real configs.
# Synthetic configs are intentionally skipped because their images are too large.
if command -v matlab > /dev/null 2>&1; then
    : > visualizations/matlab_visualization.log

    for CFG in "${CONFIG_FILES[@]}"; do
        NAME="$(basename "${CFG}" .json)"
        NUM="${NAME#config}"

        if is_synthetic "${CFG}"; then
            continue
        fi

        for OUT in "exports/cpu_config${NUM}.json" "exports/gpu_config${NUM}.json"; do
            if [[ ! -f "${OUT}" ]]; then
                FAILED+=("missing_output:$(basename "${OUT}")")
                continue
            fi

            if matlab -batch "run_road_network('${OUT}')" >> visualizations/matlab_visualization.log 2>&1; then
                VIS_COUNT=$((VIS_COUNT + 1))
            else
                FAILED+=("matlab:$(basename "${OUT}")")
            fi
        done
    done
else
    echo "MATLAB was not found. Skipping visualization."
fi

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "Processed ${COUNT} configs. Created ${VIS_COUNT} visualizations."
else
    echo "Some steps failed: ${FAILED[*]}"
    exit 1
fi
