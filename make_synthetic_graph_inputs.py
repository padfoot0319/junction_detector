#!/usr/bin/env python3

import copy
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BASE_CONFIG = 7
SYNTHETIC_CONFIGS = [
    # config number, repeat count
    (8, 10),
    (9, 50),
    (10, 100),
]

ID_STRIDE = 1_000_000_000_000
SHIFT_X = 100_000.0
SHIFT_Y = 100_000.0
GRID_COLS = 10


def add_shift(values, shift):
    out = []
    for v in values:
        if v is None:
            out.append(v)
        else:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out.append(fv + shift)
                else:
                    out.append(v)
            except Exception:
                out.append(v)
    return out


def duplicate_ids(ids, repeat_count):
    out = []
    for r in range(repeat_count):
        offset = r * ID_STRIDE
        for x in ids:
            out.append(int(x) + offset)
    return out


def duplicate_csr(row_offsets, col_indices, n, repeat_count):
    new_row_offsets = [0]
    new_col_indices = []

    for r in range(repeat_count):
        vertex_offset = r * n
        for i in range(n):
            begin = int(row_offsets[i])
            end = int(row_offsets[i + 1])

            for j in col_indices[begin:end]:
                new_col_indices.append(int(j) + vertex_offset)

            new_row_offsets.append(len(new_col_indices))

    return new_row_offsets, new_col_indices


def duplicate_point_arrays(offsets, xs, ys, repeat_count):
    item_count = len(offsets) - 1
    new_offsets = [0]
    new_x = []
    new_y = []

    for r in range(repeat_count):
        grid_x = (r % GRID_COLS) * SHIFT_X
        grid_y = (r // GRID_COLS) * SHIFT_Y

        for i in range(item_count):
            begin = int(offsets[i])
            end = int(offsets[i + 1])

            new_x.extend(add_shift(xs[begin:end], grid_x))
            new_y.extend(add_shift(ys[begin:end], grid_y))
            new_offsets.append(len(new_x))

    return new_offsets, new_x, new_y


def duplicate_plot_arrays(x0, y0, x1, y1, repeat_count):
    new_x0 = []
    new_y0 = []
    new_x1 = []
    new_y1 = []

    for r in range(repeat_count):
        grid_x = (r % GRID_COLS) * SHIFT_X
        grid_y = (r // GRID_COLS) * SHIFT_Y

        new_x0.extend(add_shift(x0, grid_x))
        new_y0.extend(add_shift(y0, grid_y))
        new_x1.extend(add_shift(x1, grid_x))
        new_y1.extend(add_shift(y1, grid_y))

    return new_x0, new_y0, new_x1, new_y1


def make_synthetic(base, output_config_number, repeat_count):
    n = int(base["num_links"])
    out = copy.deepcopy(base)

    out["source_config"] = f"synthetic_from_config{BASE_CONFIG}_x{repeat_count}"
    out["num_links"] = n * repeat_count
    out["link_ids"] = duplicate_ids(base["link_ids"], repeat_count)
    out["link_type"] = base["link_type"] * repeat_count

    out["row_offsets"], out["col_indices"] = duplicate_csr(
        base["row_offsets"],
        base["col_indices"],
        n,
        repeat_count,
    )

    out["point_offsets"], out["point_x"], out["point_y"] = duplicate_point_arrays(
        base["point_offsets"],
        base["point_x"],
        base["point_y"],
        repeat_count,
    )

    out["link_plot_x0"], out["link_plot_y0"], out["link_plot_x1"], out["link_plot_y1"] = duplicate_plot_arrays(
        base["link_plot_x0"],
        base["link_plot_y0"],
        base["link_plot_x1"],
        base["link_plot_y1"],
        repeat_count,
    )

    if "background_link_ids" in base:
        out["background_link_ids"] = duplicate_ids(base["background_link_ids"], repeat_count)

    if "background_point_offsets" in base:
        out["background_point_offsets"], out["background_x"], out["background_y"] = duplicate_point_arrays(
            base["background_point_offsets"],
            base["background_x"],
            base["background_y"],
            repeat_count,
        )

    if "background_plot_x0" in base:
        (
            out["background_plot_x0"],
            out["background_plot_y0"],
            out["background_plot_x1"],
            out["background_plot_y1"],
        ) = duplicate_plot_arrays(
            base["background_plot_x0"],
            base["background_plot_y0"],
            base["background_plot_x1"],
            base["background_plot_y1"],
            repeat_count,
        )

    secondary_count = sum(1 for t in out["link_type"] if int(t) == 2)
    primary_count = sum(1 for t in out["link_type"] if int(t) == 1)

    out["classification"]["num_secondary_ids"] = secondary_count
    out["classification"]["num_primary_ids"] = primary_count

    out["summary"] = out.get("summary", {})
    out["summary"]["num_dual_vertices"] = out["num_links"]
    out["summary"]["num_dual_directed_edges"] = len(out["col_indices"])
    out["summary"]["num_background_links"] = len(out.get("background_link_ids", []))
    out["summary"]["synthetic"] = True
    out["summary"]["base_config"] = BASE_CONFIG
    out["summary"]["repeat_count"] = repeat_count

    graph_path = PROJECT_DIR / "intermediate" / f"graph_input_config{output_config_number}.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    with graph_path.open("w", encoding="utf-8") as f:
        json.dump(out, f)

    config_marker = {
        "Synthetic": True,
        "BaseConfig": BASE_CONFIG,
        "RepeatCount": repeat_count,
        "GraphInput": f"intermediate/graph_input_config{output_config_number}.json",
        "Description": f"Synthetic benchmark from config{BASE_CONFIG}, repeated {repeat_count} times"
    }

    config_path = PROJECT_DIR / "configs" / f"config{output_config_number}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_marker, f, indent=2)

    print(
        f"config{output_config_number}: x{repeat_count}, "
        f"roads={out['num_links']}, edges={len(out['col_indices'])}, "
        f"secondary={secondary_count}, primary={primary_count}"
    )


def main():
    base_path = PROJECT_DIR / "intermediate" / f"graph_input_config{BASE_CONFIG}.json"

    if not base_path.is_file():
        raise SystemExit(
            f"Missing {base_path}. First run: python export_graph_input.py {BASE_CONFIG}"
        )

    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    for config_number, repeat_count in SYNTHETIC_CONFIGS:
        make_synthetic(base, config_number, repeat_count)


if __name__ == "__main__":
    main()